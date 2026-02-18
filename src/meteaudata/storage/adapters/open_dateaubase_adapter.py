"""open_dateaubase storage adapter for metEAUdata.

This adapter implements the StorageBackend protocol using the open_dateaubase
MSSQL schema (v1.6.0+) as a storage target.  It is the preferred adapter for
production environmental data workflows where full provenance tracking is needed.

The SQLiteAdapter (formerly SQLAdapter) remains available for local/offline use.

Usage::

    import pyodbc
    from meteaudata.storage.adapters.open_dateaubase_adapter import OpenDateaubaseAdapter

    conn = pyodbc.connect("DRIVER=...;SERVER=...;DATABASE=open_dateaubase;...")
    adapter = OpenDateaubaseAdapter(conn)

    # Load provenance context from the DB (no manual metadata_id needed)
    prov = adapter.get_provenance(metadata_id=42)

    # After processing a Signal, write lineage back
    adapter.write_lineage(signal, source_metadata_ids=[42], output_metadata_id=43)
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from meteaudata.types import DataProvenance, Signal


class OpenDateaubaseAdapter:
    """Storage backend for the open_dateaubase MSSQL schema.

    Implements the same ``StorageBackend`` protocol as ``SQLiteAdapter`` so that
    it can be used as a drop-in replacement inside a ``StorageContainer``.

    Requires: ``pyodbc`` and the ``open-dateaubase`` package installed in the
    same environment.

    Args:
        connection: An active ``pyodbc.Connection`` to an open_dateaubase
            MSSQL instance.  The caller owns the connection lifecycle.
    """

    def __init__(self, connection: Any) -> None:
        self._conn = connection

    # ------------------------------------------------------------------
    # StorageBackend protocol
    # ------------------------------------------------------------------

    def save(self, data: pd.Series, key: str, metadata: dict[str, Any]) -> None:
        """Write a time series to dbo.Value.

        ``key`` must be a string-encoded integer matching a ``Metadata_ID`` in
        the open_dateaubase.  Values are inserted row-by-row; callers should
        ensure the MetaData row already exists before calling this method.

        Args:
            data: Pandas Series whose index contains timestamps and whose
                values are numeric measurements.
            key: String representation of the target ``Metadata_ID``.
            metadata: Metadata dict (currently stored for reference; lineage
                recording should be done via :meth:`write_lineage`).
        """
        metadata_id = int(key)
        cursor = self._conn.cursor()
        for timestamp, value in zip(data.index, data.values):
            ts = timestamp.to_pydatetime() if hasattr(timestamp, "to_pydatetime") else timestamp
            val = float(value) if pd.notna(value) else None
            cursor.execute(
                "INSERT INTO [dbo].[Value] ([Metadata_ID], [Timestamp], [Value]) "
                "VALUES (?, ?, ?)",
                metadata_id,
                ts,
                val,
            )

    def load(self, key: str) -> tuple[pd.Series, dict[str, Any]]:
        """Read a time series from open_dateaubase by ``Metadata_ID``.

        Args:
            key: String representation of the ``Metadata_ID``.

        Returns:
            Tuple of (pandas Series, metadata dict from :func:`load_signal_context`).

        Raises:
            KeyError: If the ``Metadata_ID`` does not exist.
        """
        from open_dateaubase.meteaudata_bridge import load_signal_context

        metadata_id = int(key)
        meta = load_signal_context(metadata_id, self._conn)

        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT [Timestamp], [Value] FROM [dbo].[Value] "
            "WHERE [Metadata_ID] = ? ORDER BY [Timestamp]",
            metadata_id,
        )
        rows = cursor.fetchall()
        if not rows:
            raise KeyError(f"No values found for Metadata_ID={metadata_id}")

        timestamps = [row[0] for row in rows]
        values = [row[1] for row in rows]
        series = pd.Series(values, index=pd.DatetimeIndex(timestamps), name=str(metadata_id))
        return series, meta

    def exists(self, key: str) -> bool:
        """Return ``True`` if the ``Metadata_ID`` has at least one Value row."""
        metadata_id = int(key)
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM [dbo].[Value] WHERE [Metadata_ID] = ?",
            metadata_id,
        )
        return cursor.fetchone()[0] > 0

    def delete(self, key: str) -> None:
        """Mark a MetaData entry as logically deleted.

        Does **not** physically delete rows — open_dateaubase is an append-only
        audit log.  If a soft-delete mechanism is added to MetaData (e.g. an
        ``IsActive`` flag) this method should set it.  For now, raises
        ``NotImplementedError`` to make the no-op explicit.

        Args:
            key: String ``Metadata_ID``.
        """
        raise NotImplementedError(
            "open_dateaubase is append-only.  Logical deletion requires an "
            "IsActive column on MetaData (not yet implemented in schema)."
        )

    def list_keys(self) -> list[str]:
        """Return all ``Metadata_ID`` values that have at least one Value row."""
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT DISTINCT [Metadata_ID] FROM [dbo].[Value] ORDER BY [Metadata_ID]"
        )
        return [str(row[0]) for row in cursor.fetchall()]

    def to_pandas(self, data: pd.Series) -> pd.Series:
        """Pass-through (data is already pandas)."""
        return data

    def from_pandas(self, series: pd.Series) -> pd.Series:
        """Pass-through (data is already pandas)."""
        return series

    def clear(self) -> None:
        """Not supported — open_dateaubase is append-only."""
        raise NotImplementedError(
            "clear() is not supported for the open_dateaubase adapter."
        )

    def get_backend_type(self) -> str:
        """Return the backend type identifier."""
        return "open-dateaubase"

    # ------------------------------------------------------------------
    # open_dateaubase-specific methods
    # ------------------------------------------------------------------

    def get_provenance(self, metadata_id: int) -> "DataProvenance":
        """Load full DataProvenance from MetaData + all FK joins.

        Pulls parameter, unit, location, site, equipment, campaign, laboratory,
        analyst, project, and purpose from the database so that callers don't
        need to build ``DataProvenance`` manually.

        Both approaches coexist: ``DataProvenance`` can still be constructed
        manually by passing explicit field values.

        Args:
            metadata_id: Primary key of the MetaData row.

        Returns:
            A populated ``DataProvenance`` instance.
        """
        from open_dateaubase.meteaudata_bridge import load_signal_context
        from meteaudata.types import DataProvenance

        ctx = load_signal_context(metadata_id, self._conn)
        return DataProvenance(
            metadata_id=str(metadata_id),
            source_repository="open_dateaubase",
            project=ctx.get("project"),
            location=ctx["location"]["name"] if ctx.get("location") else None,
            equipment=ctx["equipment"]["name"] if ctx.get("equipment") else None,
            parameter=ctx["parameter"]["name"] if ctx.get("parameter") else None,
            purpose=ctx.get("purpose"),
        )

    def write_lineage(
        self,
        signal: "Signal",
        source_metadata_ids: list[int],
        output_metadata_id: int,
    ) -> None:
        """Persist processing lineage from a Signal's ``processing_steps`` history.

        For each ``ProcessingStep`` recorded in ``signal.processing_steps`` this
        method inserts a ``dbo.ProcessingStep`` row and the corresponding
        ``dbo.DataLineage`` edges (one Input per ``source_metadata_ids`` entry,
        one Output pointing to ``output_metadata_id``).

        ``output_metadata_id`` must already exist in ``dbo.MetaData``.

        Args:
            signal: The processed ``Signal`` whose history should be persisted.
            source_metadata_ids: ``Metadata_ID`` values of the source time series
                that were consumed by the processing steps.
            output_metadata_id: ``Metadata_ID`` of the result time series.
        """
        from open_dateaubase.meteaudata_bridge import record_processing

        for step in signal.processing_steps:
            params: dict = {}
            if step.parameters is not None:
                try:
                    params = step.parameters.model_dump()
                except AttributeError:
                    params = dict(step.parameters) if step.parameters else {}

            record_processing(
                source_metadata_ids=source_metadata_ids,
                method_name=step.description or "",
                method_version=None,
                processing_type=step.type.value if hasattr(step.type, "value") else str(step.type),
                parameters=params,
                executed_at=step.run_datetime if step.run_datetime else datetime.utcnow(),
                executed_by_person_id=None,
                output_metadata_id=output_metadata_id,
                conn=self._conn,
            )
