"""SQL database storage adapter.

This adapter stores time series data in a SQL database, enabling
SQL-based querying and filtering while supporting memory-efficient
processing of large datasets.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import pandas as pd
import json
import tempfile

from meteaudata.types import IndexMetadata

try:
    from sqlalchemy import create_engine, text, Engine
    from sqlalchemy.pool import StaticPool
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    create_engine = None
    text = None
    Engine = None
    StaticPool = None


class SQLAdapter:
    """Storage adapter using SQL database.

    This adapter stores time series data in a SQL database using SQLAlchemy.
    Supports SQLite, PostgreSQL, MySQL, and other SQLAlchemy-compatible databases.

    The adapter creates two tables:
    - time_series_data: Stores the actual time series values
    - time_series_metadata: Stores metadata as JSON

    Attributes:
        engine: SQLAlchemy engine for database operations
        connection_string: Database connection string

    Raises:
        ImportError: If SQLAlchemy is not installed
    """

    def __init__(self, connection_string: Optional[str] = None):
        """Initialize the SQL storage adapter.

        Args:
            connection_string: Database URL (e.g., 'sqlite:///path/to/db.db')
                If None, creates a temporary SQLite database.

        Raises:
            ImportError: If SQLAlchemy is not installed
        """
        if not SQLALCHEMY_AVAILABLE:
            raise ImportError(
                "SQLAlchemy is required for SQLAdapter. "
                "Install it with: pip install sqlalchemy"
            )

        if connection_string is None:
            temp_dir = Path(tempfile.mkdtemp(prefix="meteaudata_sql_"))
            db_path = temp_dir / "meteaudata.db"
            connection_string = f"sqlite:///{db_path}"

        self._connection_string = connection_string

        # For SQLite in-memory databases, use StaticPool to maintain connection
        if connection_string.startswith('sqlite:///:memory:'):
            self._engine = create_engine(
                connection_string,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool
            )
        else:
            self._engine = create_engine(connection_string)

        self._ensure_schema()

    @property
    def connection_string(self) -> str:
        """Get the database connection string.

        Returns:
            Connection string
        """
        return self._connection_string

    @property
    def engine(self) -> Engine:
        """Get the SQLAlchemy engine.

        Returns:
            SQLAlchemy engine
        """
        return self._engine

    def _ensure_schema(self) -> None:
        """Create database tables if they don't exist."""
        with self._engine.connect() as conn:
            # Create time series data table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS time_series_data (
                    key TEXT NOT NULL,
                    idx INTEGER NOT NULL,
                    index_value TEXT NOT NULL,
                    value REAL,
                    PRIMARY KEY (key, idx)
                )
            """))

            # Create index for faster lookups
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_key
                ON time_series_data(key)
            """))

            # Create metadata table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS time_series_metadata (
                    key TEXT PRIMARY KEY,
                    metadata_json TEXT NOT NULL,
                    series_name TEXT,
                    length INTEGER,
                    dtype TEXT
                )
            """))

            conn.commit()

    def save(self, data: pd.Series, key: str, metadata: Dict[str, Any]) -> None:
        """Save data and metadata to the database.

        Args:
            data: Pandas Series to store
            key: Unique identifier
            metadata: Metadata dictionary
        """
        # Extract and store index metadata
        index_metadata = IndexMetadata.extract_index_metadata(data.index)
        metadata['index_metadata'] = index_metadata.model_dump()

        with self._engine.connect() as conn:
            # Delete existing data for this key
            conn.execute(
                text("DELETE FROM time_series_data WHERE key = :key"),
                {"key": key}
            )
            conn.execute(
                text("DELETE FROM time_series_metadata WHERE key = :key"),
                {"key": key}
            )

            # Prepare data for insertion
            records = []
            for idx, (index_val, value) in enumerate(zip(data.index, data.values)):
                records.append({
                    'key': key,
                    'idx': idx,
                    'index_value': str(index_val),
                    'value': float(value) if pd.notna(value) else None
                })

            # Insert data in batches
            if records:
                # Use pandas for efficient bulk insert
                df = pd.DataFrame(records)
                df.to_sql(
                    'time_series_data',
                    conn,
                    if_exists='append',
                    index=False,
                    method='multi'
                )

            # Insert metadata
            metadata_json = json.dumps(metadata, default=str)
            conn.execute(
                text("""
                    INSERT INTO time_series_metadata
                    (key, metadata_json, series_name, length, dtype)
                    VALUES (:key, :metadata_json, :series_name, :length, :dtype)
                """),
                {
                    "key": key,
                    "metadata_json": metadata_json,
                    "series_name": str(data.name) if data.name is not None else None,
                    "length": len(data),
                    "dtype": str(data.dtype)
                }
            )

            conn.commit()

    def load(self, key: str) -> Tuple[pd.Series, Dict[str, Any]]:
        """Load data and metadata from the database.

        Args:
            key: Unique identifier

        Returns:
            Tuple of (series, metadata)

        Raises:
            KeyError: If key does not exist
        """
        with self._engine.connect() as conn:
            # Check if key exists
            result = conn.execute(
                text("SELECT COUNT(*) FROM time_series_metadata WHERE key = :key"),
                {"key": key}
            )
            if result.scalar() == 0:
                raise KeyError(f"Key '{key}' not found in storage")

            # Load metadata
            result = conn.execute(
                text("SELECT metadata_json, series_name FROM time_series_metadata WHERE key = :key"),
                {"key": key}
            )
            row = result.fetchone()
            metadata = json.loads(row[0])
            series_name = row[1]

            # Load data
            query = """
                SELECT index_value, value
                FROM time_series_data
                WHERE key = :key
                ORDER BY idx
            """
            df = pd.read_sql(text(query), conn, params={"key": key})

            # Reconstruct series
            if len(df) == 0:
                # Empty series
                series = pd.Series([], dtype=float, name=series_name)
            else:
                series = pd.Series(
                    data=df['value'].values,
                    index=df['index_value'].values,
                    name=series_name
                )

            # Reconstruct index with metadata if available
            if 'index_metadata' in metadata:
                index_meta = IndexMetadata(**metadata['index_metadata'])
                reconstructed_index = IndexMetadata.reconstruct_index(
                    series.index,
                    index_meta
                )
                series.index = reconstructed_index

            return series, metadata

    def delete(self, key: str) -> None:
        """Delete data and metadata from the database.

        Args:
            key: Unique identifier

        Raises:
            KeyError: If key does not exist
        """
        with self._engine.connect() as conn:
            # Check if key exists
            result = conn.execute(
                text("SELECT COUNT(*) FROM time_series_metadata WHERE key = :key"),
                {"key": key}
            )
            if result.scalar() == 0:
                raise KeyError(f"Key '{key}' not found in storage")

            # Delete data and metadata
            conn.execute(
                text("DELETE FROM time_series_data WHERE key = :key"),
                {"key": key}
            )
            conn.execute(
                text("DELETE FROM time_series_metadata WHERE key = :key"),
                {"key": key}
            )
            conn.commit()

    def exists(self, key: str) -> bool:
        """Check if key exists in storage.

        Args:
            key: Unique identifier

        Returns:
            True if key exists, False otherwise
        """
        with self._engine.connect() as conn:
            result = conn.execute(
                text("SELECT COUNT(*) FROM time_series_metadata WHERE key = :key"),
                {"key": key}
            )
            return result.scalar() > 0

    def list_keys(self) -> List[str]:
        """List all stored keys.

        Returns:
            List of all keys in storage
        """
        with self._engine.connect() as conn:
            result = conn.execute(
                text("SELECT key FROM time_series_metadata ORDER BY key")
            )
            return [row[0] for row in result.fetchall()]

    def to_pandas(self, data: pd.Series) -> pd.Series:
        """Convert to pandas Series (no-op for this adapter).

        Args:
            data: Pandas Series

        Returns:
            The same pandas Series
        """
        return data

    def from_pandas(self, series: pd.Series) -> pd.Series:
        """Convert from pandas Series (no-op for this adapter).

        Args:
            series: Pandas Series

        Returns:
            The same pandas Series
        """
        return series

    def clear(self) -> None:
        """Clear all stored data from the database.

        Warning:
            This operation is irreversible and removes all data.
        """
        with self._engine.connect() as conn:
            conn.execute(text("DELETE FROM time_series_data"))
            conn.execute(text("DELETE FROM time_series_metadata"))
            conn.commit()

    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Execute a custom SQL query.

        Args:
            query: SQL query string
            params: Optional query parameters

        Returns:
            Query results
        """
        with self._engine.connect() as conn:
            result = conn.execute(text(query), params or {})
            return result.fetchall()

    def get_table_names(self) -> List[str]:
        """Get list of all tables in the database.

        Returns:
            List of table names
        """
        # This is database-specific; using SQLAlchemy's inspector
        from sqlalchemy import inspect
        inspector = inspect(self._engine)
        return inspector.get_table_names()

    def get_backend_type(self) -> str:
        """Get backend type identifier.

        Returns:
            String 'sql'
        """
        return "sql"

    def get_database_size(self) -> Optional[int]:
        """Get the size of the database in bytes.

        Returns:
            Database size in bytes, or None if not available
        """
        if self._connection_string.startswith('sqlite:///'):
            # Extract file path from connection string
            db_path = self._connection_string.replace('sqlite:///', '')
            path = Path(db_path)
            if path.exists():
                return path.stat().st_size
        return None

    def close(self) -> None:
        """Close the database connection."""
        self._engine.dispose()

    def __len__(self) -> int:
        """Get number of stored items.

        Returns:
            Number of keys in storage
        """
        with self._engine.connect() as conn:
            result = conn.execute(
                text("SELECT COUNT(*) FROM time_series_metadata")
            )
            return result.scalar()

    def __repr__(self) -> str:
        """String representation.

        Returns:
            Human-readable representation
        """
        return f"SQLAdapter(connection='{self._connection_string}', keys={len(self)})"

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.close()
        except Exception:
            pass  # Ignore errors during cleanup
