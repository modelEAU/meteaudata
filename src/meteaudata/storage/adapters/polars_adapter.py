"""Polars storage adapter.

This adapter uses Polars for data storage and processing, providing
better performance for certain operations while maintaining compatibility
with the pandas-based system through conversion utilities.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Union, TYPE_CHECKING
import pandas as pd
import yaml
import tempfile
import shutil

from meteaudata.types import IndexMetadata

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False
    pl = None  # type: ignore

if TYPE_CHECKING:
    import polars as pl


class PolarsAdapter:
    """Storage adapter using Polars for data storage.

    This adapter stores data using Polars Series and Parquet files,
    providing better performance for certain operations. It automatically
    converts between pandas and Polars as needed.

    Attributes:
        base_path: Base directory for storing data files

    Raises:
        ImportError: If Polars is not installed
    """

    def __init__(self, base_path: Optional[Path] = None):
        """Initialize the Polars storage adapter.

        Args:
            base_path: Directory to store files. If None, creates a temporary directory.

        Raises:
            ImportError: If Polars is not installed
        """
        if not POLARS_AVAILABLE:
            raise ImportError(
                "Polars is required for PolarsAdapter. "
                "Install it with: pip install polars"
            )

        if base_path is None:
            base_path = Path(tempfile.mkdtemp(prefix="meteaudata_polars_"))

        self._base_path = Path(base_path)
        self._base_path.mkdir(parents=True, exist_ok=True)

    @property
    def base_path(self) -> Path:
        """Get the base path for file storage.

        Returns:
            Path to the base directory
        """
        return self._base_path

    def get_data_path(self, key: str) -> Path:
        """Get the file path for storing data.

        Args:
            key: Unique identifier

        Returns:
            Path to the Parquet data file
        """
        safe_key = key.replace("/", "_").replace("\\", "_")
        return self._base_path / f"{safe_key}.parquet"

    def get_metadata_path(self, key: str) -> Path:
        """Get the file path for storing metadata.

        Args:
            key: Unique identifier

        Returns:
            Path to the YAML metadata file
        """
        safe_key = key.replace("/", "_").replace("\\", "_")
        return self._base_path / f"{safe_key}_meta.yaml"

    def save(self, data: Union[pl.Series, pd.Series], key: str, metadata: Dict[str, Any]) -> None:
        """Save data and metadata to disk.

        Args:
            data: Polars or Pandas Series to store
            key: Unique identifier
            metadata: Metadata dictionary
        """
        data_path = self.get_data_path(key)
        meta_path = self.get_metadata_path(key)

        # Convert pandas to Polars if needed
        if isinstance(data, pd.Series):
            # Extract and store index metadata before conversion
            index_metadata = IndexMetadata.extract_index_metadata(data.index)
            metadata['index_metadata'] = index_metadata.model_dump()

            # Store the actual index values for reconstruction
            # Convert to string for YAML serialization (works for all index types)
            metadata['index_values'] = [str(val) for val in data.index]

            # Store series name in metadata before conversion
            if 'original_name' not in metadata and data.name is not None:
                metadata['original_name'] = data.name

            # Convert to Polars
            pl_series = pl.from_pandas(data)
        else:
            pl_series = data

        # Convert Series to DataFrame for Parquet storage
        series_name = pl_series.name or metadata.get('original_name', 'value')
        df = pl.DataFrame({series_name: pl_series})

        # Save data as Parquet
        df.write_parquet(data_path, compression='snappy')

        # Save metadata as YAML
        with open(meta_path, 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False, allow_unicode=True)

    def load(self, key: str) -> Tuple[pl.Series, Dict[str, Any]]:
        """Load data and metadata from disk.

        Args:
            key: Unique identifier

        Returns:
            Tuple of (Polars series, metadata)

        Raises:
            KeyError: If key does not exist
        """
        data_path = self.get_data_path(key)
        meta_path = self.get_metadata_path(key)

        if not data_path.exists():
            raise KeyError(f"Key '{key}' not found in storage (path: {data_path})")

        # Load data from Parquet
        df = pl.read_parquet(data_path)

        # Convert to Series (take first column)
        series = df.to_series(0)

        # Load metadata from YAML
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                metadata = yaml.safe_load(f) or {}
        else:
            metadata = {}

        # Note: Index values and metadata are stored separately and will be
        # reconstructed when converting to pandas via to_pandas()
        # Polars doesn't have the same index concept, so we store it in metadata

        return series, metadata

    def delete(self, key: str) -> None:
        """Delete data and metadata files from disk.

        Args:
            key: Unique identifier

        Raises:
            KeyError: If key does not exist
        """
        data_path = self.get_data_path(key)
        meta_path = self.get_metadata_path(key)

        if not data_path.exists():
            raise KeyError(f"Key '{key}' not found in storage")

        # Delete both data and metadata files
        data_path.unlink()
        if meta_path.exists():
            meta_path.unlink()

    def exists(self, key: str) -> bool:
        """Check if key exists in storage.

        Args:
            key: Unique identifier

        Returns:
            True if data file exists, False otherwise
        """
        return self.get_data_path(key).exists()

    def list_keys(self) -> List[str]:
        """List all stored keys.

        Returns:
            List of all keys in storage
        """
        keys = []
        for parquet_file in self._base_path.glob("*.parquet"):
            # Extract key from filename (remove .parquet extension)
            key = parquet_file.stem
            keys.append(key)
        return sorted(keys)

    def to_pandas(self, data: Union[pl.Series, pd.Series]) -> pd.Series:
        """Convert to pandas Series.

        Note: This method only handles the data conversion. Index metadata
        reconstruction must be done separately after loading, as the metadata
        is not part of the Polars series itself.

        Args:
            data: Polars or Pandas Series

        Returns:
            Pandas Series
        """
        if isinstance(data, pd.Series):
            return data
        return data.to_pandas()

    def from_pandas(self, series: pd.Series) -> pl.Series:
        """Convert from pandas Series to Polars Series.

        Args:
            series: Pandas Series

        Returns:
            Polars Series
        """
        return pl.from_pandas(series)

    def clear(self) -> None:
        """Clear all stored data from disk.

        Warning:
            This operation is irreversible and removes all files.
        """
        for file_path in self._base_path.glob("*.parquet"):
            file_path.unlink()
        for file_path in self._base_path.glob("*.yaml"):
            file_path.unlink()

    def cleanup(self) -> None:
        """Remove the entire storage directory.

        Warning:
            This is a destructive operation that removes the base directory
            and all its contents. Use with caution.
        """
        if self._base_path.exists():
            shutil.rmtree(self._base_path)

    def get_backend_type(self) -> str:
        """Get backend type identifier.

        Returns:
            String 'polars'
        """
        return "polars"

    def get_size_on_disk(self) -> int:
        """Get total size of stored data in bytes.

        Returns:
            Total size in bytes
        """
        total_size = 0
        for file_path in self._base_path.glob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size

    def __len__(self) -> int:
        """Get number of stored items.

        Returns:
            Number of keys in storage
        """
        return len(self.list_keys())

    def __repr__(self) -> str:
        """String representation.

        Returns:
            Human-readable representation
        """
        return f"PolarsAdapter(base_path='{self._base_path}', keys={len(self)})"
