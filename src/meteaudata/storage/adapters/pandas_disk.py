"""On-disk pandas storage adapter using Parquet files.

This adapter stores data on disk using the efficient Parquet format,
enabling processing of datasets larger than available memory.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import pandas as pd
import yaml
import tempfile
import shutil

from meteaudata.types import IndexMetadata


class PandasDiskAdapter:
    """On-disk storage adapter using pandas and Parquet format.

    This adapter stores time series data as Parquet files and metadata as
    YAML files in a specified directory. Data is only loaded into memory
    when accessed, enabling memory-efficient processing of large datasets.

    Attributes:
        base_path: Base directory for storing data files
    """

    def __init__(self, base_path: Optional[Path] = None):
        """Initialize the on-disk storage adapter.

        Args:
            base_path: Directory to store files. If None, creates a temporary directory.
        """
        if base_path is None:
            base_path = Path(tempfile.mkdtemp(prefix="meteaudata_"))

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
        # Sanitize key for filesystem
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

    def save(self, data: pd.Series, key: str, metadata: Dict[str, Any]) -> None:
        """Save data and metadata to disk.

        Args:
            data: Pandas Series to store
            key: Unique identifier
            metadata: Metadata dictionary
        """
        data_path = self.get_data_path(key)
        meta_path = self.get_metadata_path(key)

        # Extract and store index metadata
        index_metadata = IndexMetadata.extract_index_metadata(data.index)
        metadata['index_metadata'] = index_metadata.model_dump()

        # Convert Series to DataFrame for Parquet storage
        # Preserve series name and index
        df = data.to_frame()

        # Save data as Parquet
        df.to_parquet(data_path, engine='pyarrow', compression='snappy')

        # Save metadata as YAML
        with open(meta_path, 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False, allow_unicode=True)

    def load(self, key: str) -> Tuple[pd.Series, Dict[str, Any]]:
        """Load data and metadata from disk.

        Args:
            key: Unique identifier

        Returns:
            Tuple of (series, metadata)

        Raises:
            KeyError: If key does not exist
        """
        data_path = self.get_data_path(key)
        meta_path = self.get_metadata_path(key)

        if not data_path.exists():
            raise KeyError(f"Key '{key}' not found in storage (path: {data_path})")

        # Load data from Parquet
        df = pd.read_parquet(data_path, engine='pyarrow')

        # Convert back to Series
        series = df.iloc[:, 0]

        # Load metadata from YAML
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                metadata = yaml.safe_load(f) or {}
        else:
            metadata = {}

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
            String 'pandas-disk'
        """
        return "pandas-disk"

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
        return f"PandasDiskAdapter(base_path='{self._base_path}', keys={len(self)})"
