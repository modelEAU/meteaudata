"""Storage backend protocols and type definitions."""

from typing import Protocol, TypeVar, Union, Any, Optional, Dict, List
from pathlib import Path
import pandas as pd

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    pl = None
    POLARS_AVAILABLE = False


# Type alias for data that can be stored
if POLARS_AVAILABLE:
    SeriesLike = Union[pd.Series, pl.Series]
else:
    SeriesLike = pd.Series


class StorageBackend(Protocol):
    """Protocol defining the interface for data storage backends.

    All storage adapters must implement this protocol to provide consistent
    data access across different storage strategies (in-memory, on-disk, database).
    """

    def save(self, data: SeriesLike, key: str, metadata: Dict[str, Any]) -> None:
        """Save data with associated metadata.

        Args:
            data: The time series data to save
            key: Unique identifier for this data
            metadata: Metadata dictionary including index info, processing steps, etc.
        """
        ...

    def load(self, key: str) -> tuple[SeriesLike, Dict[str, Any]]:
        """Load data and metadata by key.

        Args:
            key: Unique identifier for the data to load

        Returns:
            Tuple of (data, metadata)

        Raises:
            KeyError: If key does not exist
        """
        ...

    def delete(self, key: str) -> None:
        """Delete stored data by key.

        Args:
            key: Unique identifier for the data to delete

        Raises:
            KeyError: If key does not exist
        """
        ...

    def exists(self, key: str) -> bool:
        """Check if data exists for key.

        Args:
            key: Unique identifier to check

        Returns:
            True if data exists, False otherwise
        """
        ...

    def list_keys(self) -> List[str]:
        """List all stored keys.

        Returns:
            List of all keys in the storage backend
        """
        ...

    def to_pandas(self, data: SeriesLike) -> pd.Series:
        """Convert backend-specific data to pandas Series.

        Args:
            data: Backend-specific series data

        Returns:
            Pandas Series
        """
        ...

    def from_pandas(self, series: pd.Series) -> SeriesLike:
        """Convert pandas Series to backend-specific format.

        Args:
            series: Pandas Series to convert

        Returns:
            Backend-specific series data
        """
        ...

    def clear(self) -> None:
        """Clear all stored data.

        This removes all data and metadata from the backend.
        Use with caution.
        """
        ...

    def get_backend_type(self) -> str:
        """Get the type identifier for this backend.

        Returns:
            String identifier like 'pandas-memory', 'polars', 'sql', etc.
        """
        ...


class FileBasedBackend(Protocol):
    """Protocol for file-based storage backends that use a base directory.

    Extends StorageBackend with file system specific operations.
    """

    @property
    def base_path(self) -> Path:
        """Get the base path for file storage.

        Returns:
            Path to the base directory
        """
        ...

    def get_data_path(self, key: str) -> Path:
        """Get the file path for storing data.

        Args:
            key: Unique identifier

        Returns:
            Path where data file is/should be stored
        """
        ...

    def get_metadata_path(self, key: str) -> Path:
        """Get the file path for storing metadata.

        Args:
            key: Unique identifier

        Returns:
            Path where metadata file is/should be stored
        """
        ...


class DatabaseBackend(Protocol):
    """Protocol for database-based storage backends.

    Extends StorageBackend with database-specific operations.
    """

    @property
    def connection_string(self) -> str:
        """Get the database connection string.

        Returns:
            Connection string (e.g., 'sqlite:///path/to/db.db')
        """
        ...

    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Execute a custom SQL query.

        Args:
            query: SQL query string
            params: Optional query parameters

        Returns:
            Query results
        """
        ...

    def get_table_names(self) -> List[str]:
        """Get list of all tables in the database.

        Returns:
            List of table names
        """
        ...
