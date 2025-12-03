"""Storage backend configuration."""

from pathlib import Path
from typing import Literal, Optional, Union
from pydantic import BaseModel, Field


BackendType = Literal["pandas-memory", "pandas-disk", "polars", "sql"]


class StorageConfig(BaseModel):
    """Configuration for storage backend.

    This class defines the configuration options for different storage backends
    and provides validation of configuration parameters.

    Attributes:
        backend_type: Type of backend to use
        base_path: Base directory for file-based storage (optional)
        connection_string: Database connection string for SQL backends (optional)
        auto_save: Whether to automatically save after modifications (default: True)
        lazy_load: Whether to use lazy loading (default: True)
    """

    backend_type: BackendType = Field(
        default="pandas-memory",
        description="Type of storage backend to use"
    )

    base_path: Optional[Path] = Field(
        default=None,
        description="Base directory for file-based storage (pandas-disk, polars)"
    )

    connection_string: Optional[str] = Field(
        default=None,
        description="Database connection string for SQL backend"
    )

    auto_save: bool = Field(
        default=True,
        description="Automatically save data after modifications"
    )

    lazy_load: bool = Field(
        default=True,
        description="Use lazy loading to minimize memory usage"
    )

    model_config = {
        "frozen": False,
        "extra": "forbid",
    }

    def validate_config(self) -> None:
        """Validate configuration based on backend type.

        Raises:
            ValueError: If configuration is invalid for the selected backend
        """
        if self.backend_type in ("pandas-disk", "polars"):
            if self.base_path is None:
                raise ValueError(
                    f"base_path is required for backend_type='{self.backend_type}'"
                )

        if self.backend_type == "sql":
            if self.connection_string is None:
                raise ValueError(
                    "connection_string is required for backend_type='sql'"
                )

    @classmethod
    def for_pandas_memory(cls) -> "StorageConfig":
        """Create configuration for in-memory pandas storage.

        Returns:
            StorageConfig configured for pandas in-memory backend
        """
        return cls(backend_type="pandas-memory")

    @classmethod
    def for_pandas_disk(cls, base_path: Union[Path, str]) -> "StorageConfig":
        """Create configuration for on-disk pandas storage.

        Args:
            base_path: Directory to store data files

        Returns:
            StorageConfig configured for pandas on-disk backend
        """
        if isinstance(base_path, str):
            base_path = Path(base_path)
        return cls(backend_type="pandas-disk", base_path=base_path)

    @classmethod
    def for_polars(cls, base_path: Union[Path, str]) -> "StorageConfig":
        """Create configuration for Polars storage.

        Args:
            base_path: Directory to store data files

        Returns:
            StorageConfig configured for Polars backend
        """
        if isinstance(base_path, str):
            base_path = Path(base_path)
        return cls(backend_type="polars", base_path=base_path)

    @classmethod
    def for_sql(cls, connection_string: str) -> "StorageConfig":
        """Create configuration for SQL database storage.

        Args:
            connection_string: Database connection string
                Examples:
                - SQLite: 'sqlite:///path/to/db.db'
                - PostgreSQL: 'postgresql://user:pass@localhost/dbname'
                - MySQL: 'mysql://user:pass@localhost/dbname'

        Returns:
            StorageConfig configured for SQL backend
        """
        return cls(backend_type="sql", connection_string=connection_string)

    @classmethod
    def for_sql_temp(cls) -> "StorageConfig":
        """Create configuration for temporary SQLite database.

        Creates a SQLite database in a temporary directory. Useful for
        testing or temporary large dataset processing.

        Returns:
            StorageConfig configured for temporary SQLite backend
        """
        import tempfile
        temp_dir = Path(tempfile.mkdtemp())
        db_path = temp_dir / "meteaudata_temp.db"
        return cls.for_sql(f"sqlite:///{db_path}")
