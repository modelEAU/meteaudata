"""Factory for creating storage backends."""

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from meteaudata.storage.protocols import StorageBackend
    from meteaudata.storage.config import StorageConfig


def create_backend(config: "StorageConfig") -> "StorageBackend":
    """Factory function to create a storage backend from configuration.

    Args:
        config: Storage configuration specifying backend type and parameters

    Returns:
        Initialized storage backend

    Raises:
        ValueError: If backend type is unknown or configuration is invalid
        ImportError: If required dependencies are not installed
    """
    # Validate configuration
    config.validate_config()

    backend_type = config.backend_type

    if backend_type == "pandas-memory":
        from meteaudata.storage.adapters.pandas_memory import PandasMemoryAdapter
        return PandasMemoryAdapter()

    elif backend_type == "pandas-disk":
        from meteaudata.storage.adapters.pandas_disk import PandasDiskAdapter
        return PandasDiskAdapter(base_path=config.base_path)

    elif backend_type == "polars":
        try:
            from meteaudata.storage.adapters.polars_adapter import PolarsAdapter
        except ImportError as e:
            raise ImportError(
                "Polars backend requires the 'polars' package. "
                "Install it with: pip install polars"
            ) from e
        return PolarsAdapter(base_path=config.base_path)

    elif backend_type == "sql":
        try:
            from meteaudata.storage.adapters.sql_adapter import SQLAdapter
        except ImportError as e:
            raise ImportError(
                "SQL backend requires 'sqlalchemy'. "
                "Install it with: pip install sqlalchemy"
            ) from e
        return SQLAdapter(connection_string=config.connection_string)

    else:
        raise ValueError(f"Unknown backend type: {backend_type}")
