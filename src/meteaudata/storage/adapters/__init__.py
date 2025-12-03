"""Storage adapter implementations."""

from meteaudata.storage.adapters.pandas_memory import PandasMemoryAdapter
from meteaudata.storage.adapters.pandas_disk import PandasDiskAdapter

__all__ = [
    "PandasMemoryAdapter",
    "PandasDiskAdapter",
]

# Optional adapters that may not be available
try:
    from meteaudata.storage.adapters.polars_adapter import PolarsAdapter
    __all__.append("PolarsAdapter")
except ImportError:
    pass

try:
    from meteaudata.storage.adapters.sql_adapter import SQLAdapter
    __all__.append("SQLAdapter")
except ImportError:
    pass
