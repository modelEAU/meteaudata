"""Storage abstraction layer for meteaudata.

This module provides a flexible storage backend system that supports multiple
data storage strategies including in-memory pandas, on-disk storage, Polars,
and SQL databases.
"""

from meteaudata.storage.protocols import StorageBackend, SeriesLike
from meteaudata.storage.container import SeriesContainer
from meteaudata.storage.config import StorageConfig, BackendType
from meteaudata.storage.factory import create_backend

__all__ = [
    "StorageBackend",
    "SeriesLike",
    "SeriesContainer",
    "StorageConfig",
    "BackendType",
    "create_backend",
]
