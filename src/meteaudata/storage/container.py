"""Series container with lazy loading support."""

from typing import Any, Optional, Dict, TYPE_CHECKING
import pandas as pd

if TYPE_CHECKING:
    from meteaudata.storage.protocols import StorageBackend, SeriesLike


class SeriesContainer:
    """Container for time series data with lazy loading support.

    This class wraps time series data and provides lazy loading capabilities.
    Data is only loaded from the backend when actually accessed, enabling
    memory-efficient processing of large datasets.

    Attributes:
        _backend: The storage backend to use
        _key: Unique identifier for this data in the backend
        _data: The actual data (None if not yet loaded)
        _loaded: Whether data has been loaded into memory
        _metadata: Associated metadata
    """

    def __init__(
        self,
        backend: "StorageBackend",
        key: Optional[str] = None,
        data: Optional["SeriesLike"] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the series container.

        Args:
            backend: Storage backend to use
            key: Unique identifier (required if data not provided)
            data: Optional pre-loaded data
            metadata: Optional metadata dict
        """
        if data is None and key is None:
            raise ValueError("Either 'data' or 'key' must be provided")

        self._backend = backend
        self._key = key
        self._data = data
        self._metadata = metadata or {}
        self._loaded = data is not None

    @property
    def data(self) -> "SeriesLike":
        """Get the data, loading lazily if needed.

        Returns:
            The time series data

        Raises:
            KeyError: If key doesn't exist in backend
        """
        if not self._loaded:
            if self._key is None:
                raise ValueError("Cannot load data: no key specified")
            self._data, self._metadata = self._backend.load(self._key)
            self._loaded = True
        return self._data

    @property
    def metadata(self) -> Dict[str, Any]:
        """Get the metadata, loading if needed.

        Returns:
            Metadata dictionary
        """
        if not self._metadata and not self._loaded:
            # Trigger load to get metadata
            _ = self.data
        return self._metadata

    @property
    def is_loaded(self) -> bool:
        """Check if data is currently loaded in memory.

        Returns:
            True if data is in memory, False if it needs to be loaded
        """
        return self._loaded

    @property
    def key(self) -> Optional[str]:
        """Get the storage key.

        Returns:
            The unique identifier for this data
        """
        return self._key

    def materialize(self) -> None:
        """Force load data into memory.

        This is useful when you know you'll need the data and want to
        trigger loading explicitly.
        """
        _ = self.data

    def save(self, key: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Save the data to the backend.

        Args:
            key: Optional new key (uses existing if not provided)
            metadata: Optional metadata to save (uses existing if not provided)

        Raises:
            ValueError: If no key is available
            RuntimeError: If data is not loaded
        """
        if not self._loaded:
            raise RuntimeError("Cannot save unloaded data")

        save_key = key or self._key
        if save_key is None:
            raise ValueError("No key specified for saving")

        save_metadata = metadata or self._metadata

        self._backend.save(self._data, save_key, save_metadata)
        self._key = save_key
        self._metadata = save_metadata

    def unload(self) -> None:
        """Unload data from memory.

        This frees up memory by discarding the in-memory data. The data can
        be reloaded later from the backend using the stored key.

        Raises:
            ValueError: If no key is available for reloading
        """
        if self._key is None:
            raise ValueError("Cannot unload data without a key for reloading")

        self._data = None
        self._loaded = False

    def to_pandas(self) -> pd.Series:
        """Get data as a pandas Series.

        Returns:
            Pandas Series representation of the data
        """
        return self._backend.to_pandas(self.data)

    def __repr__(self) -> str:
        """String representation.

        Returns:
            Human-readable representation
        """
        status = "loaded" if self._loaded else "not loaded"
        key_info = f"key='{self._key}'" if self._key else "no key"
        return f"SeriesContainer({key_info}, {status})"
