"""In-memory pandas storage adapter.

This adapter maintains the current behavior of storing all data in memory
using pandas Series. It provides the baseline implementation and is the
default backend for backward compatibility.
"""

from typing import Any, Dict, List, Tuple
import pandas as pd
import copy


class PandasMemoryAdapter:
    """In-memory storage adapter using pandas Series.

    This adapter stores all data in a dictionary in memory, providing fast
    access but no memory optimization for large datasets. This is the default
    backend and maintains current system behavior.

    Attributes:
        _store: Dictionary mapping keys to (series, metadata) tuples
    """

    def __init__(self):
        """Initialize the in-memory storage."""
        self._store: Dict[str, Tuple[pd.Series, Dict[str, Any]]] = {}

    def save(self, data: pd.Series, key: str, metadata: Dict[str, Any]) -> None:
        """Save data with metadata to memory.

        Args:
            data: Pandas Series to store
            key: Unique identifier
            metadata: Metadata dictionary

        Note:
            Makes a copy of both data and metadata to prevent external modifications
        """
        # Make copies to prevent external modifications
        self._store[key] = (data.copy(), copy.deepcopy(metadata))

    def load(self, key: str) -> Tuple[pd.Series, Dict[str, Any]]:
        """Load data and metadata from memory.

        Args:
            key: Unique identifier

        Returns:
            Tuple of (series, metadata)

        Raises:
            KeyError: If key does not exist
        """
        if key not in self._store:
            raise KeyError(f"Key '{key}' not found in storage")

        series, metadata = self._store[key]
        # Return copies to prevent external modifications
        return series.copy(), copy.deepcopy(metadata)

    def delete(self, key: str) -> None:
        """Delete data from memory.

        Args:
            key: Unique identifier

        Raises:
            KeyError: If key does not exist
        """
        if key not in self._store:
            raise KeyError(f"Key '{key}' not found in storage")

        del self._store[key]

    def exists(self, key: str) -> bool:
        """Check if key exists in storage.

        Args:
            key: Unique identifier

        Returns:
            True if key exists, False otherwise
        """
        return key in self._store

    def list_keys(self) -> List[str]:
        """List all stored keys.

        Returns:
            List of all keys in storage
        """
        return list(self._store.keys())

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
        """Clear all stored data from memory.

        Warning:
            This operation is irreversible and removes all data.
        """
        self._store.clear()

    def get_backend_type(self) -> str:
        """Get backend type identifier.

        Returns:
            String 'pandas-memory'
        """
        return "pandas-memory"

    def __len__(self) -> int:
        """Get number of stored items.

        Returns:
            Number of keys in storage
        """
        return len(self._store)

    def __repr__(self) -> str:
        """String representation.

        Returns:
            Human-readable representation
        """
        return f"PandasMemoryAdapter(keys={len(self._store)})"
