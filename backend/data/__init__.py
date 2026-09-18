"""A-share data foundation layer.

The public surface is deliberately small so that upper-layer strategy and
decision modules never import a vendor-specific client directly.
"""

from backend.data.config import get_settings
from backend.data.providers.base import (
    DataProvider,
    DataProviderError,
    DataSourceNotConfiguredError,
)

__all__ = [
    "DataProvider",
    "DataProviderError",
    "DataSourceNotConfiguredError",
    "get_settings",
]
