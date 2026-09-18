"""Local storage provider for ingested real data.

Reads normalized tables previously saved by ``DataPipeline`` (CSV under the
configured data root). This keeps the offline/local-storage path explicit and
separate from test fixtures.
"""

from __future__ import annotations

from backend.data.config import get_settings
from backend.data.providers.fixture import FixtureProvider


class StorageProvider(FixtureProvider):
    """Read ingested real data from the local data root."""

    name = "storage"
    data_timeliness = "historical"

    def __init__(self, base_dir=None) -> None:
        root = base_dir or get_settings().db_path.parent
        super().__init__(base_dir=root)
