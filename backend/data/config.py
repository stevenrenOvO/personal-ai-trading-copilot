from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"

@dataclass(frozen=True)
class Settings:
    tushare_token: str | None
    db_path: Path
    fixture_dir: Path

    @property
    def has_tushare_token(self) -> bool:
        token = (self.tushare_token or "").strip()
        return bool(token and token != "your_token_here")

@lru_cache(maxsize=1)
def get_settings(env_path: Path | None = None) -> Settings:
    load_dotenv(DEFAULT_ENV_PATH if env_path is None else env_path, override=False)
    db_path = Path(
        os.getenv("COPILOT_DB_PATH", str(PROJECT_ROOT / "data" / "market.db"))
    ).resolve()
    fixture_dir = Path(
        os.getenv(
            "COPILOT_FIXTURE_DIR",
            str(PROJECT_ROOT / "tests" / "fixtures"),
        )
    ).resolve()
    return Settings(
        tushare_token=os.getenv("TUSHARE_TOKEN"),
        db_path=db_path,
        fixture_dir=fixture_dir,
    )

def clear_settings_cache() -> None:
    get_settings.cache_clear()
