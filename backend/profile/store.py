"""JSONL store for user profile."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backend.profile.models import UserProfile


class ProfileStore:
    def __init__(self, root: Path) -> None:
        self.path = Path(root) / "profile" / "profile.jsonl"

    def save(self, profile: UserProfile) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(profile.to_dict(), ensure_ascii=False) + "\n")

    def load(self) -> UserProfile | None:
        if not self.path.exists():
            return None
        with self.path.open("r", encoding="utf-8") as handle:
            line = handle.readline()
            if not line:
                return None
            data = json.loads(line)
            return UserProfile.from_dict(data)
