"""Personal watchlist: at most ``WATCHLIST_LIMIT`` stocks the user is following.

Kept small on purpose. Each name is tracked against the user's own band rules,
so a long list would dilute attention. Stored as JSON under
``<data root>/personal/watchlist.json``, separate from the research paper
account.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from backend.data.atomic import atomic_write, file_lock

WATCHLIST_LIMIT = 10
_TS_CODE_RE = re.compile(r"\d{6}\.(SH|SZ|BJ)")


def _normalize_code(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("请填写有效股票代码，如 000001.SZ")
    code = value.strip().upper()
    if not _TS_CODE_RE.fullmatch(code):
        raise ValueError("请填写有效股票代码，如 000001.SZ")
    return code


def _clean_text(name: str, value: Any, limit: int) -> str:
    if value is None:
        return ""
    if not isinstance(value, str) or len(value) > limit:
        raise ValueError(f"{name} 必须为不超过 {limit} 字的文本")
    return value.strip()


class WatchlistStore:
    def __init__(self, root: Path) -> None:
        self.path = Path(root) / "personal" / "watchlist.json"

    def _read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        data = json.loads(self.path.read_text(encoding="utf-8"))
        items = data.get("items") if isinstance(data, dict) else None
        if not isinstance(items, list):
            raise ValueError("观察池文件格式错误")
        return items

    def _write(self, items: list[dict[str, Any]]) -> None:
        atomic_write(self.path, json.dumps({"items": items}, ensure_ascii=False,
                                           allow_nan=False, indent=2) + "\n")

    def list(self) -> list[dict[str, Any]]:
        return list(self._read())

    def add(self, code: Any, name: Any = None, note: Any = None) -> dict[str, Any]:
        ts_code = _normalize_code(code)
        entry = {
            "ts_code": ts_code,
            "name": _clean_text("name", name, 40),
            "note": _clean_text("note", note, 500),
            "added_at": datetime.now().isoformat(timespec="seconds"),
        }
        with file_lock(self.path):
            items = self._read()
            if any(item.get("ts_code") == ts_code for item in items):
                raise ValueError(f"{ts_code} 已在观察池中")
            if len(items) >= WATCHLIST_LIMIT:
                raise ValueError(f"观察池最多 {WATCHLIST_LIMIT} 只，请先移除一只")
            items.append(entry)
            self._write(items)
        return entry

    def remove(self, code: Any) -> bool:
        ts_code = _normalize_code(code)
        with file_lock(self.path):
            items = self._read()
            kept = [item for item in items if item.get("ts_code") != ts_code]
            if len(kept) == len(items):
                return False
            self._write(kept)
        return True
