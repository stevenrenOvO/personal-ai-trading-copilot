"""Tick-offs for the execution list: done or skipped, per basis date.

The list is rebuilt from the rules each day; what the user actually did with
each line is recorded here, keyed by the close it was based on. This is much
lighter than a trade ledger and is what later discipline statistics count.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from backend.data.atomic import atomic_write, file_lock

STATUSES = ("done", "skipped")
_KEY = re.compile(r"[a-z_]+:(\d{6}\.(SH|SZ|BJ)|plan)")


class ExecutionStore:
    def __init__(self, root: Path) -> None:
        self.path = Path(root) / "personal" / "executions.json"

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("执行记录文件格式错误")
        return data

    def get(self, basis: Optional[str]) -> dict[str, Any]:
        return dict(self._read().get(str(basis), {})) if basis else {}

    def mark(self, basis: Any, key: Any, status: Any, note: Any = None) -> dict[str, Any]:
        basis = str(basis or "").strip()
        try:
            datetime.strptime(basis, "%Y%m%d")
        except ValueError:
            raise ValueError("依据日期须为 YYYYMMDD") from None
        if not isinstance(key, str) or not _KEY.fullmatch(key):
            raise ValueError("事项标识无效")
        if status not in STATUSES and status is not None:
            raise ValueError("状态须为 done（已执行）或 skipped（放弃）")
        if note is not None and (not isinstance(note, str) or len(note) > 200):
            raise ValueError("说明须为不超过 200 字的文本")
        with file_lock(self.path):
            data = self._read()
            day = data.setdefault(basis, {})
            if status is None:
                day.pop(key, None)  # undo a tick
                record = {"key": key, "status": None}
            else:
                record = {"key": key, "status": status, "note": (note or "").strip(),
                          "at": datetime.now().isoformat(timespec="seconds")}
                day[key] = record
            atomic_write(self.path, json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
        return record
