"""Local usage counts: which views and actions the user actually uses, per day.

Collected only to decide, after the usage period, which pages to keep or drop
(roadmap V1.4). Stored in ``data/personal/usage.json`` on this computer; it is
never sent anywhere. Only event names and counts are kept -- no stock codes,
prices or other content.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from backend.data.atomic import atomic_write, file_lock

_EVENT = re.compile(r"(tab|pane|action):[a-z_]{1,32}")
MAX_DAYS = 400


class UsageStore:
    def __init__(self, root: Path) -> None:
        self.path = Path(root) / "personal" / "usage.json"

    def _read(self) -> dict[str, dict[str, int]]:
        if not self.path.exists():
            return {}
        data = json.loads(self.path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}

    def record(self, event: Any, *, day: Optional[str] = None) -> dict[str, Any]:
        if not isinstance(event, str) or not _EVENT.fullmatch(event):
            raise ValueError("使用记录事件名无效")
        day = day or datetime.now().strftime("%Y%m%d")
        with file_lock(self.path):
            data = self._read()
            counts = data.setdefault(day, {})
            counts[event] = int(counts.get(event, 0)) + 1
            for old in sorted(data)[:-MAX_DAYS]:
                data.pop(old, None)
            atomic_write(self.path, json.dumps(data, ensure_ascii=False, indent=1, sort_keys=True) + "\n")
        return {"day": day, "event": event, "count": counts[event]}

    def summary(self, days: int = 30, *, today: Optional[str] = None) -> dict[str, Any]:
        end = datetime.strptime(today, "%Y%m%d") if today else datetime.now()
        start = (end - timedelta(days=max(1, int(days)) - 1)).strftime("%Y%m%d")
        data = {d: c for d, c in self._read().items() if start <= d <= end.strftime("%Y%m%d")}
        totals: Counter = Counter()
        for counts in data.values():
            totals.update({k: int(v) for k, v in counts.items()})
        return {"days": int(days), "active_days": len(data), "since": start,
                "events": dict(totals.most_common()),
                "note": "只在本机记录使用次数，不联网；用于使用期结束后决定保留哪些功能。"}
