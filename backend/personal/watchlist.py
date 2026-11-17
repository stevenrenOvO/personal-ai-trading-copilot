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
    """Watchlist items; an item may carry ``holding`` when the user owns it.

    Holdings live here instead of a full trade ledger: the user marks a stock
    as held (cost, buy date, optional shares) when buying and as sold when
    selling. Prices update by themselves, so nothing needs daily syncing with
    the broker. Sold holdings move to ``closed`` and are kept for review.
    """

    def __init__(self, root: Path) -> None:
        self.path = Path(root) / "personal" / "watchlist.json"

    def _read_all(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"items": [], "closed": []}
        data = json.loads(self.path.read_text(encoding="utf-8"))
        items = data.get("items") if isinstance(data, dict) else None
        closed = data.get("closed", []) if isinstance(data, dict) else None
        if not isinstance(items, list) or not isinstance(closed, list):
            raise ValueError("观察池文件格式错误")
        return {"items": items, "closed": closed}

    def _read(self) -> list[dict[str, Any]]:
        return self._read_all()["items"]

    def _write(self, items: list[dict[str, Any]], closed: list[dict[str, Any]] | None = None) -> None:
        if closed is None:
            closed = self._read_all()["closed"] if self.path.exists() else []
        atomic_write(self.path, json.dumps({"items": items, "closed": closed}, ensure_ascii=False,
                                           allow_nan=False, indent=2) + "\n")

    def list(self) -> list[dict[str, Any]]:
        return list(self._read())

    def closed(self) -> list[dict[str, Any]]:
        return list(self._read_all()["closed"])

    def holdings(self) -> list[dict[str, Any]]:
        return [item for item in self._read() if isinstance(item.get("holding"), dict)]

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
                raise ValueError(f"{entry['name'] or ts_code} 已在观察池中")
            if len(items) >= WATCHLIST_LIMIT:
                raise ValueError(f"观察池最多 {WATCHLIST_LIMIT} 只，请先移除一只")
            items.append(entry)
            self._write(items)
        return entry

    def remove(self, code: Any) -> bool:
        ts_code = _normalize_code(code)
        with file_lock(self.path):
            items = self._read()
            target = next((item for item in items if item.get("ts_code") == ts_code), None)
            if target is None:
                return False
            if isinstance(target.get("holding"), dict):
                raise ValueError("这只股票还标记为持有，请先标记卖出再移除")
            self._write([item for item in items if item is not target])
        return True

    def hold(self, code: Any, *, cost: Any, entry_date: Any, shares: Any = None, name: Any = None,
             today: str | None = None) -> dict[str, Any]:
        """Mark (or update) a holding; adds the stock to the watchlist if needed."""
        ts_code = _normalize_code(code)
        holding = {"cost": _positive_price(cost, "成本价"),
                   "entry_date": _day(entry_date, "买入日期", today),
                   "shares": _optional_shares(shares),
                   "marked_at": datetime.now().isoformat(timespec="seconds")}
        with file_lock(self.path):
            items = self._read()
            item = next((i for i in items if i.get("ts_code") == ts_code), None)
            if item is None:
                if len(items) >= WATCHLIST_LIMIT:
                    raise ValueError(f"观察池最多 {WATCHLIST_LIMIT} 只，请先移除一只")
                item = {"ts_code": ts_code, "name": _clean_text("name", name, 40), "note": "",
                        "added_at": holding["marked_at"]}
                items.append(item)
            elif name and not item.get("name"):
                item["name"] = _clean_text("name", name, 40)
            item["holding"] = holding
            self._write(items)
        return item

    def sell(self, code: Any, *, exit_date: Any, exit_price: Any = None, note: Any = None,
             today: str | None = None) -> dict[str, Any]:
        """Close a holding; the stock stays in the watchlist as observed."""
        ts_code = _normalize_code(code)
        exit_day = _day(exit_date, "卖出日期", today)
        price = None if exit_price in (None, "") else _positive_price(exit_price, "卖出价")
        with file_lock(self.path):
            data = self._read_all()
            item = next((i for i in data["items"] if i.get("ts_code") == ts_code), None)
            if item is None or not isinstance(item.get("holding"), dict):
                raise ValueError("这只股票没有标记为持有")
            holding = item.pop("holding")
            if exit_day < holding["entry_date"]:
                item["holding"] = holding
                raise ValueError("卖出日期不能早于买入日期")
            record = {"ts_code": ts_code, "name": item.get("name", ""), **holding, "exit_date": exit_day,
                      "exit_price": price, "note": _clean_text("note", note, 300),
                      "return_pct": round(price / holding["cost"] - 1, 4) if price else None,
                      "closed_at": datetime.now().isoformat(timespec="seconds")}
            data["closed"].append(record)
            self._write(data["items"], data["closed"])
        return record


def _positive_price(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label}须为正数") from None
    if isinstance(value, bool) or not number > 0 or number != number or number == float("inf"):
        raise ValueError(f"{label}须为正数")
    return round(number, 3)


def _optional_shares(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError("股数须为正整数") from None
    if isinstance(value, bool) or not number.is_integer() or number <= 0:
        raise ValueError("股数须为正整数")
    return int(number)


def _day(value: Any, label: str, today: str | None = None) -> str:
    text = str(value or "").replace("-", "").strip()
    try:
        datetime.strptime(text, "%Y%m%d")
    except ValueError:
        raise ValueError(f"{label}须为 YYYYMMDD") from None
    if text > (today or datetime.now().strftime("%Y%m%d")):
        raise ValueError(f"{label}不能是未来日期")
    return text
