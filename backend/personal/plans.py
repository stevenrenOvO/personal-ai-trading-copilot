"""After-close daily plan: at most 3 new candidates plus what to do with holdings.

A plan is saved under the date it was made (the session just reviewed) and is
meant for the next session. Saving again on the same date keeps the earlier
version in ``revisions`` so a later review can see whether the plan was
changed after the fact.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from backend.data.atomic import atomic_write, file_lock
from backend.personal.real_account import _code, _date, _positive, _text

MAX_NEW_CANDIDATES = 3
HOLDING_ACTIONS = ("持有", "减仓", "离场", "加仓")


def _optional_price(value: Any, name: str) -> Optional[float]:
    return None if value in (None, "") else round(_positive(value, name), 3)


def _candidate(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise ValueError("候选须为对象")
    entry = _optional_price(item.get("entry_price"), "计划买入价")
    stop = _optional_price(item.get("stop_price"), "止损价")
    if entry is not None and stop is not None and stop >= entry:
        raise ValueError(f"{item.get('ts_code')} 止损价须低于计划买入价")
    shares = item.get("shares")
    if shares in (None, ""):
        shares = None
    elif isinstance(shares, bool) or not isinstance(shares, (int, float)) or not float(shares).is_integer() \
            or shares <= 0:
        raise ValueError("计划股数须为正整数")
    else:
        shares = int(shares)
    return {"ts_code": _code(item.get("ts_code")), "name": _text(item.get("name"), "名称", 40),
            "entry_condition": _text(item.get("entry_condition"), "买入条件", 200),
            "entry_price": entry, "stop_price": stop, "shares": shares,
            "status_at_plan": _text(item.get("status_at_plan"), "当时状态", 40),
            "reason": _text(item.get("reason"), "理由")}


def _holding(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise ValueError("持仓计划须为对象")
    action = item.get("action")
    if action not in HOLDING_ACTIONS:
        raise ValueError(f"持仓动作须为 {' / '.join(HOLDING_ACTIONS)}")
    return {"ts_code": _code(item.get("ts_code")), "action": action,
            "stop_price": _optional_price(item.get("stop_price"), "止损价"),
            "note": _text(item.get("note"), "说明", 200)}


class PlanStore:
    def __init__(self, root: Path) -> None:
        self.path = Path(root) / "personal" / "plans.json"

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("plans"), dict):
            raise ValueError("计划文件格式错误")
        return data["plans"]

    def get(self, date: Any) -> Optional[dict[str, Any]]:
        return self._read().get(_date(date, "计划日期"))

    def dates(self) -> list[str]:
        return sorted(self._read(), reverse=True)

    def save(self, data: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(data, dict):
            raise ValueError("请求体必须为 JSON 对象")
        date = _date(data.get("date"), "计划日期")
        candidates = data.get("candidates") or []
        holdings = data.get("holdings") or []
        if not isinstance(candidates, list) or not isinstance(holdings, list):
            raise ValueError("candidates / holdings 须为列表")
        if len(candidates) > MAX_NEW_CANDIDATES:
            raise ValueError(f"新候选最多 {MAX_NEW_CANDIDATES} 只")
        clean = [_candidate(item) for item in candidates]
        codes = [c["ts_code"] for c in clean]
        if len(set(codes)) != len(codes):
            raise ValueError("候选中有重复股票")
        plan = {"date": date, "candidates": clean, "holdings": [_holding(item) for item in holdings],
                "market_note": _text(data.get("market_note"), "市场判断", 300),
                "note": _text(data.get("note"), "备注")}
        now = datetime.now().isoformat(timespec="seconds")
        with file_lock(self.path):
            plans = self._read()
            previous = plans.get(date)
            if previous:
                history = previous.pop("revisions", [])
                plan.update(created_at=previous["created_at"], revision=previous["revision"] + 1,
                            revisions=history + [previous])
            else:
                plan.update(created_at=now, revision=1, revisions=[])
            plan["updated_at"] = now
            plans[date] = plan
            atomic_write(self.path, json.dumps({"plans": plans}, ensure_ascii=False,
                                               allow_nan=False, indent=2) + "\n")
        return plan
