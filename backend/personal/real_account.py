"""The user's real trades and positions, separate from the paper account.

Trades are an append-only ledger: a wrong entry is *voided* with a reason,
never deleted or edited, so every position can be traced to the records that
built it. Positions and cash are always recomputed from the ledger.

Cash starts from the risk profile's ``capital``. Fees default to the profile's
commission (with the minimum per order) plus stamp tax on sells, and can be
overridden with the broker's actual figure.
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from backend.data.atomic import atomic_write, file_lock
from backend.personal.risk_profile import RiskProfile

_CODE = re.compile(r"\d{6}\.(SH|SZ|BJ)")
_DATE = re.compile(r"\d{8}")


def _code(value: Any) -> str:
    if not isinstance(value, str) or not _CODE.fullmatch(value.strip().upper()):
        raise ValueError("请填写有效股票代码，如 000001.SZ")
    return value.strip().upper()


def _date(value: Any, name: str = "交易日期") -> str:
    text = str(value or "").replace("-", "").strip()
    if not _DATE.fullmatch(text):
        raise ValueError(f"{name}须为 YYYYMMDD")
    try:
        datetime.strptime(text, "%Y%m%d")
    except ValueError:
        raise ValueError(f"{name}无效") from None
    return text


def _positive(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name}须为正数")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name}须为正数") from None
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name}须为正数")
    return number


def _text(value: Any, name: str, limit: int = 500) -> str:
    if value is None:
        return ""
    if not isinstance(value, str) or len(value) > limit:
        raise ValueError(f"{name}须为不超过 {limit} 字的文本")
    return value.strip()


def default_fee(profile: RiskProfile, side: str, amount: float) -> float:
    commission = max(profile.min_commission, amount * profile.commission_rate)
    stamp = amount * profile.stamp_tax_rate if side == "SELL" else 0.0
    return round(commission + stamp, 2)


def replay(trades: list[dict[str, Any]], capital: float) -> dict[str, Any]:
    """Positions, cash and realised P&L from the non-voided ledger, in order."""
    positions: dict[str, dict[str, Any]] = {}
    cash = float(capital)
    realized = 0.0
    for trade in sorted((t for t in trades if not t.get("voided")),
                        key=lambda t: (t["trade_date"], t["seq"])):
        code = trade["ts_code"]
        pos = positions.setdefault(code, {"ts_code": code, "name": trade.get("name", ""), "shares": 0,
                                          "cost": 0.0, "realized_pnl": 0.0, "entry_date": None})
        if trade.get("name"):
            pos["name"] = trade["name"]
        if trade["side"] == "BUY":
            if pos["shares"] == 0:
                pos["entry_date"] = trade["trade_date"]
            pos["shares"] += trade["shares"]
            pos["cost"] += trade["amount"] + trade["fee"]
            cash -= trade["amount"] + trade["fee"]
        else:
            avg = pos["cost"] / pos["shares"] if pos["shares"] else 0.0
            released = avg * trade["shares"]
            pnl = trade["amount"] - trade["fee"] - released
            pos["shares"] -= trade["shares"]
            pos["cost"] -= released
            pos["realized_pnl"] += pnl
            realized += pnl
            cash += trade["amount"] - trade["fee"]
            if pos["shares"] == 0:
                pos["cost"] = 0.0
                pos["entry_date"] = None
    held = []
    for pos in positions.values():
        pos["realized_pnl"] = round(pos["realized_pnl"], 2)
        if pos["shares"] > 0:
            pos["cost"] = round(pos["cost"], 2)
            pos["avg_cost"] = round(pos["cost"] / pos["shares"], 4)
            held.append(pos)
    return {"positions": sorted(held, key=lambda p: p["ts_code"]),
            "closed": sorted((p for p in positions.values() if p["shares"] == 0), key=lambda p: p["ts_code"]),
            "cash": round(cash, 2), "realized_pnl": round(realized, 2)}


class RealTradeStore:
    def __init__(self, root: Path) -> None:
        self.path = Path(root) / "personal" / "real_trades.json"

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"trades": [], "next_seq": 1}
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("trades"), list):
            raise ValueError("真实交易记录文件格式错误")
        return data

    def _write(self, data: dict[str, Any]) -> None:
        atomic_write(self.path, json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2) + "\n")

    def list(self, *, date: Optional[str] = None, include_voided: bool = True) -> list[dict[str, Any]]:
        trades = self._read()["trades"]
        if date:
            trades = [t for t in trades if t["trade_date"] == _date(date)]
        if not include_voided:
            trades = [t for t in trades if not t.get("voided")]
        return sorted(trades, key=lambda t: (t["trade_date"], t["seq"]))

    def record(self, data: dict[str, Any], profile: RiskProfile, *, today: Optional[str] = None) -> dict[str, Any]:
        if not isinstance(data, dict):
            raise ValueError("请求体必须为 JSON 对象")
        side = str(data.get("side", "")).upper()
        if side not in ("BUY", "SELL"):
            raise ValueError("方向须为 BUY 或 SELL")
        trade_date = _date(data.get("trade_date"))
        today = today or datetime.now().strftime("%Y%m%d")
        if trade_date > today:
            raise ValueError("不能记录未来日期的成交")
        code = _code(data.get("ts_code"))
        price = _positive(data.get("price"), "成交价")
        shares_raw = data.get("shares")
        if isinstance(shares_raw, bool) or not isinstance(shares_raw, (int, float)) \
                or not float(shares_raw).is_integer() or shares_raw <= 0:
            raise ValueError("成交数量须为正整数（股）")
        shares = int(shares_raw)
        if side == "BUY" and shares % profile.lot_size:
            raise ValueError(f"买入数量须为 {profile.lot_size} 股的整数倍")
        amount = round(price * shares, 2)
        fee = data.get("fee")
        fee = default_fee(profile, side, amount) if fee in (None, "") else round(_nonneg(fee, "费用"), 2)
        plan_date = data.get("plan_date")
        trade = {
            "trade_date": trade_date, "ts_code": code, "name": _text(data.get("name"), "名称", 40),
            "side": side, "price": price, "shares": shares, "amount": amount, "fee": fee,
            "fee_source": "entered" if data.get("fee") not in (None, "") else "estimated",
            "reason": _text(data.get("reason"), "理由"),
            "plan_date": _date(plan_date, "计划日期") if plan_date else None,
            "followed_plan": bool(data.get("followed_plan")) if data.get("followed_plan") is not None else None,
            "created_at": datetime.now().isoformat(timespec="seconds"), "voided": False,
        }
        with file_lock(self.path):
            ledger = self._read()
            trade["seq"] = int(ledger.get("next_seq", 1))
            trade["id"] = f"R{trade_date}-{trade['seq']:04d}"
            if side == "SELL":
                held = _held_on(ledger["trades"], code, trade_date)
                if shares > held:
                    raise ValueError(f"{code} 在 {trade_date} 持有 {held} 股，不能卖出 {shares} 股")
            ledger["trades"].append(trade)
            ledger["next_seq"] = trade["seq"] + 1
            # Re-check the whole ledger: a back-dated sell must not make a later
            # sell oversold.
            _assert_never_short(ledger["trades"])
            self._write(ledger)
        return trade

    def void(self, trade_id: Any, reason: Any) -> dict[str, Any]:
        reason = _text(reason, "作废原因")
        if not reason:
            raise ValueError("作废须填写原因")
        with file_lock(self.path):
            ledger = self._read()
            match = next((t for t in ledger["trades"] if t.get("id") == trade_id), None)
            if match is None:
                raise KeyError("没有该成交记录")
            if match.get("voided"):
                raise ValueError("该记录已作废")
            match.update(voided=True, void_reason=reason, voided_at=datetime.now().isoformat(timespec="seconds"))
            _assert_never_short(ledger["trades"])  # voiding a buy must not orphan a sell
            self._write(ledger)
        return match

    def account(self, profile: RiskProfile) -> dict[str, Any]:
        return replay(self._read()["trades"], profile.capital)


def _nonneg(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name}须为非负数")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name}须为非负数") from None
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name}须为非负数")
    return number


def _held_on(trades: list[dict[str, Any]], code: str, date: str) -> int:
    held = 0
    for t in trades:
        if t.get("voided") or t["ts_code"] != code or t["trade_date"] > date:
            continue
        held += t["shares"] if t["side"] == "BUY" else -t["shares"]
    return held


def _assert_never_short(trades: list[dict[str, Any]]) -> None:
    held: dict[str, int] = {}
    for t in sorted((t for t in trades if not t.get("voided")), key=lambda t: (t["trade_date"], t["seq"])):
        held[t["ts_code"]] = held.get(t["ts_code"], 0) + (t["shares"] if t["side"] == "BUY" else -t["shares"])
        if held[t["ts_code"]] < 0:
            raise ValueError(f"{t['ts_code']} 在 {t['trade_date']} 卖出数量超过当时持仓")
