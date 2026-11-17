"""Position sizing for the user's own account.

Shares are sized from the stop distance first ("how much can I lose"), then
capped by the single-stock weight, the first-entry weight, the room left under
the market-regime exposure cap and the available cash. A-share orders are in
whole lots, and a small account pays the minimum commission on every order, so
both are part of the calculation. When not even one lot fits, the result says
which rule blocks it and what one lot would actually risk -- it never rounds
up to make a trade possible.
"""

from __future__ import annotations

import math
from typing import Any, Optional

from backend.personal.risk_profile import REGIME_LABELS, RiskProfile


def _positive(name: str, value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} 须为正数")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} 须为正数") from None
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} 须为正数")
    return number


def _non_negative(name: str, value: Any) -> float:
    if value is None or value == "":
        return 0.0
    if isinstance(value, bool):
        raise ValueError(f"{name} 须为非负数")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} 须为非负数") from None
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} 须为非负数")
    return number


def _commission(profile: RiskProfile, value: float) -> float:
    return max(profile.min_commission, value * profile.commission_rate) if value > 0 else 0.0


def round_trip_cost(profile: RiskProfile, shares: int, entry: float, exit_price: float) -> float:
    """Buy commission + sell commission + stamp tax (sell side)."""
    buy_value = shares * entry
    sell_value = shares * exit_price
    return _commission(profile, buy_value) + _commission(profile, sell_value) \
        + sell_value * profile.stamp_tax_rate


def size_position(
    profile: RiskProfile,
    entry_price: Any,
    stop_price: Any,
    *,
    risk_pct: Any = None,
    market_state: Optional[str] = None,
    current_exposure: Any = 0.0,
    existing_position: Any = 0.0,
    available_cash: Any = None,
) -> dict[str, Any]:
    entry = _positive("买入价", entry_price)
    stop = _positive("止损价", stop_price)
    if stop >= entry:
        raise ValueError("止损价必须低于买入价")
    capital = float(profile.capital)
    exposure = _non_negative("现有持仓市值", current_exposure)
    existing = _non_negative("该股已有持仓市值", existing_position)
    if existing > exposure:
        exposure = existing
    cash = None if available_cash in (None, "") else _non_negative("可用资金", available_cash)

    if risk_pct in (None, ""):
        risk = profile.risk_per_trade
    else:
        risk = _positive("单笔风险比例", risk_pct)
        if risk > profile.max_risk_per_trade:
            raise ValueError(
                f"单笔风险 {risk:.2%} 超过上限 {profile.max_risk_per_trade:.2%}"
            )

    lot = profile.lot_size
    stop_pct = (entry - stop) / entry
    risk_budget = capital * risk

    regime = profile.regime_for_state(market_state)
    exposure_cap = profile.exposure_caps[regime] if regime else None

    # Value ceilings, in yuan, for the NEW order.
    limits: list[tuple[str, str, float]] = [
        ("single_weight", f"单票上限 {profile.max_single_weight:.0%}",
         capital * profile.max_single_weight - existing),
    ]
    if existing <= 0:
        limits.append(("first_entry", f"首仓上限 {profile.first_entry_max:.0%}",
                       capital * profile.first_entry_max))
    if exposure_cap is not None:
        limits.append(("exposure", f"{REGIME_LABELS[regime]}总仓位上限 {exposure_cap:.0%}",
                       capital * exposure_cap - exposure))
    if cash is not None:
        limits.append(("cash", "可用资金", cash))

    def lot_risk(lots: int) -> float:
        shares = lots * lot
        return shares * (entry - stop) + round_trip_cost(profile, shares, entry, stop)

    def blockers(lots: int) -> list[dict[str, Any]]:
        shares = lots * lot
        value = shares * entry + _commission(profile, shares * entry)
        out = []
        loss = lot_risk(lots)
        if loss > risk_budget + 1e-9:
            out.append({"rule": "risk", "label": f"单笔风险 {risk:.2%}（{risk_budget:,.0f} 元）",
                        "limit": round(risk_budget, 2), "actual": round(loss, 2)})
        for key, label, ceiling in limits:
            if value > ceiling + 1e-9:
                out.append({"rule": key, "label": label, "limit": round(max(ceiling, 0.0), 2),
                            "actual": round(value, 2)})
        return out

    # Small accounts: a handful of lots at most, so a direct scan is exact and
    # keeps the minimum-commission step function honest.
    max_lots_by_value = int(capital // (entry * lot)) + 1
    lots = 0
    for candidate in range(1, max_lots_by_value + 1):
        if blockers(candidate):
            break
        lots = candidate

    one_lot_value = lot * entry
    one_lot_risk = lot_risk(1)
    one_lot = {
        "shares": lot,
        "value": round(one_lot_value, 2),
        "weight": round(one_lot_value / capital, 4),
        "risk_amount": round(one_lot_risk, 2),
        "risk_pct_of_capital": round(one_lot_risk / capital, 4),
    }

    warnings: list[str] = []
    if stop_pct > profile.max_stop_pct:
        warnings.append(
            f"止损距离 {stop_pct:.1%} 超过参考上限 {profile.max_stop_pct:.0%}，"
            "说明买点离支撑太远，可考虑等待更近的买点"
        )
    if regime is None:
        warnings.append("未提供市场状态，未检查总仓位上限")
    if cash is None:
        warnings.append("未提供可用资金，未检查现金是否足够")

    result: dict[str, Any] = {
        "inputs": {
            "entry_price": entry, "stop_price": stop, "risk_pct": risk,
            "market_state": market_state, "regime": regime,
            "current_exposure": exposure, "existing_position": existing,
            "available_cash": cash, "capital": capital,
        },
        "stop_pct": round(stop_pct, 4),
        "risk_budget": round(risk_budget, 2),
        "one_lot": one_lot,
        "warnings": warnings,
    }

    if lots > 0:
        shares = lots * lot
        value = shares * entry
        loss = lot_risk(lots)
        binding = blockers(lots + 1)
        result.update({
            "verdict": "可买",
            "lots": lots,
            "shares": shares,
            "value": round(value, 2),
            "weight": round(value / capital, 4),
            "risk_amount": round(loss, 2),
            "risk_pct_of_capital": round(loss / capital, 4),
            "est_round_trip_cost": round(round_trip_cost(profile, shares, entry, stop), 2),
            "limited_by": [b["label"] for b in binding] if lots < max_lots_by_value else [],
        })
        if value / capital < profile.first_entry_min and existing <= 0:
            warnings.append(
                f"仓位 {value / capital:.1%} 低于首仓下限 {profile.first_entry_min:.0%}，"
                "受风险预算限制只能小仓位"
            )
    else:
        blocked = blockers(1)
        needed = one_lot_risk / capital
        result.update({
            "verdict": "不可买",
            "lots": 0,
            "shares": 0,
            "value": 0.0,
            "weight": 0.0,
            "risk_amount": 0.0,
            "risk_pct_of_capital": 0.0,
            "blocked_by": blocked,
            "one_lot_needs_risk_pct": round(needed, 4),
            "reason": "一手（{} 股）就违反：{}".format(lot, "；".join(b["label"] for b in blocked)),
        })
    return result
