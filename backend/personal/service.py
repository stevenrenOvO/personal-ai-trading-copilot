"""Daily loop for the user's own account: watchlist status, account, pre-trade check.

Everything here reads; nothing places or records a trade on its own.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

import pandas as pd

from backend.personal.adjusted import AdjustedBarStore
from backend.personal.band_rules import BUY, BELOW_MA, NO_DATA, WEEKLY_FAIL, BandParams, evaluate
from backend.personal.plans import PlanStore
from backend.personal.real_account import RealTradeStore
from backend.personal.risk_profile import REGIME_LABELS, RiskProfileStore
from backend.personal.sizing import size_position
from backend.personal.watchlist import WatchlistStore


def _history(runtime):
    return getattr(getattr(runtime, "provider", None), "history", None)


def evaluation_day(runtime) -> Optional[str]:
    """Latest session whose daily bar is final (today after the close)."""
    try:
        from backend.market.session import get_trading_session

        info = get_trading_session().describe_cached()
        if info.get("expected_daily_date"):
            return str(info["expected_daily_date"])
    except Exception:  # noqa: BLE001
        pass
    history = _history(runtime)
    if history is not None:
        try:
            return history.coverage().get("end")
        except Exception:  # noqa: BLE001
            return None
    return None


def _week_complete(runtime, last_date: str) -> Optional[bool]:
    """True when no open session follows ``last_date`` in the same ISO week."""
    try:
        day = datetime.strptime(last_date, "%Y%m%d")
        week = day.isocalendar()[:2]
        cal = runtime.provider.trade_cal(start_date=last_date, end_date=(day + pd.Timedelta(days=7)).strftime("%Y%m%d"),
                                         is_open=True)
        later = [str(d) for d in cal["cal_date"] if str(d) > last_date]
        if not later:
            return None  # calendar does not reach far enough: let the rules fall back
        nxt = datetime.strptime(min(later), "%Y%m%d")
        return nxt.isocalendar()[:2] != week
    except Exception:  # noqa: BLE001
        return None


def _raw_close(runtime, code: str, date: Optional[str]) -> Optional[float]:
    history = _history(runtime)
    if history is None or not date:
        return None
    try:
        rows = history.load(ts_codes=[code], start_date=date, end_date=date)
    except Exception:  # noqa: BLE001
        return None
    return float(rows["close"].iloc[0]) if not rows.empty else None


def stock_status(runtime, code: str, *, as_of: Optional[str] = None, position: Optional[dict] = None,
                 store: Optional[AdjustedBarStore] = None, params: BandParams = BandParams()) -> dict[str, Any]:
    as_of = as_of or evaluation_day(runtime)
    if not as_of:
        return {"ts_code": code, "entry": NO_DATA, "reason": "交易日历不可用"}
    store = store or AdjustedBarStore(runtime.root)
    bars, basis = store.bars(code, as_of, _history(runtime))
    pos = None
    if position and position.get("shares"):
        pos = {"entry_price": position["avg_cost"], "entry_date": position["entry_date"],
               "entry_close_raw": _raw_close(runtime, code, position["entry_date"])}
    if bars.empty:
        result = {"entry": NO_DATA, "reason": "没有可用日线", "conditions": []}
    else:
        last = str(bars["trade_date"].iloc[-1])
        result = evaluate(bars, params=params, last_week_complete=_week_complete(runtime, last), position=pos)
        if last < as_of:
            basis.setdefault("warnings", []).append(f"最新日线为 {last}，晚于应有的 {as_of}，状态可能滞后")
    result.update(ts_code=code, evaluation_day=as_of, basis=basis)
    return result


def watchlist_status(runtime) -> dict[str, Any]:
    as_of = evaluation_day(runtime)
    profile, _ = RiskProfileStore(runtime.root).load()
    account = RealTradeStore(runtime.root).account(profile)
    held = {p["ts_code"]: p for p in account["positions"]}
    store = AdjustedBarStore(runtime.root)
    entries = WatchlistStore(runtime.root).list()
    if as_of:
        store.prefetch([entry["ts_code"] for entry in entries], as_of)
    items = []
    for entry in entries:
        status = stock_status(runtime, entry["ts_code"], as_of=as_of, position=held.get(entry["ts_code"]),
                              store=store)
        status.update(name=entry.get("name", ""), note=entry.get("note", ""), held=entry["ts_code"] in held)
        items.append(status)
    return {"evaluation_day": as_of, "items": items, "params": BandParams().to_dict()}


# -- prices / market state -----------------------------------------------------

def _snapshot(runtime):
    try:
        from backend.api.data_status import snapshot_available
        from backend.market.quote_service import get_quote_service

        frame, meta = get_quote_service().get()
        if snapshot_available(runtime, frame, meta, meta.quote_date):
            return frame, meta
    except Exception:  # noqa: BLE001
        pass
    return None, None


def latest_prices(runtime, codes: list[str]) -> dict[str, dict[str, Any]]:
    """Latest price per code: today's snapshot when present, else the last stored close."""
    out: dict[str, dict[str, Any]] = {}
    frame, meta = _snapshot(runtime)
    if frame is not None:
        rows = frame[frame["ts_code"].isin(codes)]
        for row in rows.itertuples(index=False):
            if row.price and row.price > 0:
                out[row.ts_code] = {"price": float(row.price), "date": meta.quote_date,
                                    "source": f"{meta.source} 行情快照 {meta.quote_time or ''}".strip()}
    history = _history(runtime)
    missing = [c for c in codes if c not in out]
    if history is not None and missing:
        try:
            rows = history.load(ts_codes=missing)
        except Exception:  # noqa: BLE001
            rows = pd.DataFrame()
        if not rows.empty:
            for code, group in rows.sort_values("trade_date").groupby("ts_code"):
                last = group.iloc[-1]
                out[code] = {"price": float(last["close"]), "date": str(last["trade_date"]), "source": "本地日线收盘"}
    return out


def current_market_state(runtime) -> dict[str, Any]:
    frame, meta = _snapshot(runtime)
    if frame is not None:
        try:
            from backend.market.snapshot import compute_full_market_state

            state = compute_full_market_state(frame, meta.quote_date)
            return {"state": state.state, "date": meta.quote_date, "source": "全市场行情快照"}
        except Exception:  # noqa: BLE001
            pass
    return {"state": None, "date": None, "source": None}


def account_view(runtime) -> dict[str, Any]:
    profile, source = RiskProfileStore(runtime.root).load()
    account = RealTradeStore(runtime.root).account(profile)
    codes = [p["ts_code"] for p in account["positions"]]
    prices = latest_prices(runtime, codes)
    store = AdjustedBarStore(runtime.root)
    as_of = evaluation_day(runtime)
    if as_of and codes:
        store.prefetch(codes, as_of)
    exposure = 0.0
    for pos in account["positions"]:
        quote = prices.get(pos["ts_code"])
        if quote:
            value = quote["price"] * pos["shares"]
            exposure += value
            pos.update(price=quote["price"], price_date=quote["date"], price_source=quote["source"],
                       market_value=round(value, 2), unrealized_pnl=round(value - pos["cost"], 2),
                       weight=round(value / profile.capital, 4))
        else:
            pos.update(price=None, market_value=None, unrealized_pnl=None, weight=None)
        status = stock_status(runtime, pos["ts_code"], as_of=as_of, position=pos, store=store)
        pos["exit"] = status.get("exit")
        pos["status_basis"] = status.get("basis")
    account.update(capital=profile.capital, profile_source=source, exposure=round(exposure, 2),
                   exposure_pct=round(exposure / profile.capital, 4),
                   unpriced=[p["ts_code"] for p in account["positions"] if p.get("price") is None])
    return account


def limit_rate(ts_code: str, name: str = "") -> float:
    """Daily price-limit ratio by board (main board ST stocks: 5%)."""
    symbol = ts_code.split(".")[0]
    if ts_code.endswith(".BJ"):
        return 0.30
    if symbol.startswith(("30", "688")):
        return 0.20
    return 0.05 if "ST" in str(name).upper() else 0.10


def limit_checks(code: str, row: Optional[dict], meta: Any, entry_price: Any) -> list[tuple[str, str]]:
    """(level, detail) items about price limits; empty when nothing to flag.

    Intraday the limit is today's (from the previous close). After the close
    a planned entry is for the next session, whose limit comes from today's close.
    """
    if not row:
        return []
    try:
        entry = float(entry_price) if entry_price not in (None, "") else None
        price, pre_close = float(row["price"]), float(row["pre_close"])
    except (TypeError, ValueError, KeyError):
        return []
    if price <= 0 or pre_close <= 0:
        return []
    rate = limit_rate(code, row.get("name", ""))
    final = getattr(meta, "status", None) == "final"
    out: list[tuple[str, str]] = []
    today_up = round(pre_close * (1 + rate), 2)
    if price >= today_up:
        out.append(("warn", f"今日{'收于' if final else '当前封'}涨停 {today_up:.2f}，"
                            f"{'下一交易日可能高开或一字板，难按计划价成交' if final else '排队买入很可能无法成交'}"))
    base, label = (price, "下一交易日") if final else (pre_close, "今日")
    up = round(base * (1 + rate), 2)
    down = round(base * (1 - rate), 2)
    if entry is not None and entry > up:
        out.append(("fail", f"买入价 {entry:.2f} 高于{label}涨停价 {up:.2f}，无法成交"))
    elif entry is not None and entry < down:
        out.append(("fail", f"买入价 {entry:.2f} 低于{label}跌停价 {down:.2f}，无法成交"))
    return out


# -- pre-trade check -------------------------------------------------------------

def _latest_plan(store: PlanStore, today: str) -> Optional[dict[str, Any]]:
    dates = [d for d in store.dates() if d <= today]
    return store.get(dates[0]) if dates else None


def pretrade_check(runtime, query: dict[str, Any]) -> dict[str, Any]:
    from backend.personal.real_account import _code

    code = _code(query.get("ts_code"))
    profile, profile_source = RiskProfileStore(runtime.root).load()
    account = account_view(runtime)
    held = {p["ts_code"]: p for p in account["positions"]}
    status = stock_status(runtime, code, position=held.get(code))
    checks: list[dict[str, Any]] = []

    def add(key, label, level, detail):
        checks.append({"key": key, "label": label, "level": level, "detail": detail})

    # 1. band rules
    entry = status.get("entry")
    if entry == BUY:
        add("rules", "波段规则", "pass", entry)
    elif entry in (WEEKLY_FAIL, BELOW_MA, NO_DATA):
        add("rules", "波段规则", "fail", f"{entry}：{status.get('reason') or '不满足买入前提'}")
    else:
        add("rules", "波段规则", "warn", f"{entry}：买点条件尚未全部满足")

    # 2. watchlist / plan discipline
    watch = {item["ts_code"] for item in WatchlistStore(runtime.root).list()}
    add("watchlist", "观察池", "pass" if code in watch else "warn",
        "在观察池中" if code in watch else "不在观察池中")
    today = datetime.now().strftime("%Y%m%d")
    plan = _latest_plan(PlanStore(runtime.root), today)
    planned = plan and any(c["ts_code"] == code for c in plan["candidates"])
    add("plan", "计划内", "pass" if planned else "warn",
        f"在 {plan['date']} 的计划中" if planned else
        (f"不在最近的计划（{plan['date']}）中，属于计划外交易" if plan else "还没有保存过计划"))

    # 3. sizing with real exposure and cash
    ma = status.get("ma") or {}
    prices = latest_prices(runtime, [code])
    entry_price = query.get("entry") or (prices.get(code) or {}).get("price")
    stop_price = query.get("stop") or ma.get("suggested_stop")
    market = {"state": query.get("market_state"), "source": "手动选择"} if query.get("market_state") \
        else current_market_state(runtime)
    sizing = None
    if entry_price and stop_price:
        try:
            existing = (held.get(code) or {}).get("market_value") or 0.0
            sizing = size_position(profile, entry_price, stop_price, risk_pct=query.get("risk_pct"),
                                   market_state=market.get("state"), current_exposure=account["exposure"],
                                   existing_position=existing, available_cash=account["cash"])
        except ValueError as exc:
            add("sizing", "仓位", "fail", str(exc))
        else:
            if sizing["lots"] > 0:
                add("sizing", "仓位", "pass",
                    f"规则允许 {sizing['lots']} 手（{sizing['shares']} 股），约 {sizing['value']:,.0f} 元，"
                    f"止损亏损约 {sizing['risk_amount']:,.0f} 元")
            else:
                add("sizing", "仓位", "fail", sizing["reason"])
    else:
        add("sizing", "仓位", "fail", "缺少买入价或止损价（21日线不可用时请手动填写止损价）")

    # 4. total exposure room
    regime = profile.regime_for_state(market.get("state"))
    if regime:
        cap = profile.exposure_caps[regime]
        room = profile.capital * cap - account["exposure"]
        add("exposure", "总仓位余量", "pass" if room > 0 else "fail",
            f"{REGIME_LABELS[regime]}上限 {cap:.0%}，已用 {account['exposure_pct']:.1%}，余量约 {max(room, 0):,.0f} 元")
    else:
        add("exposure", "总仓位余量", "warn", "市场状态未知，未检查总仓位上限")

    # 5. price limits (needs today's snapshot)
    snap, snap_meta = _snapshot(runtime)
    row = None
    if snap is not None:
        match = snap[snap["ts_code"] == code]
        row = match.iloc[0].to_dict() if not match.empty else None
    limits = limit_checks(code, row, snap_meta, entry_price)
    if limits:
        for level, detail in limits:
            add("limit", "涨跌停", level, detail)
    elif row is not None:
        add("limit", "涨跌停", "pass", "买入价在涨跌停范围内，未封涨停")

    if account["unpriced"]:
        add("pricing", "持仓估值", "warn", f"{', '.join(account['unpriced'])} 没有最新价，总仓位可能低估")

    levels = [c["level"] for c in checks]
    verdict = "不通过" if "fail" in levels else "有警告" if "warn" in levels else "通过"
    return {"ts_code": code, "verdict": verdict, "checks": checks,
            "inputs": {"entry_price": entry_price, "stop_price": stop_price,
                       "stop_source": "手动填写" if query.get("stop") else "21日线下方 1%"},
            "market": market, "sizing": sizing, "status": status, "profile_source": profile_source,
            "account": {k: account[k] for k in ("cash", "exposure", "exposure_pct", "capital")},
            "note": "检查只说明是否符合你自己的规则，不构成买卖建议；是否下单由你决定。"}
