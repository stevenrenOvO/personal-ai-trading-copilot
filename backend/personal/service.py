"""Daily loop for the user's own account: watchlist status, account, pre-trade check.

Everything here reads; nothing places or records a trade on its own.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Optional

import pandas as pd

from backend.personal.adjusted import AdjustedBarStore
from backend.personal.executions import ExecutionStore
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
    if position and position.get("avg_cost") and position.get("entry_date"):
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


def current_positions(runtime, profile=None) -> list[dict[str, Any]]:
    """What the user holds: holdings marked in the watchlist (plus any old ledger positions)."""
    out: dict[str, dict[str, Any]] = {}
    for item in WatchlistStore(runtime.root).holdings():
        h = item["holding"]
        out[item["ts_code"]] = {"ts_code": item["ts_code"], "name": item.get("name", ""), "shares": h.get("shares"),
                                "avg_cost": h["cost"], "entry_date": h["entry_date"], "source": "watchlist"}
    try:
        ledger = RealTradeStore(runtime.root).account(profile or RiskProfileStore(runtime.root).load()[0])
    except Exception:  # noqa: BLE001 - an unreadable old ledger must not hide marked holdings
        ledger = {"positions": []}
    for pos in ledger["positions"]:
        out.setdefault(pos["ts_code"], {"ts_code": pos["ts_code"], "name": pos.get("name", ""),
                                        "shares": pos["shares"], "avg_cost": pos["avg_cost"],
                                        "entry_date": pos["entry_date"], "source": "ledger"})
    return list(out.values())


def watchlist_status(runtime) -> dict[str, Any]:
    as_of = evaluation_day(runtime)
    held = {p["ts_code"]: p for p in current_positions(runtime)}
    store = AdjustedBarStore(runtime.root)
    entries = WatchlistStore(runtime.root).list()
    if as_of:
        store.prefetch([entry["ts_code"] for entry in entries], as_of)
    items = []
    for entry in entries:
        status = stock_status(runtime, entry["ts_code"], as_of=as_of, position=held.get(entry["ts_code"]),
                              store=store)
        status.update(name=entry.get("name", ""), note=entry.get("note", ""), held=entry["ts_code"] in held,
                      holding=entry.get("holding"))
        word, level, fact = watch_word(status)
        status.update(word=word, word_level=level, fact=fact)
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


def market_state_for(runtime, day: Optional[str]) -> dict[str, Any]:
    """Market state as of the close of ``day`` -- never an intraday reading.

    Taken from the closing full-market snapshot of that day and remembered in
    the history store, so the same day always gives the same answer.
    """
    empty = {"state": None, "date": day, "source": None}
    history = _history(runtime)
    if not day:
        return empty
    key = f"market_state:{day}"
    if history is not None:
        try:
            saved = history.get_state(key)
            if saved:
                return json.loads(saved)
        except Exception:  # noqa: BLE001
            pass
    from backend.market.snapshot import compute_full_market_state

    frame, meta = _snapshot(runtime)
    source = "收盘全市场行情快照"
    if frame is None or str(meta.quote_date) != str(day) or getattr(meta, "status", None) != "final":
        # Not that day's closing snapshot (e.g. opened mid-session the next
        # day): rebuild the same reading from that day's stored daily bars.
        frame, source = _bars_as_snapshot(history, day), "本地日线（收盘）"
        if frame is None:
            return empty
    try:
        state = compute_full_market_state(frame, day)
    except Exception:  # noqa: BLE001
        return empty
    record = {"state": state.state, "date": str(day), "source": source}
    if history is not None:
        try:
            history.set_state(key, json.dumps(record, ensure_ascii=False))
        except Exception:  # noqa: BLE001
            pass
    return record


def _bars_as_snapshot(history, day: str):
    """One day's stored bars in the snapshot column layout, or None if too few."""
    if history is None:
        return None
    try:
        bars = history.load(start_date=day, end_date=day)
    except Exception:  # noqa: BLE001
        return None
    if bars is None or len(bars) < 1000:
        return None
    return bars.rename(columns={"close": "price", "vol": "volume"})


def current_market_state(runtime) -> dict[str, Any]:
    return market_state_for(runtime, evaluation_day(runtime))


def account_view(runtime) -> dict[str, Any]:
    """Holdings valued at the evaluation-day close; cash estimated from cost."""
    profile, source = RiskProfileStore(runtime.root).load()
    positions = current_positions(runtime, profile)
    store = AdjustedBarStore(runtime.root)
    as_of = evaluation_day(runtime)
    if as_of and positions:
        store.prefetch([p["ts_code"] for p in positions], as_of)
    exposure = invested = 0.0
    for pos in positions:
        status = stock_status(runtime, pos["ts_code"], as_of=as_of, position=pos, store=store)
        ex = status.get("exit") or {}
        close = ex.get("close") or (status.get("ma") or {}).get("close")
        pos.update(exit=ex, status_basis=status.get("basis"), close=close, price=close, price_date=as_of,
                   price_source=f"{as_of} 收盘" if as_of else None, market_value=None, weight=None,
                   unrealized_pnl=None)
        if pos.get("shares"):
            invested += pos["avg_cost"] * pos["shares"]
            if close:
                value = close * pos["shares"]
                exposure += value
                pos.update(market_value=round(value, 2), weight=round(value / profile.capital, 4),
                           unrealized_pnl=round(value - pos["avg_cost"] * pos["shares"], 2))
    return {"positions": positions, "capital": profile.capital, "profile_source": source,
            "cash": round(profile.capital - invested, 2), "cash_estimated": True,
            "exposure": round(exposure, 2), "exposure_pct": round(exposure / profile.capital, 4),
            "price_basis": as_of,
            "unpriced": [p["ts_code"] for p in positions if p.get("shares") and not p.get("close")],
            "unknown_shares": [p["ts_code"] for p in positions if not p.get("shares")]}


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

def next_day_limit(close: Any, profile) -> Optional[float]:
    """Highest price worth paying next session: last close plus the chase limit."""
    try:
        value = float(close)
    except (TypeError, ValueError):
        return None
    return round(value * (1 + profile.chase_limit), 2) if value > 0 else None


def _latest_plan(store: PlanStore, today: str) -> Optional[dict[str, Any]]:
    dates = [d for d in store.dates() if d <= today]
    return store.get(dates[0]) if dates else None


def pretrade_check(runtime, query: dict[str, Any]) -> dict[str, Any]:
    from backend.personal.stocks import resolve_stock

    code, name = resolve_stock(runtime, query.get("ts_code") or query.get("name"))
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
    limit = next_day_limit(ma.get("close"), profile)
    entry_price = query.get("entry") or limit
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
        add("pricing", "持仓估值", "warn", f"{', '.join(account['unpriced'])} 没有收盘价，总仓位可能低估")
    if account["unknown_shares"]:
        add("pricing", "持仓估值", "warn",
            f"{', '.join(account['unknown_shares'])} 没填股数，总仓位和可用资金按 0 计，可能低估")

    levels = [c["level"] for c in checks]
    verdict = "不通过" if "fail" in levels else "有警告" if "warn" in levels else "通过"
    return {"ts_code": code, "name": name, "verdict": verdict, "checks": checks,
            "inputs": {"entry_price": entry_price, "stop_price": stop_price,
                       "entry_source": "手动填写" if query.get("entry") else
                       f"下一交易日限价：{status.get('evaluation_day') or ''} 收盘价 + 追高上限 {profile.chase_limit:.0%}",
                       "stop_source": "手动填写" if query.get("stop") else "21日线下方 1%"},
            "market": market, "sizing": sizing, "status": status, "profile_source": profile_source,
            "account": {k: account[k] for k in ("cash", "exposure", "exposure_pct", "capital")},
            "note": "检查只说明是否符合你自己的规则，不构成买卖建议；是否下单由你决定。"}


# -- today's actions ------------------------------------------------------------

_ACTION_ORDER = {"sell": 0, "plan_exit": 1, "near_stop": 2, "buy": 3, "plan": 4, "reminder": 5, "hold": 6, "none": 7}
_NEAR_STOP = 0.03


def _day_text(day: Any) -> str:
    text = str(day or "")
    return f"{text[:4]}-{text[4:6]}-{text[6:8]}" if len(text) == 8 and text.isdigit() else text


_WEEKDAYS = "一二三四五六日"
_REGIME_PLAIN = {"strong": "市场向好", "range": "市场震荡", "weak": "市场偏弱"}


def _px(value: Any) -> str:
    """Price for people: 1,486.85."""
    try:
        return f"{float(value):,.2f}"
    except (TypeError, ValueError):
        return "—"


def watch_word(item: dict[str, Any]) -> tuple[str, str, str]:
    """One status word, its colour level and one plain fact for a stock in 我的股票."""
    entry, ex = item.get("entry"), item.get("exit") or {}
    if item.get("held"):
        verdict = ex.get("verdict") or ""
        pnl = ""
        if ex.get("fresh"):
            pnl = "今天刚买"   # no close since the buy: a return vs. yesterday's close would mislead
        elif ex.get("profit_pct") is not None:
            p = float(ex["profit_pct"])
            pnl = f"{'赚' if p >= 0 else '亏'} {abs(p):.1%}"
        if verdict.startswith("离场"):
            why = "跌破止损线" if "21日线" in verdict else "从最高点回落超过 10%"
            return "该卖了", "danger", " · ".join(x for x in (pnl, why) if x)
        stop, close = ex.get("stop"), ex.get("close")
        if verdict == NO_DATA or not stop or not close:
            return "看不清", "muted", "数据不够，暂时算不出止损线"
        distance = close / stop - 1
        if distance < 0:
            return "留意", "warn", " · ".join(x for x in (pnl, "已在止损线下方") if x)
        if distance <= _NEAR_STOP:
            return "留意", "warn", " · ".join(x for x in (pnl, f"离止损只差 {distance:.1%}") if x)
        return "安全", "muted", " · ".join(x for x in (pnl, f"离止损还有 {distance:.0%}") if x)
    words = {
        BUY: ("可以买", "ok", "买入信号出现了"),
        "等待金叉": ("快了", "warn", "快到买点了"),
        "金叉未放量": ("快了", "warn", "信号出现了，但成交量不够"),
        "周线符合，等待日线信号": ("等待", "muted", "趋势向上，等买点"),
        BELOW_MA: ("等待", "muted", "跌破 21 日线，先不看"),
        WEEKLY_FAIL: ("等待", "muted", "趋势不对，先不看"),
    }
    return words.get(entry, ("看不清", "muted", "数据不够，暂时判断不了"))


def _session_label(day: Optional[str]) -> str:
    """'11-23 周一' for a YYYYMMDD trading day."""
    try:
        parsed = datetime.strptime(str(day), "%Y%m%d")
    except (TypeError, ValueError):
        return "下一交易日"
    return f"{parsed:%m-%d} 周{_WEEKDAYS[parsed.weekday()]}"


def next_session(runtime, day: Optional[str]) -> Optional[str]:
    """First open trading day after ``day`` from the exchange calendar (None if unknown)."""
    if not day:
        return None
    try:
        start = (datetime.strptime(str(day), "%Y%m%d") + pd.Timedelta(days=1)).strftime("%Y%m%d")
        end = (datetime.strptime(str(day), "%Y%m%d") + pd.Timedelta(days=20)).strftime("%Y%m%d")
        cal = runtime.provider.trade_cal(start_date=start, end_date=end, is_open=True)
        days = sorted(str(d) for d in cal["cal_date"])
        return days[0] if days else None
    except Exception:  # noqa: BLE001 - calendar may be unavailable; fall back to wording
        return None


def _session_info() -> dict[str, Any]:
    try:
        from backend.market.session import get_trading_session

        return get_trading_session().describe_cached()
    except Exception:  # noqa: BLE001
        return {}


def today_view(runtime) -> dict[str, Any]:
    """The execution list: what to do in the next session, judged on the last close.

    The user does not watch the market intraday. Everything here is decided on
    the last confirmed daily close and phrased as orders that can be placed in
    the broker app the evening before or before the open (limit buys with a
    give-up price, conditional stop orders). Built only from the user's own
    rules and records; nothing here is a recommendation to trade.
    """
    profile, _ = RiskProfileStore(runtime.root).load()
    account = account_view(runtime)
    watch = watchlist_status(runtime)
    info = _session_info()
    phase = info.get("phase")
    trading_day = str(info.get("current_trading_day") or datetime.now().strftime("%Y%m%d"))
    basis = watch.get("evaluation_day") or account.get("price_basis")
    after_close = phase in ("收盘", "盘后", "休市")
    next_day = next_session(runtime, basis) if after_close else None
    next_label = _session_label(next_day) if next_day else "下一交易日"
    target = next_label if after_close else "今天"
    if after_close:
        notice = (f"{_day_text(basis)} 收盘数据已出：以下是 {next_label} 的执行单，"
                  "可以今晚或开盘前在券商 App 设好条件单。")
    else:
        notice = (f"现在是{phase or '交易日'}：以下依据 {_day_text(basis)} 收盘，按单执行即可，盘中不必盯盘；"
                  "今天收盘后再更新。")
    market = market_state_for(runtime, basis)
    regime = profile.regime_for_state(market.get("state"))
    cap = profile.exposure_caps[regime] if regime else None

    plans = PlanStore(runtime.root)
    dates = plans.dates()
    plan = next((plans.get(d) for d in dates if basis and d <= basis), None)
    held = {p["ts_code"]: p for p in account["positions"]}
    status_by_code = {item["ts_code"]: item for item in watch["items"]}
    planned = {c["ts_code"]: c for c in (plan or {}).get("candidates", [])}
    plan_holdings = {h["ts_code"]: h for h in (plan or {}).get("holdings", [])}
    marks = ExecutionStore(runtime.root).get(basis)
    actions: list[dict[str, Any]] = []

    def label(code, name):
        return f"{name} {code}" if name else code

    def add(kind, level, code, name, title, detail, next_step, check=False, order=None,
            headline="", instruction="", sub=""):
        key = f"{kind}:{code or 'plan'}"
        actions.append({"kind": kind, "level": level, "ts_code": code, "name": name or "", "title": title,
                        "detail": detail, "next_step": next_step, "check": check, "order": order,
                        "key": key, "mark": marks.get(key),
                        "headline": headline or title, "instruction": instruction or next_step, "sub": sub})

    def size(entry, stop):
        if account["unknown_shares"]:
            return {"lots": 0, "reason": "有持仓没填股数，总仓位和可用资金不准，先在观察池补填股数",
                    "plain": "有持仓没填股数，先补上再算能买多少"}
        if not entry or not stop or float(stop) >= float(entry):
            return None
        try:
            return size_position(profile, entry, stop, market_state=market.get("state"),
                                 current_exposure=account["exposure"], available_cash=account["cash"])
        except ValueError:
            return None

    def short(name, code):
        return name or code

    for pos in account["positions"]:
        code, name, ex = pos["ts_code"], pos.get("name", ""), pos.get("exit") or {}
        verdict, stop, close = ex.get("verdict") or "", ex.get("stop"), pos.get("close")
        shares = f" {pos['shares']} 股" if pos.get("shares") else "（全部）"
        all_shares = f"全部 {pos['shares']} 股" if pos.get("shares") else "全部"
        if verdict.startswith("离场"):
            add("sell", "danger", code, name, f"卖出 {label(code, name)}",
                f"{verdict}：{ex.get('rule', '')}；{_day_text(basis)} 收盘 {close}，离场线 {stop}",
                f"{target}开盘后卖出{shares}；成交后在观察池点「已卖出」",
                order={"side": "sell", "when": target, "shares": pos.get("shares")},
                headline=f"卖出 {short(name, code)}", instruction=f"开盘后卖掉{all_shares}",
                sub="跌破了止损线" if "21日线" in verdict else "从最高点回落超过了 10%")
            continue
        if verdict == NO_DATA or not stop:
            add("near_stop", "warn", code, name, f"无法判断离场：{label(code, name)}",
                ex.get("reason") or "缺少复权日线或买入日数据", "检查数据后手动确认离场线",
                headline=short(name, code), instruction="数据不够，暂时算不出止损线，请自己留意")
            continue
        distance = (close / stop - 1) if close and stop else None
        stop_order = f"止损条件单：价格 ≤ {stop} 时卖出{shares}（每天收盘后按新的离场线更新）"
        hp = plan_holdings.get(code)
        if hp and hp["action"] in ("离场", "减仓"):
            add("plan_exit", "warn", code, name, f"计划{hp['action']} {label(code, name)}",
                f"来自 {_day_text(plan['date'])} 的计划{('：' + hp['note']) if hp.get('note') else ''}",
                f"{target}按计划{hp['action']}；成交后在观察池更新持有状态",
                headline=f"按计划{hp['action']} {short(name, code)}", instruction=f"开盘后{hp['action']}",
                sub=hp.get("note") or "")
        elif distance is not None and distance <= _NEAR_STOP:
            add("near_stop", "warn", code, name, f"接近离场线 {label(code, name)}",
                f"{_day_text(basis)} 收盘 {close}，离场线 {stop}，只差 {distance:.1%}", stop_order,
                order={"side": "stop", "price": stop, "shares": pos.get("shares")},
                headline=short(name, code), instruction=f"止损单设在 {_px(stop)}",
                sub=(ex.get("note") or "已在止损线下方，收盘仍在线下就该卖") if distance < 0
                else f"只差 {distance:.1%} 就会卖出")
        else:
            add("hold", "info", code, name, f"持有 {label(code, name)}",
                f"离场线 {stop}" + (f"，距 {_day_text(basis)} 收盘 {distance:.1%}" if distance is not None else ""),
                stop_order, order={"side": "stop", "price": stop, "shares": pos.get("shares")},
                headline=short(name, code), instruction=f"止损单设在 {_px(stop)}",
                sub=(f"离止损还有 {distance:.0%}" if distance is not None else "")
                + ("（刚买入）" if ex.get("fresh") else ""))

    for item in watch["items"]:
        code = item["ts_code"]
        if item.get("entry") != BUY or code in held:
            continue
        cross = (item.get("macd") or {}).get("cross") or {}
        ma = item.get("ma") or {}
        stop, limit = ma.get("suggested_stop"), next_day_limit(ma.get("close"), profile)
        sized = size(limit, stop)
        in_plan = code in planned
        detail = (f"{_day_text(cross.get('date'))} 金叉放量（量比 {cross.get('volume_ratio')}）；"
                  + (f"在 {_day_text(plan['date'])} 的计划内" if in_plan else "不在计划内"))
        nm = short(item.get("name", ""), code)
        if sized and sized["lots"] > 0:
            add("buy", "ok", code, item.get("name", ""), f"买点 {label(code, item.get('name', ''))}", detail,
                f"{target}限价 ≤ {limit} 买入 {sized['shares']} 股（约 {sized['value']:,.0f} 元）；"
                f"开盘高于 {limit} 就放弃；成交后设止损条件单 ≤ {stop}", check=True,
                order={"side": "buy", "limit": limit, "shares": sized["shares"], "stop": stop},
                headline=f"买入 {nm}", instruction=f"挂限价 {_px(limit)} 元，买 {sized['shares']} 股",
                sub=f"开盘高于 {_px(limit)} 就不买 · 买到后止损设 {_px(stop)}" + ("" if in_plan else " · 不在计划内"))
        else:
            add("buy", "warn", code, item.get("name", ""), f"买点 {label(code, item.get('name', ''))}（买不了）",
                detail, f"按你的风控买不了：{(sized or {}).get('reason') or '缺少限价或止损价'}", check=True,
                headline=f"买入 {nm}（暂时买不了）",
                instruction=(sized or {}).get("plain") or "按你的资金和仓位上限，连 1 手（100 股）都买不了")

    for code, cand in planned.items():
        if code in held or (status_by_code.get(code) or {}).get("entry") == BUY:
            continue
        item = status_by_code.get(code) or {}
        ma = item.get("ma") or {}
        entry = cand.get("entry_price") or next_day_limit(ma.get("close"), profile)
        stop = cand.get("stop_price") or ma.get("suggested_stop")
        sized = size(entry, stop)
        shares = cand.get("shares") or (sized or {}).get("shares")
        state = item.get("entry") or "不在观察池，状态未知"
        order = ({"side": "buy", "limit": entry, "shares": shares, "stop": stop}
                 if entry and stop and shares else None)
        nm = short(cand.get("name", "") or item.get("name", ""), code)
        word = watch_word(item)[0] if item else "不在我的股票里"
        add("plan", "info", code, cand.get("name", "") or item.get("name", ""),
            f"计划候选 {label(code, cand.get('name', '') or item.get('name', ''))}",
            f"{cand.get('entry_condition') or '计划买入'}（当前状态：{state}）",
            (f"条件满足时：限价 ≤ {entry} 买入 {shares} 股，止损 {stop}" if order
             else "条件满足时先做交易前检查"), check=True, order=order,
            headline=f"计划买入 {nm}",
            instruction=(f"条件到了再买：限价 {_px(entry)}，{shares} 股，止损 {_px(stop)}" if order
                         else "条件到了再买，先点「能买吗」检查"),
            sub=f"现在：{word}" + (f" · {cand['entry_condition']}" if cand.get("entry_condition") else ""))

    for code in account["unknown_shares"]:
        name = held.get(code, {}).get("name", "")
        add("reminder", "warn", code, name, f"补填股数 {label(code, name)}",
            "持有但没填股数：总仓位、可用资金和新的买入股数都无法准确计算",
            "在观察池点「标记持有」重新保存，填上股数",
            headline=f"补填股数：{short(name, code)}", instruction="在「我的股票」里点它，补上股数")
    if after_close and basis and basis not in dates:
        add("reminder", "warn", "", "", "保存明日计划", "今天收盘后还没有保存计划",
            "在「计划」里写下明日候选（最多 3 只）和每只持仓的动作",
            headline="写明日计划", instruction="花 2 分钟写下明天想买什么、持有的怎么处理")
    elif not dates:
        add("reminder", "info", "", "", "还没有任何计划", "没有计划时，任何买入都属于计划外交易",
            "收盘后在「计划」里保存第一份计划",
            headline="写第一份计划", instruction="收盘后在「计划」里写下明天想买什么")

    actions.sort(key=lambda a: _ACTION_ORDER.get(a["kind"], 9))
    if not any(a["kind"] != "hold" for a in actions):
        add("none", "info", "", "", "没有需要处理的事项", "持仓没有触发离场，观察池没有出现买点。",
            "继续按计划等待；持仓的止损条件单保持有效",
            headline="不用操作", instruction="持有的股票止损单保持不变")

    todo = [a for a in actions if a["kind"] not in ("hold", "none")]
    holds = [a for a in actions if a["kind"] == "hold"]
    room = round(profile.capital * cap - account["exposure"], 2) if cap is not None else None
    if cap is not None:
        market_line = f"{_REGIME_PLAIN.get(regime, '市场')}，最多用 {profile.capital * cap:,.0f} 元（{cap:.0%}）"
        if account["exposure"] > 0:
            market_line += f"，还能用 {max(room, 0):,.0f} 元"
    else:
        market_line = "市场状态还不知道，买之前先点「能买吗」检查"
    hero = {
        "day_line": (f"{next_label} · 开盘前设好就行，不用盯盘" if after_close
                     else f"今天 {_session_label(trading_day)} · 按单执行，不用盯盘"),
        "headline": f"要做 {len(todo)} 件事" if todo else "不用操作",
        "market_line": market_line,
        "holds_line": f"其余 {len(holds)} 只持有中，止损单都已设好" if holds else "",
        "basis_line": f"依据 {_day_text(basis)} 收盘" if basis else "",
    }

    return {
        "trading_day": trading_day, "phase": phase, "evaluation_day": basis, "basis_date": basis,
        "mode": "next" if after_close else "today", "next_session": next_day,
        "title": (f"执行单 · {next_label}" if after_close else "今日执行单"),
        "notice": notice, "plan_date": plan["date"] if plan else None,
        "plan_saved_today": bool(basis and basis in dates),
        "market": {**market, "regime": regime, "regime_label": REGIME_LABELS.get(regime) if regime else None,
                   "exposure_cap": cap},
        "account": {"capital": account["capital"], "cash": account["cash"], "cash_estimated": True,
                    "exposure": account["exposure"], "exposure_pct": account["exposure_pct"],
                    "positions": len(account["positions"]), "unknown_shares": account["unknown_shares"],
                    "room": room},
        "counts": {"sell": sum(a["kind"] == "sell" for a in actions),
                   "buy": sum(a["kind"] == "buy" and a["level"] == "ok" for a in actions),
                   "watch": len(watch["items"])},
        "chase_limit": profile.chase_limit,
        "hero": hero, "setup": not watch["items"] and not account["positions"],
        "actions": actions,
        "note": "只依据你自己的规则和记录列出待办，不构成买卖建议；条件单由你在券商 App 设置。",
    }
