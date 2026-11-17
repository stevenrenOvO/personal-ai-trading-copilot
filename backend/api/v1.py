"""V1.0 REST-ish endpoints on top of the standard-library WSGI server."""

from __future__ import annotations

import json
import math
import re
import time
import uuid
import hashlib
import logging
from datetime import datetime, timezone
from urllib.parse import parse_qs
from typing import Any, Optional

from backend.ai.agent import AICopilot
from backend.api.runtime import Runtime
from backend.api.data_status import date_status
from backend.backtest.ashare_engine import AShareBacktestEngine
from backend.backtest.metrics import calculate_metrics
from backend.board.engine import BoardEngine
from backend.board.ladder import LimitLadderEngine
from backend.emotion.engine import EmotionEngine
from backend.market.engine import MarketEngine, MarketState
from backend.market.quote_service import get_quote_service
from backend.market.session import get_trading_session
from backend.market.snapshot import (
    compute_full_market_board_environment,
    compute_full_market_sectors,
    compute_full_market_state,
    rank_snapshot,
)
from backend.opportunity.engine import OpportunityEngine
from backend.opportunity.candidates import CandidatePoolEngine
from backend.opportunity.setups import OpportunitySetupEngine, SetupResult
from backend.opportunity.ranking import OpportunityRankingEngine, RankedPlan
from backend.risk.rules import DecisionRiskEngine, RiskContext
from backend.sector.engine import SectorEngine
from backend.journal.audit import JournalAuditEngine
from backend.journal.trades import TradeRecord
from backend.strategies.base import TradeSignal


def _json(status: str, data: dict) -> tuple[str, list, bytes]:
    def clean(value):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]
        return value
    return status, [("Content-Type", "application/json; charset=utf-8")], json.dumps(
        clean(data), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _query(environ: dict) -> dict[str, str]:
    raw = environ.get("QUERY_STRING", "")
    parsed = parse_qs(raw)
    return {k: v[0] for k, v in parsed.items() if v}


def _body(environ: dict) -> dict:
    length = int(environ.get("CONTENT_LENGTH") or 0)
    if length < 0 or length > 1_048_576:
        raise ValueError("请求体大小无效（上限 1 MB）")
    raw = environ["wsgi.input"].read(length).decode("utf-8")
    def invalid_constant(value):
        raise ValueError(f"非法数值: {value}")
    data = json.loads(raw, parse_constant=invalid_constant) if raw else {}
    if not isinstance(data, dict):
        raise ValueError("请求体必须为 JSON 对象")
    return data


def _date(environ: dict, runtime: Runtime | None = None, default: str = "20240103") -> str:
    q = _query(environ)
    if "date" in q:
        _validate_date(q["date"])
        return q["date"]
    if runtime is not None:
        from backend.api.data_status import display_dates
        latest = display_dates(runtime).get("date")
        if latest:
            return latest
    return default


def _validate_date(value):
    if not isinstance(value, str) or not re.fullmatch(r"\d{8}", value):
        raise ValueError("日期格式须为 YYYYMMDD")
    datetime.strptime(value, "%Y%m%d")
    return value


def _validate_trade(data, *, paper=False):
    if not isinstance(data.get("ts_code"), str) or not re.fullmatch(r"\d{6}\.(SH|SZ|BJ)", data["ts_code"]):
        raise ValueError("请填写有效股票代码，如 000001.SZ")
    actions = ("BUY", "SELL") if paper else ("BUY", "SELL", "HOLD", "WAIT", "MISSED_SIGNAL")
    if data.get("action", "BUY") not in actions:
        raise ValueError("交易动作无效")
    for name in ("price", "quantity", "position_before", "position_after", "strength"):
        value = data.get(name)
        if name == "price" and value == "":
            data[name] = None
            continue
        if value is not None and value != "":
            if isinstance(value, bool):
                raise ValueError(f"{name} 不能为布尔值")
            try:
                number = float(value)
            except OverflowError as exc:
                raise ValueError(f"{name} 数值超出可用范围") from exc
            if not math.isfinite(number) or number < 0 or (name == "price" and number <= 0):
                raise ValueError(f"{name} 必须为有效正值")
            data[name] = number
    if "trade_date" in data and data["trade_date"] not in (None, ""):
        _validate_date(data["trade_date"])
    elif "trade_date" in data:
        data["trade_date"] = ""
    for name in ("symbol", "strategy_id", "strategy_version", "signal_id", "market_state",
                 "emotion_state", "sector_state", "stock_state", "system_recommendation",
                 "user_reason", "execution_reason", "reason"):
        if name in data and (not isinstance(data[name], str) or len(data[name]) > 4000):
            raise ValueError(f"{name} 必须为不超过 4000 字的文本")
    result = data.get("result")
    if result is not None:
        if not isinstance(result, dict):
            raise ValueError("result 必须为对象")
        for name in ("pnl", "return_pct", "holding_days"):
            if result.get(name) is not None:
                if isinstance(result[name], bool):
                    raise ValueError(f"result.{name} 必须为有限数值")
                try:
                    result[name] = float(result[name])
                except OverflowError as exc:
                    raise ValueError(f"result.{name} 数值超出可用范围") from exc
                if not math.isfinite(result[name]):
                    raise ValueError(f"result.{name} 必须为有限数值")
                if name == "holding_days" and result[name] < 0:
                    raise ValueError("持有天数不可为负数")
    if paper and (data.get("price") in (None, "") or data.get("quantity") in (None, "")
                  or float(data["quantity"]) <= 0 or not float(data["quantity"]).is_integer()):
        raise ValueError("模拟交易必须填写正价格与正整数数量")


def dispatch(environ: dict, start_response, runtime: Runtime) -> bool:
    path = environ.get("PATH_INFO", "")
    method = environ.get("REQUEST_METHOD", "GET")
    q = _query(environ)

    handlers: dict[str, Any] = {
        "GET /market/state": lambda: _market(environ, runtime),
        "GET /market/snapshot": lambda: _market_snapshot(environ),
        "GET /emotion/state": lambda: _emotion(environ, runtime),
        "GET /sectors": lambda: _sectors(environ, runtime),
        "GET /opportunities": lambda: _opportunities(environ, runtime),
        "GET /opportunities/candidates": lambda: _opportunity_candidates(environ, runtime),
        "GET /opportunities/setups": lambda: _opportunity_setups(environ, runtime),
        "GET /opportunities/plan": lambda: _opportunity_plan(environ, runtime),
        "GET /board/environment": lambda: _board(environ, runtime),
        "GET /board/ladder": lambda: _board_ladder(environ, runtime),
        "GET /risk": lambda: _risk(environ, runtime),
        "GET /positions": lambda: _positions(environ, runtime),
        "GET /strategies": lambda: _strategies(runtime),
        "GET /strategies/health": lambda: _strategy_health(runtime),
        "GET /trades": lambda: _trades(runtime),
        "POST /trades": lambda: _post_trade(environ, runtime),
        "GET /profile": lambda: _profile(runtime),
        "GET /backtest/runs": lambda: _backtest_runs(runtime),
        "POST /backtest/runs": lambda: _post_backtest_run(environ, runtime),
        "POST /paper/trade": lambda: _post_paper_trade(environ, runtime),
        "GET /data/latest": lambda: _data_latest(runtime),
        "GET /data/history/dates": lambda: _data_history_dates(runtime),
        "GET /data/ingest/status": lambda: _ingest_status(runtime),
        "GET /ai/copilot": lambda: _ai_copilot(environ, runtime),
        "GET /personal/risk-profile": lambda: _personal_risk_profile(runtime),
        "POST /personal/risk-profile": lambda: _post_personal_risk_profile(environ, runtime),
        "GET /personal/sizing": lambda: _personal_sizing(environ, runtime),
        "GET /personal/watchlist": lambda: _personal_watchlist(runtime),
        "POST /personal/watchlist": lambda: _post_personal_watchlist(environ, runtime),
        "POST /personal/watchlist/remove": lambda: _post_personal_watchlist_remove(environ, runtime),
        "GET /personal/watchlist/status": lambda: _personal_watchlist_status(runtime),
        "GET /personal/account": lambda: _personal_account(runtime),
        "GET /personal/trades": lambda: _personal_trades(environ, runtime),
        "POST /personal/trades": lambda: _post_personal_trade(environ, runtime),
        "POST /personal/trades/void": lambda: _post_personal_trade_void(environ, runtime),
        "GET /personal/plan": lambda: _personal_plan(environ, runtime),
        "POST /personal/plan": lambda: _post_personal_plan(environ, runtime),
        "GET /personal/pretrade": lambda: _personal_pretrade(environ, runtime),
    }

    key = f"{method} {path}"
    if key in handlers:
        try:
            from backend.data.auto_ingest import ensure_today_bars
            ensure_today_bars(runtime)
        except Exception:
            pass
        try:
            status, headers, body = handlers[key]()
        except (ValueError, TypeError, KeyError, UnicodeError) as exc:
            status, headers, body = _json("400 Bad Request", {"status": "error", "detail": str(exc)})
        start_response(status, headers)
        return body, True

    match = re.match(r"^/trades/([^/]+)/audit$", path)
    if method in ("GET", "POST") and match:
        status, headers, body = _audit_trade(environ, runtime, match.group(1))
        start_response(status, headers)
        return body, True

    return b"", False


def _market(environ, runtime: Runtime):
    date = _date(environ, runtime)
    latest = _latest_date(runtime)
    df, meta = get_quote_service().get()
    if _snapshot_matches(df, meta, date, runtime):
        state = compute_full_market_state(df, date)
        payload = state.to_dict()
        metadata = _snapshot_meta(meta)
        if not _local_history_gap(runtime, date):
            structure = _market_state(runtime, date)
            for field in ("max_consecutive_up", "yesterday_limit_up_performance", "index_pct"):
                payload[field] = getattr(structure, field)
            metadata.update(structural_date=date, structural_source="local_history")
        return _json("200 OK", {"status": "ok", "data": payload, "meta": metadata})
    if latest and date == latest and not date_status(runtime, date)["complete"]:
        # Snapshot not ready yet: say so explicitly instead of silently using
        # a different (CSI300) data source.
        state = MarketState(date=date, state="数据准备中", confidence=0.0, risk_level="high")
        payload = state.to_dict()
        payload.update(available=False, score=None, recommended_exposure=None)
        return _json("200 OK", {"status": "ok", "data": payload, "meta": _snapshot_meta(meta, requested_date=date, unavailable=True)})
    state = _market_state(runtime, date)
    payload = state.to_dict()
    if state.state == "数据不足":
        payload.update(available=False, score=None, recommended_exposure=None)
    return _json("200 OK", {"status": "ok", "data": payload, "meta": _historical_meta(date, runtime=runtime)})


def _market_state(runtime: Runtime, date: str):
    """Full-market state for one day, memoised on the runtime."""
    if _local_history_gap(runtime, date) or (runtime.provider.name == "storage" and len(runtime.provider.stock_basic()) < 1000):
        return MarketState(date=date, state="数据不足", confidence=0.0, risk_level="high")
    return runtime.analysis_cache.get_or_compute(
        ("market", date),
        lambda: MarketEngine(runtime.provider).calculate(date),
    )


def _emotion_state(runtime: Runtime, date: str, market=None, ladder=None):
    """Rule-based emotion stage for a confirmed trading day (Step 3)."""

    def build():
        base = market if market is not None else _market_state(runtime, date)
        structure = ladder if ladder is not None else _ladder_snapshot(runtime, date)
        return EmotionEngine(runtime.provider).calculate(
            date, market=base, ladder=structure
        )

    return runtime.analysis_cache.get_or_compute(("emotion", date), build)


def _sector_scores(runtime: Runtime, date: str, top: int | None = 5):
    """Full sector ranking, memoised; ``top`` only slices the cached result.

    The opportunity engine needs the *whole* ranking to know a stock's sector
    rank, while the dashboard shows the top few -- one computation serves both.
    """
    if _local_history_gap(runtime, date):
        return []
    ranked = runtime.analysis_cache.get_or_compute(
        ("sectors", date),
        lambda: SectorEngine(runtime.provider).ranked(date),
    )
    return ranked if top is None else ranked[:top]


def _board_state(runtime: Runtime, date: str, market=None, emotion=None):
    def build():
        base_market = market if market is not None else _market_state(runtime, date)
        base_emotion = (
            emotion
            if emotion is not None
            else _emotion_state(runtime, date, base_market)
        )
        return BoardEngine(runtime.provider).calculate(
            date, market=base_market, emotion=base_emotion
        )

    return runtime.analysis_cache.get_or_compute(("board", date), build)


def _ladder_snapshot(runtime: Runtime, date: str):
    """Limit-up ladder / leader structure / yesterday premium for one day."""
    if _local_history_gap(runtime, date):
        from backend.board.ladder import LadderSnapshot
        return LadderSnapshot(date=date, available=False, notes=("当日行情覆盖不足",))
    return runtime.analysis_cache.get_or_compute(
        ("ladder", date),
        lambda: LimitLadderEngine(runtime.provider).snapshot(date),
    )


def _board_ladder(environ, runtime: Runtime):
    date = _date(environ, runtime)
    snapshot = _ladder_snapshot(runtime, date)
    meta = _historical_meta(date, kind="historical_scan", runtime=runtime)
    if not snapshot.available:
        meta["status"] = "no_data"
        meta["error"] = "；".join(snapshot.notes) or "该交易日无本地数据"
    return _json(
        "200 OK", {"status": "ok", "data": snapshot.to_dict(), "meta": meta}
    )


def _market_snapshot(environ):
    q = _query(environ)
    df, meta = get_quote_service().get()

    top = int(q.get("top", 100))
    sort_by = q.get("sort", "pct_chg")
    ascending = q.get("ascending", "0") == "1"
    min_amount = float(q["min_amount"]) if "min_amount" in q else None
    min_price = float(q["min_price"]) if "min_price" in q else None
    ranked = rank_snapshot(
        df,
        sort_by=sort_by,
        ascending=ascending,
        top_n=top,
        min_amount=min_amount,
        min_price=min_price,
    )
    return _json(
        "200 OK",
        {
            "status": "ok",
            "meta": meta.to_dict(),
            "data": ranked.to_dict(orient="records"),
        },
    )


def _emotion(environ, runtime: Runtime):
    date = _date(environ, runtime)
    # The emotion stage is a *structural* judgement: it needs confirmed daily
    # bars (who closed at the limit, who broke the board). While today's bars
    # do not exist yet, the honest answer is the most recent confirmed day plus
    # an explicit basis date -- never an intraday guess.
    basis = date
    basis_note = None
    if _local_history_gap(runtime, date):
        fallback = _history_latest_date(runtime, before=date)
        if fallback:
            basis = fallback
            basis_note = (
                f"{date} 的行情尚未落地到本地历史（本地覆盖 {_history_coverage(runtime) or '—'}），"
                f"情绪阶段基于最近确认交易日 {fallback} 的真实结构"
            )
    ladder = _ladder_snapshot(runtime, basis)
    market = _market_state(runtime, basis)
    state = _emotion_state(runtime, basis, market, ladder)
    payload = state.to_dict()
    meta = _historical_meta(basis, runtime=runtime)
    if basis_note:
        meta["status"] = "no_data"
        meta["error"] = basis_note
        payload["basis_note"] = basis_note
    return _json("200 OK", {"status": "ok", "data": payload, "meta": meta})


def _sectors(environ, runtime: Runtime):
    date = _date(environ, runtime)
    top = int(_query(environ).get("top", 5))
    latest = _latest_date(runtime)
    df, meta = get_quote_service().get()
    if _snapshot_matches(df, meta, date, runtime):
        industry = _industry_map(runtime)
        sectors = compute_full_market_sectors(df, industry)[:top]
        return _json("200 OK", {"status": "ok", "data": sectors, "meta": _snapshot_meta(meta)})
    if latest and date == latest and not date_status(runtime, date)["complete"]:
        return _json("200 OK", {"status": "ok", "data": [], "meta": _snapshot_meta(meta, requested_date=date, unavailable=True)})
    sectors = _sector_scores(runtime, date, top)
    return _json(
        "200 OK",
        {"status": "ok", "data": [s.to_dict() for s in sectors], "meta": _historical_meta(date, runtime=runtime)},
    )


def _opportunities(environ, runtime: Runtime):
    date = _date(environ, runtime)
    if _local_history_gap(runtime, date):
        # No bars for this day yet (typical before/while the market is open):
        # the daily-bar strategy cannot be evaluated at all. Rather than an
        # empty list that reads like a market verdict, show the most recent
        # completed day and label exactly which day that is.
        fallback = _history_latest_date(runtime, before=date)
        if fallback and fallback != date:
            opps = _scan_opportunities(runtime, fallback)
            meta = _historical_meta(fallback, kind="historical_scan", runtime=runtime)
            meta["status"] = "no_data"
            meta["error"] = (
                f"{date} 的行情尚未落地到本地历史（本地覆盖 {_history_coverage(runtime) or '—'}），"
                f"以下为最近交易日 {fallback} 的机会"
            )
            return _json(
                "200 OK",
                {"status": "ok", "data": [o.to_dict() for o in opps], "meta": meta},
            )
        return _json(
            "200 OK",
            {
                "status": "ok",
                "data": [],
                "meta": {
                    **_historical_meta(date, kind="historical_scan", runtime=runtime),
                    "status": "no_data",
                    "error": (
                        f"该交易日尚未落地到本地历史（本地覆盖 {_history_coverage(runtime) or '—'}），"
                        "无法计算当日机会；请选择已有数据的交易日"
                    ),
                },
            },
        )
    opps = _scan_opportunities(runtime, date)
    return _json(
        "200 OK",
        {
            "status": "ok",
            "data": [o.to_dict() for o in opps],
            "meta": _historical_meta(date, kind="historical_scan", runtime=runtime),
        },
    )


def _candidate_pool(runtime: Runtime, date: str):
    """Candidate pool + four-layer hard elimination (Step 4 stage 1)."""

    def build():
        ladder = _ladder_snapshot(runtime, date)
        market = _market_state(runtime, date)
        emotion = _emotion_state(runtime, date, market, ladder)
        sectors = _sector_scores(runtime, date, None)
        return CandidatePoolEngine(runtime.provider).build(
            date,
            stage=emotion.emotion_cycle,
            stage_rule=emotion.matched_rule,
            ladder=ladder,
            sectors=sectors,
            market=market,
        )

    return runtime.analysis_cache.get_or_compute(("candidates", date), build)


def _typed_opportunities(runtime: Runtime, date: str):
    """Seven setup types + entry state machine (Step 4 stage 2)."""

    def build():
        pool = _candidate_pool(runtime, date)
        market = _market_state(runtime, date)
        ladder = _ladder_snapshot(runtime, date)
        emotion = _emotion_state(runtime, date, market, ladder)
        sectors = {
            s.name: s for s in _sector_scores(runtime, date, None)
        }
        return OpportunitySetupEngine(runtime.provider).build(
            date,
            pool=pool,
            ladder=ladder,
            stage=emotion.emotion_cycle,
            stage_rule=emotion.matched_rule,
            sectors=sectors,
        )

    return runtime.analysis_cache.get_or_compute(("setups", date), build)


def _trade_plan(runtime: Runtime, date: str):
    """Ranked, risk-capped trading plan (Step 4 stage 3)."""

    def build():
        setups = _typed_opportunities(runtime, date)
        snapshot = runtime.account_snapshot(date)
        profile = runtime.profile.load_or_create()
        config = getattr(profile, "config", None)
        max_weight = float(getattr(config, "max_position_weight", 0.20) or 0.20)
        return OpportunityRankingEngine(
            runtime.provider, max_position_weight=max_weight
        ).build(
            date,
            setups=setups,
            state={
                "equity": snapshot.equity,
                "cash": snapshot.cash,
                "positions": dict(snapshot.positions),
            },
        )

    return runtime.analysis_cache.get_or_compute(("plan", date, runtime.paper.snapshot().version), build)


def _opportunity_plan(environ, runtime: Runtime):
    date = _date(environ, runtime)
    if _local_history_gap(runtime, date):
        fallback = _history_latest_date(runtime, before=date)
        if fallback and fallback != date:
            plan = _trade_plan(runtime, fallback)
            meta = _historical_meta(fallback, kind="historical_scan", runtime=runtime)
            meta["status"] = "no_data"
            meta["error"] = (
                f"{date} 的行情尚未落地到本地历史（本地覆盖 {_history_coverage(runtime) or '—'}），"
                f"以下为最近交易日 {fallback} 的交易计划"
            )
            payload = plan.to_dict()
            payload.update(requested_date=date, basis_date=fallback, historical_only=True,
                can_attack=False, max_new_positions=0, available_exposure=0,
                attack_note=f"仅供 {fallback} 历史复盘，不能作为 {date} 的执行计划")
            for item in payload.get("plans", []):
                item["historical_only"] = True
                if isinstance(item.get("position"), dict):
                    item["position"].update(shares=0, lots=0, amount=0, suggested_weight=0,
                        note="当日结构未确认，历史计划不可执行")
            return _json(
                "200 OK", {"status": "ok", "data": payload, "meta": meta}
            )
        return _json(
            "200 OK",
            {
                "status": "ok",
                "data": RankedPlan(date=date, available=False).to_dict(),
                "meta": {
                    **_historical_meta(date, kind="historical_scan", runtime=runtime),
                    "status": "no_data",
                    "error": f"本地历史不含 {date}，无法生成交易计划",
                },
            },
        )
    plan = _trade_plan(runtime, date)
    return _json(
        "200 OK",
        {
            "status": "ok",
            "data": plan.to_dict(),
            "meta": _historical_meta(date, kind="historical_scan", runtime=runtime),
        },
    )


def _opportunity_setups(environ, runtime: Runtime):
    date = _date(environ, runtime)
    if _local_history_gap(runtime, date):
        fallback = _history_latest_date(runtime, before=date)
        if fallback and fallback != date:
            result = _typed_opportunities(runtime, fallback)
            meta = _historical_meta(fallback, kind="historical_scan", runtime=runtime)
            meta["status"] = "no_data"
            meta["error"] = (
                f"{date} 的行情尚未落地到本地历史（本地覆盖 {_history_coverage(runtime) or '—'}），"
                f"以下为最近交易日 {fallback} 的机会识别结果"
            )
            return _json(
                "200 OK", {"status": "ok", "data": result.to_dict(), "meta": meta}
            )
        return _json(
            "200 OK",
            {
                "status": "ok",
                "data": SetupResult(date=date, available=False).to_dict(),
                "meta": {
                    **_historical_meta(date, kind="historical_scan", runtime=runtime),
                    "status": "no_data",
                    "error": f"本地历史不含 {date}，无法识别机会类型",
                },
            },
        )
    result = _typed_opportunities(runtime, date)
    return _json(
        "200 OK",
        {
            "status": "ok",
            "data": result.to_dict(),
            "meta": _historical_meta(date, kind="historical_scan", runtime=runtime),
        },
    )


def _opportunity_candidates(environ, runtime: Runtime):
    date = _date(environ, runtime)
    if _local_history_gap(runtime, date):
        fallback = _history_latest_date(runtime, before=date)
        if fallback and fallback != date:
            pool = _candidate_pool(runtime, fallback)
            meta = _historical_meta(fallback, kind="historical_scan", runtime=runtime)
            meta["status"] = "no_data"
            meta["error"] = (
                f"{date} 的行情尚未落地到本地历史（本地覆盖 {_history_coverage(runtime) or '—'}），"
                f"以下为最近交易日 {fallback} 的候选池"
            )
            return _json(
                "200 OK", {"status": "ok", "data": pool.to_dict(), "meta": meta}
            )
        return _json(
            "200 OK",
            {
                "status": "ok",
                "data": CandidatePool(date=date, available=False).to_dict(),
                "meta": {
                    **_historical_meta(date, kind="historical_scan", runtime=runtime),
                    "status": "no_data",
                    "error": f"本地历史不含 {date}，无法构建候选池",
                },
            },
        )
    pool = _candidate_pool(runtime, date)
    return _json(
        "200 OK",
        {
            "status": "ok",
            "data": pool.to_dict(),
            "meta": _historical_meta(date, kind="historical_scan", runtime=runtime),
        },
    )


def _scan_opportunities(runtime: Runtime, date: str):
    strategy = runtime.strategies.get("strong_sector_breakout") or runtime.strategies.get(
        "ma_cross"
    )
    engine = OpportunityEngine(runtime.provider, strategy, SectorEngine(runtime.provider))

    def build():
        market = _market_state(runtime, date)
        emotion = _emotion_state(runtime, date, market)
        board = _board_state(runtime, date, market, emotion)
        sectors = _sector_scores(runtime, date, None)
        return engine.scan(
            date, market=market, emotion=emotion, board=board, sectors=sectors
        )

    return runtime.analysis_cache.get_or_compute(("opportunities", date), build)


def _history_latest_date(runtime: Runtime, before: str | None = None) -> Optional[str]:
    """Newest trading day the local store can answer for."""
    history = getattr(runtime.provider, "history", None)
    if history is None:
        return None
    from backend.api.data_status import latest_complete
    return latest_complete(runtime, before)


def _local_history_gap(runtime: Runtime, date: str) -> bool:
    """True when the local store is the source but has no bars for ``date``."""
    from backend.data.providers.history import HistoryProvider

    if not isinstance(runtime.provider, HistoryProvider):
        return False
    from backend.api.data_status import date_status
    return not date_status(runtime, date)["complete"]


def _board(environ, runtime: Runtime):
    date = _date(environ, runtime)
    latest = _latest_date(runtime)
    df, meta = get_quote_service().get()
    if _snapshot_matches(df, meta, date, runtime):
        data = _full_market_board(date, df, runtime)
        return _json("200 OK", {"status": "ok", "data": data, "meta": _snapshot_meta(meta)})
    if latest and date == latest and not date_status(runtime, date)["complete"]:
        return _json(
            "200 OK",
            {
                "status": "ok",
                "data": {"date": date, "grade": "—", "limit_up_count": None, "ladder_available": False,
                         "ladder_note": "当日数据准备中"},
                "meta": _snapshot_meta(meta, requested_date=date, unavailable=True),
            },
        )
    if _local_history_gap(runtime, date):
        return _json("200 OK", {"status": "ok", "data": {"date": date, "grade": "—", "available": False,
            "score": None, "recommended_exposure": None, "limit_up_count": None,
            "ladder_available": False, "ladder_note": "当日行情覆盖不足，不能生成涨停环境结论"},
            "meta": _historical_meta(date, runtime=runtime)})
    env = _board_state(runtime, date)
    return _json("200 OK", {"status": "ok", "data": env.to_dict(), "meta": _historical_meta(date, runtime=runtime)})


def _latest_date(runtime: Runtime) -> Optional[str]:
    """The current trading day: exchange calendar first, history as fallback."""
    try:
        day = get_trading_session().current_trading_day()
        if day:
            return day
    except Exception:
        pass
    try:
        return runtime.provider.latest_trade_date()
    except Exception:
        return None


def _snapshot_meta(meta, requested_date=None, unavailable=False) -> dict:
    payload = meta.to_dict()
    if payload.get("error"):
        payload["error"] = "行情数据源暂时不可用，请检查网络或稍后刷新；缓存行情不代表当前交易日"
    if requested_date and payload.get("quote_date") != requested_date:
        payload.update(status="no_data", is_stale=True, requested_date=requested_date,
            error=f"{requested_date} 的行情尚未确认；缓存报价日期为 {payload.get('quote_date') or '未知'}")
    elif unavailable:
        payload.update(status="no_data", is_stale=True, requested_date=requested_date,
            error=f"{requested_date} 行情快照覆盖不足，等待数据源更新")
    try:
        payload["session_phase"] = get_trading_session().phase()
    except Exception:
        payload["session_phase"] = "未知"
    return payload


def _snapshot_matches(frame, meta, date, runtime=None):
    from backend.api.data_status import snapshot_available
    return snapshot_available(runtime, frame, meta, date)


def _historical_meta(date: str, kind: str = "historical", runtime: Runtime | None = None) -> dict:
    coverage = _history_coverage(runtime)
    payload = {
        "source": ("local_history" if runtime and runtime.provider.name == "history" else runtime.provider.name if runtime else "unknown"),
        "is_fixture": bool(runtime and runtime.provider.name == "fixture"),
        "kind": kind,
        "as_of": date,
        "quote_time": None,
        "age_seconds": None,
        "is_stale": False,
        "status": "ok",
        "error": None,
        "rows": None,
        "coverage": coverage,
    }
    if runtime is not None and hasattr(runtime.provider, "history"):
        from backend.api.data_status import date_status
        status = date_status(runtime, date)
        payload.update(requested_date=date, basis_date=date, rows=status["valid_rows"],
                       coverage_ratio=status["coverage_ratio"])
        if not status["complete"]:
            payload.update(status="no_data", error=f"{date} 日线覆盖 {status['coverage_ratio']:.1%}，正在补齐，尚不能生成全市场结构结论")
    return payload


_HISTORY_INFO: Optional[tuple[float, Optional[dict]]] = None
_COVERAGE_TTL_SECONDS = 300.0


def _history_info(runtime: Runtime | None = None) -> Optional[dict]:
    """Cached coverage of the full-market history store.

    Cached for a few minutes only: backfilling more days while the server runs
    must not leave the banner lying about how far the local data reaches.
    """
    if runtime is None or not hasattr(runtime.provider, "history"):
        return None
    def compute():
        cov = runtime.provider.history.coverage()
        return cov if cov.get("start") and cov.get("end") else None
    return runtime.analysis_cache.get_or_compute(("coverage",), compute)


def _history_coverage(runtime: Runtime | None = None) -> Optional[str]:
    """Human-readable coverage, for honest empty-state text."""
    info = _history_info(runtime)
    if not info:
        return None
    return f"{info['start']}~{info['end']}，{info.get('codes', 0)} 个代码"


_HISTORY_DAY_CACHE: dict[str, tuple[float, int]] = {}


def _history_rows_on(date: str, runtime: Runtime) -> int:
    if not hasattr(runtime.provider, "history"):
        return 0
    return runtime.analysis_cache.get_or_compute(("rows_on", date),
        lambda: runtime.provider.history.rows_on(date))


def _history_board_extras(runtime: Runtime, date: str) -> Optional[dict]:
    """Ladder / leader / yesterday-premium for ``date`` from the local store.

    The live snapshot can answer "涨停多少家" but not "几连板、谁是龙头" -- those
    need several days of bars. When the local store already holds this day, the
    ladder is computed from it and labelled as a local-history figure so it is
    never confused with the live snapshot.
    """
    if _local_history_gap(runtime, date) or _history_rows_on(date, runtime) < 100:
        return None
    try:
        env = _board_state(runtime, date)
    except Exception:
        return None
    if env.grade == "D" and env.factors == ("数据不足",):
        return None
    return {
        "ladder": env.ladder.to_dict(),
        "emotion_cycle": env.emotion_cycle,
        "leader": env.leader,
        "leader_name": env.leader_name,
        "leader_height": env.leader_height,
        "yesterday_limit_up_premium": env.yesterday_limit_up_premium,
        "source": "local_history",
        "note": (
            "连板梯队/龙头由本地历史日线计算（数据来源 local_history），"
            "与上方实时快照的涨停家数口径可能略有差异"
        ),
    }


def _industry_map(runtime: Runtime) -> dict[str, str]:
    mapping: dict[str, str] = {}
    try:
        basic = runtime.provider.stock_basic(list_status="L")
    except Exception:
        basic = runtime.provider.stock_basic()
    if not basic.empty and "ts_code" in basic.columns and "industry" in basic.columns:
        for _, row in basic.iterrows():
            industry = str(row.get("industry", "") or "")
            if industry:
                mapping[str(row["ts_code"])] = industry
    return mapping


def _full_market_board(date: str, df, runtime: Runtime | None = None) -> dict:
    extras = _history_board_extras(runtime, date) if runtime is not None else None
    payload = compute_full_market_board_environment(df, date, history_extras=extras)
    # Snapshot strength is not the confirmed structural emotion cycle.
    payload["emotion_cycle"] = extras.get("emotion_cycle") if extras else None
    payload["emotion_basis_date"] = date if extras else None
    payload["emotion_source"] = "local_history" if extras else "unavailable"
    return payload


def _risk(environ, runtime: Runtime):
    date = _date(environ, runtime)
    market = _market_state(runtime, date)
    emotion = _emotion_state(runtime, date, market)
    snapshot = runtime.account_snapshot(date)
    risk = DecisionRiskEngine().decide(
        RiskContext(
            equity=snapshot.equity,
            cash=snapshot.cash,
            peak_equity=snapshot.peak_equity,
            positions=snapshot.positions,
            prices=snapshot.valuation.get("prices", {}),
        ),
        market=market,
        emotion=emotion,
        has_positions=any(p.quantity > 0 for p in snapshot.positions.values()),
    )
    return _json("200 OK", {"status": "ok", "data": risk.to_dict()})


def _positions(environ, runtime: Runtime):
    snapshot = runtime.account_snapshot(_latest_date(runtime))
    return _json("200 OK", {"status": "ok", "data": snapshot.to_dict()})


def _strategies(runtime: Runtime):
    return _json(
        "200 OK",
        {
            "status": "ok",
            "data": [
                {
                    "strategy_id": s.strategy_id,
                    "version": s.version,
                    "status": s.status,
                    "enabled": s.enabled,
                }
                for s in runtime.strategies.values()
            ],
        },
    )


def _strategy_health(runtime: Runtime):
    return _json(
        "200 OK",
        {"status": "ok", "data": [h.to_dict() for h in runtime.registry.store.list()]},
    )


def _trades(runtime: Runtime):
    trades = runtime.journal.list()
    return _json("200 OK", {"status": "ok", "data": [t.to_dict() for t in trades]})


def _post_trade(environ, runtime: Runtime):
    data = _body(environ)
    _validate_trade(data)
    request_id = data.get("request_id")
    if request_id is not None and (not isinstance(request_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{8,128}", request_id)):
        raise ValueError("request_id 必须为 8 至 128 位字母、数字、下划线或短横线")
    try:
        trade = TradeRecord.create(
            symbol=data.get("symbol", data.get("ts_code", "").split(".")[0]),
            ts_code=data["ts_code"],
            action=data.get("action", "BUY"),
            price=data.get("price"),
            quantity=float(data.get("quantity", 0)),
            position_before=float(data.get("position_before", 0)),
            position_after=float(data.get("position_after", 0)),
            strategy_id=data.get("strategy_id", ""),
            strategy_version=data.get("strategy_version", ""),
            signal_id=data.get("signal_id", ""),
            market_state=data.get("market_state", ""),
            emotion_state=data.get("emotion_state", ""),
            sector_state=data.get("sector_state", ""),
            stock_state=data.get("stock_state", ""),
            system_recommendation=data.get("system_recommendation", ""),
            user_reason=data.get("user_reason", ""),
            execution_reason=data.get("execution_reason", ""),
            result=data.get("result"),
            trade_date=data.get("trade_date", ""),
        )
        if request_id:
            trade.trade_id = "decision_" + hashlib.sha256(request_id.encode()).hexdigest()
        try:
            saved = runtime.journal.record(trade)
        except ValueError as exc:
            return _json("409 Conflict", {"status": "error", "detail": str(exc)})
        runtime.context_cache.clear()
        return _json("201 Created" if saved is trade else "200 OK", {"status": "ok", "data": saved.to_dict()})
    except KeyError as exc:
        return _json("400 Bad Request", {"status": "error", "detail": f"missing {exc.args[0]}"})


def _audit_trade(environ, runtime: Runtime, trade_id: Optional[str] = None):
    trades = runtime.journal.list()
    auditor = JournalAuditEngine()
    if trade_id:
        match = next((t for t in trades if t.trade_id == trade_id), None)
        if match is None:
            return _json("404 Not Found", {"status": "error", "detail": "trade not found"})
        result = auditor.auditor.audit(match)
        return _json("200 OK", {"status": "ok", "data": result.to_dict()})
    return _json("200 OK", {"status": "ok", "data": auditor.audit_journal(trades)})


def _profile(runtime: Runtime):
    profile = runtime.profile.load_or_create()
    profile = runtime.profile.refresh_stats_from_trades(profile)
    return _json("200 OK", {"status": "ok", "data": profile.to_dict()})


def _backtest_runs(runtime: Runtime):
    from backend.api.backtest_store import BacktestStore
    return _json("200 OK", {"status": "ok", "data": BacktestStore(runtime.root).list(limit=50)})


def _post_backtest_run(environ, runtime: Runtime):
    from backend.data.atomic import file_lock
    try:
        with file_lock(runtime.root / "backtest" / "active", timeout=0):
            return _run_backtest(environ, runtime)
    except TimeoutError:
        return _json("409 Conflict", {"status": "error", "detail": "已有回测正在运行，请等待完成后再试"})


def _run_backtest(environ, runtime: Runtime):
    data = _body(environ)
    strategy_id = data.get("strategy", "ma_cross")
    if strategy_id not in runtime.strategies:
        raise ValueError("未知策略")
    for name in ("start_date", "end_date"):
        if data.get(name):
            _validate_date(data[name])
    if data.get("start_date") and data.get("end_date") and data["start_date"] > data["end_date"]:
        raise ValueError("开始日期不可晚于结束日期")
    strategy = runtime.strategies[strategy_id]
    try:
        engine = AShareBacktestEngine(runtime.provider, strategy)
        result = engine.run(
            start_date=data.get("start_date"),
            end_date=data.get("end_date"),
        )
        metrics = calculate_metrics(result.equity_curve, trade_log=result.trade_log)
        run = {
            "run_id": f"run_{uuid.uuid4().hex[:12]}",
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "strategy_id": strategy_id,
            "strategy_version": strategy.version,
            "start_date": data.get("start_date"),
            "end_date": data.get("end_date"),
            "metrics": metrics,
            "trade_log": [t.to_dict() for t in result.trade_log],
            "equity_curve": result.equity_curve.to_dict(orient="records"),
            "drawdown_curve": result.drawdown_curve.to_dict(orient="records")
            if result.drawdown_curve is not None
            else [],
            "lookahead_issues": result.lookahead_issues or [],
        }
        from backend.api.backtest_store import BacktestStore
        BacktestStore(runtime.root).save(run)
        runtime.backtest_runs.append(run)
        runtime.backtest_runs[:] = runtime.backtest_runs[-50:]
        return _json("201 Created", {"status": "ok", "data": run})
    except ValueError:
        raise
    except Exception:
        logging.getLogger(__name__).exception("Backtest failed or could not be saved")
        return _json("500 Internal Server Error", {"status": "error", "detail": "回测未完成或结果无法保存，请查看运行日志后重试"})


def _post_paper_trade(environ, runtime: Runtime):
    data = _body(environ)
    _validate_trade(data, paper=True)
    from backend.paper.engine import IdempotencyConflict
    try:
        signal_id = data.get("signal_id", f"paper_{uuid.uuid4().hex[:10]}")
        if not signal_id.strip():
            raise ValueError("signal_id 不可为空")
        # Recover a confirmed fill before consulting today's calendar, current
        # listing or prices. A lost response remains recoverable on another day.
        order = runtime.paper.replay_order(signal_id, ts_code=data["ts_code"],
            action=data.get("action", "BUY"), price=data["price"], quantity=data["quantity"],
            trade_date=data.get("trade_date") or None)
        if order is not None:
            return _json("200 OK", {"status": "ok", "data": {"order": order.to_dict(),
                "snapshot": runtime.account_snapshot(order.trade_date).to_dict()}})
        session = get_trading_session()
        default_trade_date = session.current_trading_day()
        if not default_trade_date:
            raise ValueError("交易日历不可用")
        trade_date = _validate_date(data.get("trade_date") or default_trade_date)
        if trade_date != default_trade_date:
            raise ValueError("当前账户仅接受当前交易日，历史重放请使用独立账户")
        code = data["ts_code"]
        basic = runtime.provider.stock_basic(ts_codes=[code])
        if basic.empty or code not in set(basic["ts_code"]):
            raise ValueError("标的未在股票名单中")
        if float(data["quantity"]) % 100 and data.get("action", "BUY") == "BUY":
            raise ValueError("买入数量必须为 100 股整数倍")
        signal = TradeSignal(
            ts_code=data["ts_code"],
            trade_date=trade_date,
            action=data.get("action", "BUY"),
            price=data.get("price"),
            quantity=(
                float(data["quantity"])
                if data.get("quantity") not in (None, "")
                else None
            ),
            strength=float(data.get("strength", 1.0)),
            reason=data.get("reason", "paper manual"),
            strategy_id=data.get("strategy_id", "manual"),
            strategy_version=data.get("strategy_version", "1.1"),
            signal_id=signal_id,
        )
        from backend.backtest.execution import rejection_reason
        limits = runtime.provider.stk_limit(ts_codes=[code], trade_date=trade_date)
        if limits.empty:
            raise ValueError("当日涨跌停数据不可用，无法验证成交")
        suspended = runtime.provider.suspend_d(ts_codes=[code], trade_date=trade_date)
        bars = runtime.provider.daily(ts_codes=[code], start_date=trade_date, end_date=trade_date)
        reason = rejection_reason(signal.action, float(signal.price), limits.iloc[0].to_dict(),
            suspended=not suspended.empty, bar=bars.iloc[0].to_dict() if not bars.empty else None)
        if reason:
            from backend.paper.models import PaperOrder
            order = PaperOrder(code, trade_date, signal.action, float(signal.price), 0,
                               status="REJECTED", reason=reason)
            return _json("200 OK", {"status": "ok", "data": {
                "order": order.to_dict(), "snapshot": runtime.account_snapshot(trade_date).to_dict()}})
        order = runtime.paper.execute_signal(signal)
        if order.status == "FILLED":
            runtime.analysis_cache.clear()
            runtime.context_cache.clear()
        snapshot = runtime.account_snapshot(trade_date)
        return _json(
            "201 Created" if order.status == "FILLED" else "200 OK",
            {"status": "ok", "data": {"order": order.to_dict(), "snapshot": snapshot.to_dict()}},
        )
    except IdempotencyConflict as exc:
        return _json("409 Conflict", {"status": "error", "detail": str(exc)})
    except KeyError as exc:
        return _json("400 Bad Request", {"status": "error", "detail": f"missing {exc.args[0]}"})


def _ai_copilot(environ, runtime: Runtime):
    date = _date(environ, runtime)
    query = _query(environ).get("q")
    copilot = AICopilot(
        provider=runtime.provider,
        strategy=runtime.strategies.get("strong_sector_breakout"),
        context_cache=runtime.context_cache,
        runtime=runtime,
    )
    if query:
        result = copilot.ask(query, date)
    else:
        result = copilot.daily_briefing(date)
    return _json("200 OK", {"status": "ok", "data": result})


def _data_latest(runtime: Runtime):
    from backend.api.data_status import display_dates
    return _json(
        "200 OK",
        {
            "status": "ok",
            "data": display_dates(runtime),
        },
    )


_HISTORY_DATES: Optional[tuple[float, dict]] = None


def _data_history_dates(runtime: Runtime):
    def compute():
        from backend.api.data_status import history_dates, latest_complete
        if hasattr(runtime.provider, "history"):
            dates = runtime.provider.history.distinct_dates()
            coverage = runtime.provider.history.coverage()
        elif runtime.provider.name in ("fixture", "storage"):
            frame = runtime.provider.daily()
            dates = sorted(frame["trade_date"].astype(str).unique()) if not frame.empty else []
            coverage = {}
        else:
            dates, coverage = [], {}
        return {"dates": sorted(dates, reverse=True), "latest": dates[-1] if dates else None,
            "coverage": _history_coverage(runtime), "codes": coverage.get("codes"),
            "rows": coverage.get("rows"), "source": runtime.provider.name,
            "is_fixture": runtime.provider.name == "fixture", "latest_complete_date": latest_complete(runtime),
            "history_dates": history_dates(runtime)}
    payload = runtime.analysis_cache.get_or_compute(("history_dates", int(time.monotonic() // 5)), compute)
    return _json("200 OK", {"status": "ok", "data": payload})


def _ingest_status(runtime: Runtime):
    from backend.data.auto_ingest import status
    return _json("200 OK", {"status": "ok", "data": status(runtime)})


# ---- V1.2 personal account: risk profile and position sizing ----

def _personal_risk_profile(runtime: Runtime):
    from backend.personal import RiskProfileStore
    profile, source = RiskProfileStore(runtime.root).load()
    return _json("200 OK", {"status": "ok", "meta": {"source": source}, "data": profile.to_dict()})


def _post_personal_risk_profile(environ, runtime: Runtime):
    from backend.personal import RiskProfileStore
    profile = RiskProfileStore(runtime.root).update(_body(environ))
    return _json("200 OK", {"status": "ok", "meta": {"source": "saved"}, "data": profile.to_dict()})


def _personal_sizing(environ, runtime: Runtime):
    from backend.personal import RiskProfileStore, size_position
    q = _query(environ)
    if "entry" not in q or "stop" not in q:
        raise ValueError("请提供买入价 entry 与止损价 stop")
    profile, source = RiskProfileStore(runtime.root).load()
    result = size_position(
        profile, q["entry"], q["stop"],
        risk_pct=q.get("risk_pct"),
        market_state=q.get("market_state") or None,
        current_exposure=q.get("exposure"),
        existing_position=q.get("existing"),
        available_cash=q.get("cash"),
    )
    return _json("200 OK", {"status": "ok", "profile_source": source, "data": result})


def _personal_watchlist(runtime: Runtime):
    from backend.personal import WATCHLIST_LIMIT, WatchlistStore
    items = WatchlistStore(runtime.root).list()
    return _json("200 OK", {"status": "ok", "meta": {"limit": WATCHLIST_LIMIT},
                            "data": {"items": items, "count": len(items)}})


def _post_personal_watchlist(environ, runtime: Runtime):
    from backend.personal import WatchlistStore
    body = _body(environ)
    entry = WatchlistStore(runtime.root).add(body.get("ts_code"), body.get("name"), body.get("note"))
    return _json("200 OK", {"status": "ok", "data": entry})


def _post_personal_watchlist_remove(environ, runtime: Runtime):
    from backend.personal import WatchlistStore
    body = _body(environ)
    removed = WatchlistStore(runtime.root).remove(body.get("ts_code"))
    if not removed:
        return _json("404 Not Found", {"status": "error", "detail": "观察池中没有该股票"})
    return _json("200 OK", {"status": "ok", "data": {"removed": body.get("ts_code")}})


def _personal_watchlist_status(runtime: Runtime):
    from backend.personal.service import watchlist_status
    return _json("200 OK", {"status": "ok", "data": watchlist_status(runtime)})


def _personal_account(runtime: Runtime):
    from backend.personal.service import account_view
    return _json("200 OK", {"status": "ok", "data": account_view(runtime)})


def _personal_trades(environ, runtime: Runtime):
    from backend.personal.real_account import RealTradeStore
    q = _query(environ)
    trades = RealTradeStore(runtime.root).list(date=q.get("date") or None)
    return _json("200 OK", {"status": "ok", "data": {"trades": trades, "date": q.get("date")}})


def _post_personal_trade(environ, runtime: Runtime):
    from backend.personal import RiskProfileStore
    from backend.personal.real_account import RealTradeStore
    profile, _ = RiskProfileStore(runtime.root).load()
    trade = RealTradeStore(runtime.root).record(_body(environ), profile)
    return _json("200 OK", {"status": "ok", "data": trade})


def _post_personal_trade_void(environ, runtime: Runtime):
    from backend.personal.real_account import RealTradeStore
    body = _body(environ)
    try:
        trade = RealTradeStore(runtime.root).void(body.get("id"), body.get("reason"))
    except KeyError as exc:
        return _json("404 Not Found", {"status": "error", "detail": exc.args[0]})
    return _json("200 OK", {"status": "ok", "data": trade})


def _personal_plan(environ, runtime: Runtime):
    from backend.personal.plans import MAX_NEW_CANDIDATES, PlanStore
    q = _query(environ)
    store = PlanStore(runtime.root)
    plan = store.get(q["date"]) if q.get("date") else None
    return _json("200 OK", {"status": "ok", "meta": {"dates": store.dates(), "max_candidates": MAX_NEW_CANDIDATES},
                            "data": plan})


def _post_personal_plan(environ, runtime: Runtime):
    from backend.personal.plans import PlanStore
    return _json("200 OK", {"status": "ok", "data": PlanStore(runtime.root).save(_body(environ))})


def _personal_pretrade(environ, runtime: Runtime):
    from backend.personal.service import pretrade_check
    return _json("200 OK", {"status": "ok", "data": pretrade_check(runtime, _query(environ))})
