"""V1.0 REST-ish endpoints on top of the standard-library WSGI server."""

from __future__ import annotations

import json
import re
import time
import uuid
from datetime import datetime, timezone
from urllib.parse import parse_qs
from typing import Any, Optional

from backend.ai.agent import AICopilot
from backend.api.runtime import Runtime
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
    return status, [("Content-Type", "application/json; charset=utf-8")], json.dumps(
        data, ensure_ascii=False
    ).encode("utf-8")


def _query(environ: dict) -> dict[str, str]:
    raw = environ.get("QUERY_STRING", "")
    parsed = parse_qs(raw)
    return {k: v[0] for k, v in parsed.items() if v}


def _body(environ: dict) -> dict:
    try:
        length = int(environ.get("CONTENT_LENGTH") or 0)
        raw = environ["wsgi.input"].read(length).decode("utf-8")
        return json.loads(raw) if raw else {}
    except Exception:
        return {}


def _date(environ: dict, runtime: Runtime | None = None, default: str = "20240103") -> str:
    q = _query(environ)
    if "date" in q:
        return q["date"]
    if runtime is not None:
        latest = _latest_date(runtime)
        if latest:
            return latest
    return default


def dispatch(environ: dict, start_response, runtime: Runtime) -> bool:
    path = environ.get("PATH_INFO", "")
    method = environ.get("REQUEST_METHOD", "GET")
    q = _query(environ)

    # 收盘后自动补当天日线（每天一次、后台执行、不阻塞请求）。
    try:
        from backend.data.auto_ingest import ensure_today_bars

        ensure_today_bars(runtime)
    except Exception:  # noqa: BLE001 - ingestion must never break serving
        pass

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
        "GET /ai/copilot": lambda: _ai_copilot(environ, runtime),
    }

    key = f"{method} {path}"
    if key in handlers:
        status, headers, body = handlers[key]()
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
    # The live snapshot only describes the CURRENT trading day. Using it for any
    # other date would mislabel today's data with an old date.
    if latest and date == latest:
        df, meta = get_quote_service().get()
        if not df.empty:
            state = compute_full_market_state(df, date)
            return _json("200 OK", {"status": "ok", "data": state.to_dict(), "meta": _snapshot_meta(meta)})
        # Snapshot not ready yet: say so explicitly instead of silently using
        # a different (CSI300) data source.
        state = MarketState(date=date, state="数据准备中", confidence=0.0, risk_level="high")
        return _json("200 OK", {"status": "ok", "data": state.to_dict(), "meta": _snapshot_meta(meta)})
    state = _market_state(runtime, date)
    return _json("200 OK", {"status": "ok", "data": state.to_dict(), "meta": _historical_meta(date)})


def _market_state(runtime: Runtime, date: str):
    """Full-market state for one day, memoised on the runtime."""
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
    return runtime.analysis_cache.get_or_compute(
        ("ladder", date),
        lambda: LimitLadderEngine(runtime.provider).snapshot(date),
    )


def _board_ladder(environ, runtime: Runtime):
    date = _date(environ, runtime)
    snapshot = _ladder_snapshot(runtime, date)
    meta = _historical_meta(date, kind="historical_scan")
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
        fallback = _history_latest_date()
        if fallback:
            basis = fallback
            basis_note = (
                f"{date} 的行情尚未落地到本地历史（本地覆盖 {_history_coverage() or '—'}），"
                f"情绪阶段基于最近确认交易日 {fallback} 的真实结构"
            )
    ladder = _ladder_snapshot(runtime, basis)
    market = _market_state(runtime, basis)
    state = _emotion_state(runtime, basis, market, ladder)
    payload = state.to_dict()
    meta = _historical_meta(basis)
    if basis_note:
        meta["status"] = "no_data"
        meta["error"] = basis_note
        payload["basis_note"] = basis_note
    return _json("200 OK", {"status": "ok", "data": payload, "meta": meta})


def _sectors(environ, runtime: Runtime):
    date = _date(environ, runtime)
    top = int(_query(environ).get("top", 5))
    latest = _latest_date(runtime)
    if latest and date == latest:
        df, meta = get_quote_service().get()
        if not df.empty:
            industry = _industry_map(runtime)
            sectors = compute_full_market_sectors(df, industry)[:top]
            return _json("200 OK", {"status": "ok", "data": sectors, "meta": _snapshot_meta(meta)})
        return _json("200 OK", {"status": "ok", "data": [], "meta": _snapshot_meta(meta)})
    sectors = _sector_scores(runtime, date, top)
    return _json(
        "200 OK",
        {"status": "ok", "data": [s.to_dict() for s in sectors], "meta": _historical_meta(date)},
    )


def _opportunities(environ, runtime: Runtime):
    date = _date(environ, runtime)
    if _local_history_gap(runtime, date):
        # No bars for this day yet (typical before/while the market is open):
        # the daily-bar strategy cannot be evaluated at all. Rather than an
        # empty list that reads like a market verdict, show the most recent
        # completed day and label exactly which day that is.
        fallback = _history_latest_date()
        if fallback and fallback != date:
            opps = _scan_opportunities(runtime, fallback)
            meta = _historical_meta(fallback, kind="historical_scan")
            meta["status"] = "no_data"
            meta["error"] = (
                f"{date} 的行情尚未落地到本地历史（本地覆盖 {_history_coverage() or '—'}），"
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
                    **_historical_meta(date, kind="historical_scan"),
                    "status": "no_data",
                    "error": (
                        f"该交易日尚未落地到本地历史（本地覆盖 {_history_coverage() or '—'}），"
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
            "meta": _historical_meta(date, kind="historical_scan"),
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
        snapshot = runtime.paper.snapshot()
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

    return runtime.analysis_cache.get_or_compute(("plan", date), build)


def _opportunity_plan(environ, runtime: Runtime):
    date = _date(environ, runtime)
    if _local_history_gap(runtime, date):
        fallback = _history_latest_date()
        if fallback and fallback != date:
            plan = _trade_plan(runtime, fallback)
            meta = _historical_meta(fallback, kind="historical_scan")
            meta["status"] = "no_data"
            meta["error"] = (
                f"{date} 的行情尚未落地到本地历史（本地覆盖 {_history_coverage() or '—'}），"
                f"以下为最近交易日 {fallback} 的交易计划"
            )
            return _json(
                "200 OK", {"status": "ok", "data": plan.to_dict(), "meta": meta}
            )
        return _json(
            "200 OK",
            {
                "status": "ok",
                "data": RankedPlan(date=date, available=False).to_dict(),
                "meta": {
                    **_historical_meta(date, kind="historical_scan"),
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
            "meta": _historical_meta(date, kind="historical_scan"),
        },
    )


def _opportunity_setups(environ, runtime: Runtime):
    date = _date(environ, runtime)
    if _local_history_gap(runtime, date):
        fallback = _history_latest_date()
        if fallback and fallback != date:
            result = _typed_opportunities(runtime, fallback)
            meta = _historical_meta(fallback, kind="historical_scan")
            meta["status"] = "no_data"
            meta["error"] = (
                f"{date} 的行情尚未落地到本地历史（本地覆盖 {_history_coverage() or '—'}），"
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
                    **_historical_meta(date, kind="historical_scan"),
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
            "meta": _historical_meta(date, kind="historical_scan"),
        },
    )


def _opportunity_candidates(environ, runtime: Runtime):
    date = _date(environ, runtime)
    if _local_history_gap(runtime, date):
        fallback = _history_latest_date()
        if fallback and fallback != date:
            pool = _candidate_pool(runtime, fallback)
            meta = _historical_meta(fallback, kind="historical_scan")
            meta["status"] = "no_data"
            meta["error"] = (
                f"{date} 的行情尚未落地到本地历史（本地覆盖 {_history_coverage() or '—'}），"
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
                    **_historical_meta(date, kind="historical_scan"),
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
            "meta": _historical_meta(date, kind="historical_scan"),
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


def _history_latest_date() -> Optional[str]:
    """Newest trading day the local store can answer for."""
    info = _history_info()
    return str(info["end"]) if info and info.get("end") else None


def _local_history_gap(runtime: Runtime, date: str) -> bool:
    """True when the local store is the source but has no bars for ``date``."""
    from backend.data.providers.history import HistoryProvider

    if not isinstance(runtime.provider, HistoryProvider):
        return False
    return _history_rows_on(date) == 0


def _board(environ, runtime: Runtime):
    date = _date(environ, runtime)
    latest = _latest_date(runtime)
    if latest and date == latest:
        df, meta = get_quote_service().get()
        if not df.empty:
            data = _full_market_board(date, df, runtime)
            return _json("200 OK", {"status": "ok", "data": data, "meta": _snapshot_meta(meta)})
        return _json(
            "200 OK",
            {
                "status": "ok",
                "data": {"date": date, "grade": "—", "limit_up_count": None, "ladder_available": False,
                         "ladder_note": "当日数据准备中"},
                "meta": _snapshot_meta(meta),
            },
        )
    env = _board_state(runtime, date)
    return _json("200 OK", {"status": "ok", "data": env.to_dict(), "meta": _historical_meta(date)})


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


def _snapshot_meta(meta) -> dict:
    payload = meta.to_dict()
    try:
        payload["session_phase"] = get_trading_session().phase()
    except Exception:
        payload["session_phase"] = "未知"
    return payload


def _historical_meta(date: str, kind: str = "historical") -> dict:
    coverage = _history_coverage()
    return {
        "source": "local_history",
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


_HISTORY_INFO: Optional[tuple[float, Optional[dict]]] = None
_COVERAGE_TTL_SECONDS = 300.0


def _history_info() -> Optional[dict]:
    """Cached coverage of the full-market history store.

    Cached for a few minutes only: backfilling more days while the server runs
    must not leave the banner lying about how far the local data reaches.
    """
    global _HISTORY_INFO
    now = time.monotonic()
    if _HISTORY_INFO is not None and now - _HISTORY_INFO[0] < _COVERAGE_TTL_SECONDS:
        return _HISTORY_INFO[1]

    info: Optional[dict] = None
    try:
        from backend.data.history import MarketHistory

        cov = MarketHistory().coverage()
        if cov.get("start") and cov.get("end"):
            info = cov
    except Exception:  # noqa: BLE001
        info = None
    _HISTORY_INFO = (now, info)
    return info


def _history_coverage() -> Optional[str]:
    """Human-readable coverage, for honest empty-state text."""
    info = _history_info()
    if not info:
        return None
    return f"{info['start']}~{info['end']}，{info.get('codes', 0)} 个代码"


_HISTORY_DAY_CACHE: dict[str, tuple[float, int]] = {}


def _history_rows_on(date: str) -> int:
    """How many bars the local store holds for one trading day (0 = not collected)."""
    now = time.monotonic()
    cached = _HISTORY_DAY_CACHE.get(date)
    if cached is not None and now - cached[0] < _COVERAGE_TTL_SECONDS:
        return cached[1]
    try:
        from backend.data.history import MarketHistory

        rows = MarketHistory().rows_on(date)
    except Exception:
        rows = 0
    _HISTORY_DAY_CACHE[date] = (now, rows)
    return rows


def _history_board_extras(runtime: Runtime, date: str) -> Optional[dict]:
    """Ladder / leader / yesterday-premium for ``date`` from the local store.

    The live snapshot can answer "涨停多少家" but not "几连板、谁是龙头" -- those
    need several days of bars. When the local store already holds this day, the
    ladder is computed from it and labelled as a local-history figure so it is
    never confused with the live snapshot.
    """
    if _history_rows_on(date) < 100:
        return None
    try:
        env = _board_state(runtime, date)
    except Exception:
        return None
    if env.grade == "D" and env.factors == ("数据不足",):
        return None
    return {
        "ladder": env.ladder.to_dict(),
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
    return compute_full_market_board_environment(df, date, history_extras=extras)


def _risk(environ, runtime: Runtime):
    date = _date(environ, runtime)
    market = _market_state(runtime, date)
    emotion = _emotion_state(runtime, date, market)
    snapshot = runtime.paper.snapshot()
    risk = DecisionRiskEngine().decide(
        RiskContext(
            equity=snapshot.equity,
            cash=snapshot.cash,
            peak_equity=max(snapshot.equity, runtime.paper.initial_cash),
        ),
        market=market,
        emotion=emotion,
        has_positions=bool(snapshot.positions),
    )
    return _json("200 OK", {"status": "ok", "data": risk.to_dict()})


def _positions(environ, runtime: Runtime):
    snapshot = runtime.paper.snapshot()
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
        )
        runtime.journal.record(trade)
        return _json("201 Created", {"status": "ok", "data": trade.to_dict()})
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
    return _json("200 OK", {"status": "ok", "data": runtime.backtest_runs})


def _post_backtest_run(environ, runtime: Runtime):
    data = _body(environ)
    strategy_id = data.get("strategy", "ma_cross")
    strategy = runtime.strategies.get(strategy_id, runtime.strategies["ma_cross"])
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
        runtime.backtest_runs.append(run)
        return _json("201 Created", {"status": "ok", "data": run})
    except Exception as exc:
        return _json("500 Internal Server Error", {"status": "error", "detail": str(exc)})


def _post_paper_trade(environ, runtime: Runtime):
    data = _body(environ)
    try:
        default_trade_date = runtime.provider.latest_trade_date() or "20240103"
        signal = TradeSignal(
            ts_code=data["ts_code"],
            trade_date=data.get("trade_date", default_trade_date),
            action=data.get("action", "BUY"),
            price=data.get("price"),
            quantity=(
                float(data["quantity"])
                if data.get("quantity") not in (None, "")
                else None
            ),
            strength=float(data.get("strength", 1.0)),
            reason=data.get("reason", "paper manual"),
            strategy_id=data.get("strategy_id", "strong_sector_breakout"),
            strategy_version=data.get("strategy_version", "1.0"),
            signal_id=data.get("signal_id", f"paper_{uuid.uuid4().hex[:10]}"),
        )
        order = runtime.paper.execute_signal(signal)
        snapshot = runtime.paper.snapshot()
        return _json(
            "201 Created" if order.status == "FILLED" else "200 OK",
            {"status": "ok", "data": {"order": order.to_dict(), "snapshot": snapshot.to_dict()}},
        )
    except KeyError as exc:
        return _json("400 Bad Request", {"status": "error", "detail": f"missing {exc.args[0]}"})


def _ai_copilot(environ, runtime: Runtime):
    date = _date(environ, runtime)
    query = _query(environ).get("q")
    copilot = AICopilot(
        provider=runtime.provider,
        strategy=runtime.strategies.get("strong_sector_breakout"),
        context_cache=runtime.context_cache,
    )
    if query:
        result = copilot.ask(query, date)
    else:
        result = copilot.daily_briefing(date)
    return _json("200 OK", {"status": "ok", "data": result})


def _data_latest(runtime: Runtime):
    latest = _latest_date(runtime)
    try:
        info = get_trading_session().describe()
    except Exception:
        info = {}
    return _json(
        "200 OK",
        {
            "status": "ok",
            "data": {"date": latest, "phase": info.get("phase"), "now": info.get("now")},
        },
    )


_HISTORY_DATES: Optional[tuple[float, dict]] = None


def _data_history_dates(_runtime: Runtime):
    """Trading days the local store can actually answer for.

    The dashboard used a free date input, so picking a day outside the local
    coverage returned the same empty "数据不足" page for every date -- which
    reads as "the data never changes". Offering the real list removes that trap.
    """
    global _HISTORY_DATES
    now = time.monotonic()
    if _HISTORY_DATES is not None and now - _HISTORY_DATES[0] < _COVERAGE_TTL_SECONDS:
        return _json("200 OK", {"status": "ok", "data": _HISTORY_DATES[1]})

    payload: dict = {"dates": [], "latest": None, "coverage": None}
    try:
        from backend.data.history import MarketHistory

        history = MarketHistory()
        dates = history.distinct_dates()
        coverage = history.coverage()
        payload = {
            "dates": sorted(dates, reverse=True),
            "latest": dates[-1] if dates else None,
            "coverage": _history_coverage(),
            "codes": coverage.get("codes"),
            "rows": coverage.get("rows"),
        }
    except Exception as exc:  # noqa: BLE001
        payload["error"] = str(exc)
    _HISTORY_DATES = (now, payload)
    return _json("200 OK", {"status": "ok", "data": payload})
