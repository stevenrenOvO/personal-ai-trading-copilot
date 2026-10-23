"""System context builder for the AI layer.

Aggregates the live system state (market, emotion, sectors, opportunities,
board, risk, positions, strategies, trades, profile) into a plain dict that
tools can consume without importing vendor clients.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
import json

from backend.board.engine import BoardEngine
from backend.data.config import get_settings
from backend.data.providers.base import DataProvider
from backend.data.providers.fixture import FixtureProvider
from backend.emotion.engine import EmotionEngine
from backend.journal.audit import JournalAuditEngine
from backend.journal.trades import TradeJournal
from backend.market.engine import MarketEngine
from backend.market.quote_service import get_quote_service
from backend.market.snapshot import (
    compute_full_market_board_environment,
    compute_full_market_sectors,
    compute_full_market_state,
)
from backend.opportunity.ranking import OpportunityRankingEngine
from backend.paper.engine import PaperTradingEngine
from backend.profile.engine import ProfileEngine
from backend.profile.models import UserProfile
from backend.risk.rules import DecisionRiskEngine, RiskContext
from backend.sector.engine import SectorEngine
from backend.strategies.base import Strategy
from backend.strategies.strong_sector_breakout import StrongSectorBreakoutStrategy


def build_context(
    date: str,
    *,
    provider: Optional[DataProvider] = None,
    strategy: Optional[Strategy] = None,
    root: Optional[Path] = None,
    runtime: Optional[Any] = None,
) -> dict[str, Any]:
    """Build a grounded context snapshot for one trading day."""
    provider = provider or FixtureProvider(base_dir=get_settings().fixture_dir)
    strategy = strategy or StrongSectorBreakoutStrategy()
    root = root or get_settings().db_path.parent
    if runtime is not None:
        from backend.api import v1
        environ = {"QUERY_STRING": f"date={date}"}
        def read(handler):
            return json.loads(handler(environ, runtime)[2])
        snapshot = runtime.account_snapshot(date)
        runtime.paper.sync_journal()
        trades = runtime.journal.list()
        profile = runtime.profile.refresh_stats_from_trades(runtime.profile.load_or_create())
        payloads = {key: read(handler) for key, handler in (
            ("market", v1._market), ("emotion", v1._emotion), ("sectors", v1._sectors),
            ("board", v1._board), ("risk", v1._risk), ("trade_plan", v1._opportunity_plan))}
        plan = payloads["trade_plan"]["data"]
        return {"date": date, "data_mode": "shared_runtime",
            **{key: payload["data"] for key, payload in payloads.items()},
            "data_meta": {key: payload.get("meta", {}) for key, payload in payloads.items()},
            "opportunities": plan.get("plans", []),
            "positions": snapshot.to_dict(), "strategies": [
                {"strategy_id": s.strategy_id, "version": s.version,
                 "status": s.status, "enabled": s.enabled} for s in runtime.strategies.values()],
            "trades": [t.to_dict() for t in trades],
            "audit": JournalAuditEngine().audit_journal(trades), "profile": profile.to_dict()}

    market_engine = MarketEngine(provider)
    emotion_engine = EmotionEngine(provider)
    sector_engine = SectorEngine(provider)
    board_engine = BoardEngine(provider)

    # Prefer the same full-market snapshot path the dashboard uses so the AI,
    # the API and the UI never disagree about the current market.
    full_df = _try_full_market(provider, date)
    if full_df is not None:
        market = compute_full_market_state(full_df, date)
        emotion = emotion_engine.calculate(date, market=market)
        sectors = compute_full_market_sectors(full_df, _industry_map(provider))[:5]
        board = compute_full_market_board_environment(full_df, date)
        data_mode = "full_market_snapshot"
    else:
        market = market_engine.calculate(date)
        emotion = emotion_engine.calculate(date, market=market)
        sectors = [s.to_dict() for s in sector_engine.ranked(date, top_n=5)]
        board = board_engine.calculate(date).to_dict()
        data_mode = "daily_history"

    paper = PaperTradingEngine(state_path=root / "paper" / "state.json")
    snapshot = paper.snapshot()
    trades = TradeJournal(root).list()
    profile = ProfileEngine(root).load_or_create()
    audit = JournalAuditEngine().audit_journal(trades)

    risk = DecisionRiskEngine().decide(
        RiskContext(
            equity=snapshot.equity,
            cash=snapshot.cash,
            peak_equity=max(snapshot.equity, 1_000_000.0),
        ),
        market=market,
        emotion=emotion,
        has_positions=bool(snapshot.positions),
    )
    plan = OpportunityRankingEngine(provider).build(date, state={
        "equity": snapshot.equity, "cash": snapshot.cash, "positions": snapshot.positions}).to_dict()

    return {
        "date": date,
        "data_mode": data_mode,
        "market": market.to_dict(),
        "emotion": emotion.to_dict(),
        "sectors": list(sectors),
        "opportunities": plan.get("plans", []),
        "trade_plan": plan,
        "data_meta": {"trade_plan": {"status": "ok" if plan.get("available") else "no_data",
            "kind": "historical_scan", "as_of": date, "source": provider.name}},
        "board": board,
        "risk": risk.to_dict(),
        "positions": snapshot.to_dict(),
        "strategies": [
            {
                "strategy_id": strategy.strategy_id,
                "version": strategy.version,
                "status": strategy.status,
                "enabled": strategy.enabled,
            }
        ],
        "trades": [t.to_dict() for t in trades],
        "audit": audit,
        "profile": profile.to_dict(),
    }


def _try_full_market(provider: DataProvider, date: str) -> Optional[Any]:
    """Return the full-market snapshot when it covers the requested date."""
    try:
        latest = provider.latest_trade_date()
    except Exception:
        latest = None
    if not latest or date < latest:
        return None
    try:
        df, meta = get_quote_service().get()
    except Exception:
        return None
    return None if (df is None or df.empty or meta.is_stale
                    or (getattr(meta, "quote_date", None) or str(meta.quote_time or "")[:8]) != date) else df


def _industry_map(provider: DataProvider) -> dict[str, str]:
    try:
        basic = provider.stock_basic(list_status="L")
    except Exception:
        basic = provider.stock_basic()
    mapping: dict[str, str] = {}
    if not basic.empty and "ts_code" in basic.columns and "industry" in basic.columns:
        for _, row in basic.iterrows():
            industry = str(row.get("industry", "") or "")
            if industry:
                mapping[str(row["ts_code"])] = industry
    return mapping
