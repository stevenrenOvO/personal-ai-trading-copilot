"""System context builder for the AI layer.

Aggregates the live system state (market, emotion, sectors, opportunities,
board, risk, positions, strategies, trades, profile) into a plain dict that
tools can consume without importing vendor clients.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

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
from backend.opportunity.engine import OpportunityEngine
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
) -> dict[str, Any]:
    """Build a grounded context snapshot for one trading day."""
    provider = provider or FixtureProvider(base_dir=get_settings().fixture_dir)
    strategy = strategy or StrongSectorBreakoutStrategy()
    root = root or get_settings().db_path.parent

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

    opportunity_engine = OpportunityEngine(provider, strategy, sector_engine)
    opportunities = [o.to_dict() for o in opportunity_engine.scan(date)]

    paper = PaperTradingEngine()
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

    return {
        "date": date,
        "data_mode": data_mode,
        "market": market.to_dict(),
        "emotion": emotion.to_dict(),
        "sectors": list(sectors),
        "opportunities": opportunities,
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
        df, _meta = get_quote_service().get()
    except Exception:
        return None
    return None if df is None or df.empty else df


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
