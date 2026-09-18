"""Shared runtime state for the WSGI API.

Selects the live provider when a valid Tushare token is configured and falls
back to the fixture provider otherwise, so the dashboard remains usable
offline without ever treating fixtures as production data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from backend.data.config import get_settings
from backend.api.memo import TTLCache
from backend.data.providers.base import DataProvider
from backend.data.providers.akshare import AkShareProvider
from backend.data.providers.baostock import BaoStockProvider
from backend.data.providers.fallback import FallbackProvider
from backend.data.providers.fixture import FixtureProvider
from backend.data.providers.history import HistoryProvider
from backend.data.providers.lazy import LazyProvider
from backend.data.providers.storage import StorageProvider
from backend.data.providers.tushare import TushareProvider
from backend.journal.trades import TradeJournal
from backend.paper.engine import PaperTradingEngine
from backend.profile.engine import ProfileEngine
from backend.strategies.health import StrategyRegistry
from backend.strategies.ma_cross import MaCrossStrategy
from backend.strategies.strong_sector_breakout import StrongSectorBreakoutStrategy

# A store has to cover a meaningful slice of the market before it is allowed to
# answer "how is the A-share market today". The accepted phase-1 CSV store only
# held the CSI300 (300 codes), which made breadth and sector ranks describe an
# index basket, so the full-market history is preferred when it exists.
MIN_FULL_MARKET_CODES = 1000


@dataclass
class Runtime:
    provider: DataProvider
    paper: PaperTradingEngine
    journal: TradeJournal
    profile: ProfileEngine
    registry: StrategyRegistry
    root: Path
    strategies: dict[str, Any] = field(default_factory=dict)
    backtest_runs: list[dict] = field(default_factory=list)
    # Derived, date-keyed analysis (market / emotion / sector / board /
    # opportunity). Lives on the runtime so every request in one process shares
    # it, and so a test's runtime can never read another test's cache.
    analysis_cache: TTLCache = field(
        default_factory=lambda: TTLCache(ttl_seconds=180.0, maxsize=128)
    )
    # The AI context also embeds positions / trades, so it expires sooner.
    context_cache: TTLCache = field(
        default_factory=lambda: TTLCache(ttl_seconds=45.0, maxsize=16)
    )

    @classmethod
    def create(cls, root: Optional[Path] = None) -> "Runtime":
        settings = get_settings()
        root = root or settings.db_path.parent
        provider = cls._choose_provider(settings, root)

        paper = PaperTradingEngine(
            t_plus_one=True, state_path=root / "paper" / "state.json"
        )
        journal = TradeJournal(root)
        paper.journal = journal

        registry = StrategyRegistry(root)
        strong = StrongSectorBreakoutStrategy()
        baseline = MaCrossStrategy()
        registry.register(strong)
        registry.register(baseline)
        strategies = {
            strong.strategy_id: strong,
            baseline.strategy_id: baseline,
        }

        return cls(
            provider=provider,
            paper=paper,
            journal=journal,
            profile=ProfileEngine(root),
            registry=registry,
            root=root,
            strategies=strategies,
        )

    @classmethod
    def _choose_provider(cls, settings, root: Path) -> DataProvider:
        if settings.has_tushare_token:
            try:
                return TushareProvider(token=settings.tushare_token)
            except Exception:
                # Never leak the token; fall back to free/local sources.
                pass

        # 1. Full-market daily history (SQLite): the only local store that can
        #    answer market-wide breadth, limit-board and sector questions.
        history_provider = cls._history_provider(root)
        if history_provider is not None:
            return history_provider

        # 2. The accepted CSI300 CSV store (fast, offline).
        if (root / "daily.csv").exists() or (root / "stock_basic.csv").exists():
            try:
                return StorageProvider(base_dir=root)
            except Exception:
                pass

        # 3. Free network providers, then test fixtures. These are wrapped so
        #    that choosing the chain never opens a vendor connection: a missing
        #    local store must not cost a 25s login timeout at start-up.
        try:
            return FallbackProvider(
                [
                    LazyProvider(BaoStockProvider, name="baostock"),
                    LazyProvider(AkShareProvider, name="akshare"),
                    LazyProvider(
                        lambda: FixtureProvider(base_dir=settings.fixture_dir),
                        name="fixture",
                    ),
                ]
            )
        except Exception:
            pass

        return FixtureProvider(base_dir=settings.fixture_dir)

    @classmethod
    def _history_provider(cls, root: Path) -> Optional[HistoryProvider]:
        """Return a full-market history provider, or None when unusable."""
        db_path = root / "history" / "market_history.db"
        if not db_path.exists():
            return None
        try:
            if (root / "daily.csv").exists():
                try:
                    fallback: Optional[DataProvider] = StorageProvider(base_dir=root)
                except Exception:
                    fallback = None
            else:
                fallback = None
            provider = HistoryProvider(base_dir=root, fallback=fallback)
            if provider.coverage().get("codes", 0) < MIN_FULL_MARKET_CODES:
                return None
            return provider
        except Exception:
            return None
