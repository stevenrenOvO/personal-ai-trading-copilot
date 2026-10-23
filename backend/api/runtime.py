"""Shared runtime state for the WSGI API.

Prefers locally ingested real data and uses a lazy network fallback chain.
Fixtures require an explicit demo setting and never enter production fallback.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
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
        paper.sync_journal()

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

    def account_snapshot(self, date: str | None = None):
        """Value the shared account using quotes for the requested day or prior bars."""
        import math
        def valid_price(value):
            try:
                return not isinstance(value, bool) and math.isfinite(float(value)) and float(value) > 0
            except (TypeError, ValueError):
                return False
        codes = [code for code, p in self.paper.snapshot().positions.items() if p.quantity > 0]
        prices = {}
        price_dates = {}
        sources = {}
        if codes:
            if date is None:
                from backend.market.session import get_trading_session
                date = get_trading_session().current_trading_day()
            try:
                frame = self.provider.daily(ts_codes=codes, end_date=date)
                if not frame.empty:
                    frame = frame[frame["ts_code"].isin(codes) & (frame["trade_date"].astype(str) <= date) & frame["close"].map(valid_price)]
                    last = frame.sort_values("trade_date").groupby("ts_code").tail(1).set_index("ts_code")
                    prices = last["close"].to_dict()
                    price_dates = last["trade_date"].astype(str).to_dict()
                    sources = {code: getattr(self.provider, "name", "unknown") for code in prices}
            except Exception:
                pass
            try:
                from backend.market.quote_service import get_quote_service
                frame, meta = get_quote_service().get()
                quote_date = getattr(meta, "quote_date", None) or str(meta.quote_time or "")[:8]
                if not meta.is_stale and quote_date and date and quote_date <= date and not frame.empty:
                    frame = frame[frame["ts_code"].isin(codes) & frame["price"].map(valid_price)]
                    frame = frame[frame["ts_code"].map(lambda code: quote_date >= price_dates.get(code, ""))]
                    prices.update(frame.set_index("ts_code")["price"].to_dict())
                    for code in frame["ts_code"]:
                        price_dates[code] = quote_date
                        sources[code] = meta.source
            except Exception:
                pass
        prices = {code: float(price) for code, price in prices.items()
                  if valid_price(price)}
        snapshot = self.paper.snapshot(prices)
        codes = [code for code, position in snapshot.positions.items() if position.quantity > 0]
        return replace(snapshot, valuation={"requested_date": date,
            "prices": prices, "price_dates": price_dates, "sources": sources,
            "missing_codes": [code for code in codes if code not in prices],
            "older_price_codes": [code for code in prices if price_dates.get(code) != date]})

    @classmethod
    def _choose_provider(cls, settings, root: Path) -> DataProvider:
        if getattr(settings, "demo_mode", False):
            return FixtureProvider(base_dir=settings.fixture_dir)

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

        # 3. Network providers. Production never falls back to test fixtures.
        #    that choosing the chain never opens a vendor connection: a missing
        #    local store must not cost a 25s login timeout at start-up.
        chain = [LazyProvider(BaoStockProvider, name="baostock"),
                 LazyProvider(AkShareProvider, name="akshare")]
        if settings.has_tushare_token:
            chain.insert(0, LazyProvider(lambda: TushareProvider(token=settings.tushare_token), name="tushare"))
        return FallbackProvider(chain)

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
            # A newly initialized catalog may have no bars yet. Select it for
            # resumable background ingestion; API coverage gates still prevent
            # partial data from being presented as a market-wide conclusion.
            if (provider.coverage().get("codes", 0) < MIN_FULL_MARKET_CODES
                    and len(provider.stock_basic()) < MIN_FULL_MARKET_CODES):
                return None
            return provider
        except Exception:
            return None
