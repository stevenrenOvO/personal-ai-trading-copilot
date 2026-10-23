"""Explicit offline demo services; never reuse a real account or live feed."""
from datetime import datetime

import pandas as pd

from backend.data.config import get_settings
from backend.data.providers.fixture import FixtureProvider
from backend.market.quote_service import QuoteService, set_quote_service
from backend.market.session import TradingSession, set_trading_session


def configure_demo_services() -> None:
    settings = get_settings()
    if not settings.demo_mode:
        raise ValueError("Offline demo must be explicitly enabled")
    provider = FixtureProvider(settings.fixture_dir)
    last = provider.latest_trade_date()
    if not last:
        raise ValueError("Demo fixtures have no trading dates")
    base = settings.db_path.parent / "cache"
    def calendar(start, end):
        frame = provider.trade_cal()
        if frame.empty:
            return [(str(day), 1) for day in sorted(provider.daily()["trade_date"].unique())]
        return [(str(row.cal_date), int(row.is_open)) for row in frame.itertuples()]
    session = TradingSession(db_path=base / "session.db", calendar_fetcher=calendar,
        now_fn=lambda: datetime.strptime(last + "160000", "%Y%m%d%H%M%S"))
    session.ensure_calendar()
    set_trading_session(session)
    set_quote_service(QuoteService(db_path=base / "quotes.db",
        fetcher=lambda: (pd.DataFrame(), {"source": "fixture"})))
