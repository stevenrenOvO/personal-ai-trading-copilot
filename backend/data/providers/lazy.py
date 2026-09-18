"""Deferred provider construction.

Building a vendor provider is not free: ``BaoStockProvider`` logs in over the
network in its constructor. Putting such providers directly into the fallback
chain meant the app paid that cost just to *decide* the chain -- an offline
machine (and every offline test) blocked on a 25s socket timeout, and a slow
vendor login delayed start-up.

``LazyProvider`` keeps the same interface but only constructs the real provider
when a data method is actually called.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import pandas as pd

from backend.data.providers.base import DataProvider


class LazyProvider(DataProvider):
    """Wrap a provider factory; connect on first real request."""

    def __init__(self, factory: Callable[[], DataProvider], *, name: Optional[str] = None) -> None:
        self._factory = factory
        self._provider: Optional[DataProvider] = None
        self.name = name or "lazy"
        self.data_timeliness = "historical"

    def _get(self) -> DataProvider:
        if self._provider is None:
            self._provider = self._factory()
        return self._provider

    @property
    def is_connected(self) -> bool:
        """True once the real provider has been constructed."""
        return self._provider is not None

    def _delegate(self, method: str, /, **kwargs: Any) -> pd.DataFrame:
        return getattr(self._get(), method)(**kwargs)

    def stock_basic(self, *, ts_codes=None, list_status=None) -> pd.DataFrame:
        return self._delegate("stock_basic", ts_codes=ts_codes, list_status=list_status)

    def daily(self, *, ts_codes=None, start_date=None, end_date=None) -> pd.DataFrame:
        return self._delegate(
            "daily", ts_codes=ts_codes, start_date=start_date, end_date=end_date
        )

    def daily_basic(self, *, ts_codes=None, trade_date=None) -> pd.DataFrame:
        return self._delegate("daily_basic", ts_codes=ts_codes, trade_date=trade_date)

    def stk_limit(self, *, ts_codes=None, trade_date=None) -> pd.DataFrame:
        return self._delegate("stk_limit", ts_codes=ts_codes, trade_date=trade_date)

    def trade_cal(self, *, exchange=None, start_date=None, end_date=None, is_open=None):
        return self._delegate(
            "trade_cal",
            exchange=exchange,
            start_date=start_date,
            end_date=end_date,
            is_open=is_open,
        )

    def adj_factor(self, *, ts_codes=None, trade_date=None, start_date=None, end_date=None):
        return self._delegate(
            "adj_factor",
            ts_codes=ts_codes,
            trade_date=trade_date,
            start_date=start_date,
            end_date=end_date,
        )

    def suspend_d(self, *, ts_codes=None, trade_date=None, start_date=None, end_date=None):
        return self._delegate(
            "suspend_d",
            ts_codes=ts_codes,
            trade_date=trade_date,
            start_date=start_date,
            end_date=end_date,
        )

    def latest_trade_date(self):
        return self._get().latest_trade_date()
