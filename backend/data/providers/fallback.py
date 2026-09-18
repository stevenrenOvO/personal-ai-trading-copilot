"""Provider fallback chain."""

from __future__ import annotations

import pandas as pd

from backend.data.providers.base import DataProvider, DataProviderError


class FallbackProvider(DataProvider):
    """Try providers in order and return the first non-empty result.

    If every provider fails for a given table, the last error is raised unless
    all providers returned empty frames, in which case an empty frame is
    returned so upstream engines can degrade gracefully.
    """

    def __init__(self, providers: list[DataProvider]) -> None:
        if not providers:
            raise ValueError("FallbackProvider requires at least one provider")
        self.providers = providers
        self.name = "+".join(p.name for p in providers)
        self.data_timeliness = getattr(providers[0], "data_timeliness", "historical")

    def _call(self, method: str, *args, **kwargs):
        last_error: Exception | None = None
        for provider in self.providers:
            fn = getattr(provider, method)
            try:
                result = fn(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                continue
            if isinstance(result, pd.DataFrame) and result.empty:
                continue
            return result
        if last_error is not None:
            raise DataProviderError(
                f"all providers failed for {method}: {last_error}"
            ) from last_error
        return pd.DataFrame()

    def stock_basic(self, *, ts_codes=None, list_status=None):
        return self._call("stock_basic", ts_codes=ts_codes, list_status=list_status)

    def daily(self, *, ts_codes=None, start_date=None, end_date=None):
        return self._call(
            "daily", ts_codes=ts_codes, start_date=start_date, end_date=end_date
        )

    def daily_basic(self, *, ts_codes=None, trade_date=None):
        return self._call("daily_basic", ts_codes=ts_codes, trade_date=trade_date)

    def stk_limit(self, *, ts_codes=None, trade_date=None):
        return self._call("stk_limit", ts_codes=ts_codes, trade_date=trade_date)

    def trade_cal(self, *, exchange=None, start_date=None, end_date=None, is_open=None):
        return self._call(
            "trade_cal",
            exchange=exchange,
            start_date=start_date,
            end_date=end_date,
            is_open=is_open,
        )

    def adj_factor(self, *, ts_codes=None, trade_date=None, start_date=None, end_date=None):
        return self._call(
            "adj_factor",
            ts_codes=ts_codes,
            trade_date=trade_date,
            start_date=start_date,
            end_date=end_date,
        )

    def suspend_d(self, *, ts_codes=None, trade_date=None, start_date=None, end_date=None):
        return self._call(
            "suspend_d",
            ts_codes=ts_codes,
            trade_date=trade_date,
            start_date=start_date,
            end_date=end_date,
        )

    def latest_trade_date(self):
        for provider in self.providers:
            try:
                date = provider.latest_trade_date()
            except Exception:
                continue
            if date:
                return date
        return None
