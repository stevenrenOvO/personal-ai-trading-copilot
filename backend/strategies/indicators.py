"""Technical indicators used by strategies."""

from __future__ import annotations

import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    """Return a simple moving average.

    Values before the window is full are ``NaN``.
    """

    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series")
    if window <= 0:
        raise ValueError("window must be positive")

    return series.rolling(window=window, min_periods=window).mean()
