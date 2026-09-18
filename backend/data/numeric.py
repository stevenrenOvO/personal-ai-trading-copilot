"""NaN-safe numeric aggregation for engine outputs.

A missing value must stay missing all the way to the API. Two failure modes
motivate these helpers:

* ``DataFrame.sum()`` returns ``0.0`` when every value is NaN, which would show
  "今日成交额 0 元" for a day whose turnover simply was not collected.
* ``float('nan')`` serializes to bare ``NaN`` in ``json.dumps``, which is not
  valid JSON and makes the browser's ``JSON.parse`` throw.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd


def safe_sum(values: object) -> Optional[float]:
    """Sum ignoring NaN; ``None`` when there is no real value at all."""
    numbers = pd.to_numeric(pd.Series(values), errors="coerce")
    if numbers.notna().sum() == 0:
        return None
    total = float(numbers.sum(skipna=True))
    return None if pd.isna(total) else total


def safe_mean(values: object) -> Optional[float]:
    """Mean ignoring NaN; ``None`` when there is no real value at all."""
    numbers = pd.to_numeric(pd.Series(values), errors="coerce")
    if numbers.notna().sum() == 0:
        return None
    mean = float(numbers.mean(skipna=True))
    return None if pd.isna(mean) else mean
