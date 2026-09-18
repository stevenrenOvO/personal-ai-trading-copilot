"""Shared helpers for A-share free data providers."""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from typing import Optional

import pandas as pd


def ts_to_source_code(ts_code: str) -> str:
    """Convert ``600000.SH`` to ``sh.600000``."""
    symbol, exchange = ts_code.split(".")
    return f"{exchange.lower()}.{symbol}"


def source_to_ts_code(source_code: str) -> str:
    """Convert ``sh.600000`` to ``600000.SH``."""
    exchange, symbol = source_code.split(".")
    return f"{symbol}.{exchange.upper()}"


def market_name(ts_code: str) -> str:
    if ts_code.endswith(".SH"):
        return "上海"
    if ts_code.endswith(".SZ"):
        return "深圳"
    if ts_code.endswith(".BJ"):
        return "北京"
    return ""


def limit_rate(ts_code: str, is_st: bool) -> float:
    """Return the daily price-limit ratio for an A-share code."""
    symbol = ts_code.split(".")[0]
    if ts_code.endswith(".BJ"):
        return 0.30
    if symbol.startswith(("30", "688")):
        return 0.20
    if is_st:
        return 0.05
    return 0.10


def round_price(value: float) -> float:
    """Round to 0.01 yuan using half-up (A-share convention)."""
    if value is None:
        return value
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def to_source_date(date: str) -> str:
    """Convert ``20240102`` to ``2024-01-02``."""
    return f"{date[:4]}-{date[4:6]}-{date[6:]}"


def from_source_date(date: str) -> str:
    """Convert ``2024-01-02`` to ``20240102``."""
    return str(date).replace("-", "")


def compute_limit_prices(daily_raw: pd.DataFrame) -> pd.DataFrame:
    """Compute up/down limit prices from raw daily bars with ``is_st``.

    ``daily_raw`` must contain ``trade_date``, ``ts_code``, ``pre_close`` and
    ``is_st`` columns. Limit prices are derived from exchange price-limit
    rules, not from a vendor field.

    A full-market day is ~5 200 rows and a 20-day ladder multiplies that by 20,
    so the calculation is vectorised: row-wise ``iterrows`` made the board page
    take ~20s. Rounding stays half-up (A-share convention), which numpy's
    banker's rounding would get wrong at values like 10.15 * 1.10.
    """
    columns = ["trade_date", "ts_code", "up_limit", "down_limit"]
    if daily_raw is None or daily_raw.empty:
        return pd.DataFrame(columns=columns)

    pre_close = pd.to_numeric(daily_raw.get("pre_close"), errors="coerce")
    valid = pre_close.notna() & (pre_close > 0)
    if not valid.any():
        return pd.DataFrame(columns=columns)

    rows = daily_raw.loc[valid]
    prices = pre_close.loc[valid].astype("float64")
    codes = rows["ts_code"].astype(str)
    flags = (
        rows["is_st"] if "is_st" in rows.columns else pd.Series(False, index=rows.index)
    )
    rates = pd.Series(
        [
            limit_rate(code, is_st_flag(flag))
            for code, flag in zip(codes.tolist(), flags.tolist())
        ],
        index=rows.index,
        dtype="float64",
    )
    up_limit = [round_price(value) for value in (prices * (1.0 + rates)).tolist()]
    down_limit = [round_price(value) for value in (prices * (1.0 - rates)).tolist()]

    return pd.DataFrame(
        {
            "trade_date": rows["trade_date"].astype(str),
            "ts_code": codes,
            "up_limit": up_limit,
            "down_limit": down_limit,
        }
    ).reset_index(drop=True)


_ST_TOKENS = ("1", "Y", "S", "ST", "*ST", "TRUE", "T")


def is_st_flag(value: object) -> bool:
    """Interpret the many shapes an ``is_st`` flag arrives in.

    Vendors disagree (``"1"``, ``"Y"``, ``"N"``, ``True``, ``1.0``), and the
    local history provider passes real booleans, so a plain string comparison
    would silently compute a 10% limit for an ST stock.
    """
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    try:
        if isinstance(value, (int, float)) and not pd.isna(value):
            return float(value) == 1.0
    except TypeError:  # pragma: no cover - defensive
        return False
    text = str(value).strip().upper()
    return text in _ST_TOKENS


# Main-board ST stocks are capped at 5%, but a point-in-time name snapshot goes
# stale: a stock still named "*ST xxx" may already trade on the normal 10% band.
# The cap is therefore only honoured when the stock's own recent moves fit
# inside a 5% band.
ST_RATE_MAX_MOVE = 5.6


def effective_limit_rate(
    ts_code: str,
    named_st: bool,
    recent_max_abs_pct: Optional[float] = None,
) -> float:
    """Limit ratio, correcting a stale ST flag with the stock's own history."""
    board_rate = limit_rate(ts_code, False)
    if board_rate > 0.10 or not named_st:
        return board_rate
    if recent_max_abs_pct is None:
        return limit_rate(ts_code, True)
    try:
        if pd.isna(recent_max_abs_pct):
            return limit_rate(ts_code, True)
    except TypeError:  # pragma: no cover - defensive
        return limit_rate(ts_code, True)
    if float(recent_max_abs_pct) > ST_RATE_MAX_MOVE:
        return 0.10
    return 0.05
