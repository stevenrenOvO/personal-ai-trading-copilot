"""Fast unadjusted daily bars from Tencent's public k-line endpoint.

Used to fill missing trading days quickly (~0.1s per stock vs ~0.4s for
BaoStock). The ``day`` series is unadjusted, so it stays consistent with the
BaoStock history (adjustflag=3). Tencent does not expose a per-day turnover
value, so ``amount`` is left empty rather than estimated.

Corporate-action guard: on an ex-dividend / bonus-share day (``cqr``) the raw
previous close is *not* a valid basis for a percentage change, so a naive
``close / pre_close - 1`` turns a 10-for-4.5 bonus issue into a fake -31.9%
crash. Such bars keep their real OHLCV but get ``change``/``pct_chg`` left
empty, because the true adjusted move is unknown from this source.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import requests

_URL = "https://web.ifzq.gtimg.cn/appstock/app/kline/kline"

# Extra slack for prices that round to the limit (e.g. 0.73 -> 0.58 is -20.55%
# on a 20% board), so only genuine corporate actions are flagged.
_LIMIT_SLACK = 0.6


def _limit_pct(ts_code: str) -> float:
    symbol, _, exchange = ts_code.partition(".")
    if exchange.upper() == "BJ":
        return 30.0
    if symbol.startswith(("30", "688")):
        return 20.0
    return 10.0


def _code(ts_code: str) -> str:
    symbol, _, exchange = ts_code.partition(".")
    return f"{exchange.lower()}{symbol}"


def _iso(date_key: str) -> str:
    return f"{date_key[:4]}-{date_key[4:6]}-{date_key[6:]}"


def fetch_daily(
    ts_code: str,
    start_date: str,
    end_date: str,
    *,
    timeout: float = 15.0,
    pad_days: int = 20,
) -> pd.DataFrame:
    """Return unadjusted daily bars for one stock in the internal schema.

    The request is padded backwards so the first row of the requested range
    still has a computable ``pre_close``.
    """
    code = _code(ts_code)
    padded_start = (
        datetime.strptime(start_date, "%Y%m%d") - timedelta(days=pad_days)
    ).strftime("%Y-%m-%d")
    params = {"param": f"{code},day,{padded_start},{_iso(end_date)},40,"}
    resp = requests.get(_URL, params=params, timeout=timeout)
    node = ((resp.json() or {}).get("data") or {}).get(code) or {}
    rows = node.get("day") or []
    if not rows:
        return pd.DataFrame()

    records: list[dict] = []
    corporate_action_rows = 0
    prev_close: Optional[float] = None
    limit_pct = _limit_pct(ts_code)
    for row in rows:
        if len(row) < 6:
            continue
        date = str(row[0]).replace("-", "")
        open_, close, high, low = (
            _to_float(row[1]),
            _to_float(row[2]),
            _to_float(row[3]),
            _to_float(row[4]),
        )
        volume = _to_float(row[5])
        if None in (open_, close, high, low):
            prev_close = close
            continue
        change = round(close - prev_close, 4) if prev_close else None
        pct_chg = (
            round(change / prev_close * 100.0, 4)
            if change is not None and prev_close
            else None
        )
        if pct_chg is not None and abs(pct_chg) > limit_pct + _LIMIT_SLACK:
            # Ex-dividend / split day: the raw move cannot be a real trade move.
            corporate_action_rows += 1
            change = None
            pct_chg = None
        if start_date <= date <= end_date:
            records.append(
                {
                    "trade_date": date,
                    "ts_code": ts_code,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "pre_close": prev_close,
                    "change": change,
                    "pct_chg": pct_chg,
                    "vol": volume * 100.0 if volume is not None else None,
                    "amount": None,   # Tencent daily k-line has no turnover value
                }
            )
        prev_close = close
    frame = pd.DataFrame(records)
    frame.attrs["corporate_action_rows"] = corporate_action_rows
    return frame


def _to_float(value: object) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
