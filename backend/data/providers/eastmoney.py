"""Fast daily-bar backfill from Eastmoney's public kline endpoint.

Used for gap filling: it returns *unadjusted* (fqt=0) daily bars that match the
BaoStock history (adjustflag=3), and it is roughly 10x faster per stock than
BaoStock, so a few missing trading days can be filled in minutes.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import requests

_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
_FIELDS2 = "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61"


def _secid(ts_code: str) -> str:
    symbol, _, exchange = ts_code.partition(".")
    prefix = "1" if exchange.upper() == "SH" else "0"
    return f"{prefix}.{symbol}"


def fetch_daily(
    ts_code: str,
    start_date: str,
    end_date: str,
    *,
    timeout: float = 15.0,
) -> pd.DataFrame:
    """Return unadjusted daily bars for one stock in the internal schema."""
    params = {
        "secid": _secid(ts_code),
        "klt": "101",       # daily
        "fqt": "0",         # unadjusted, matches the BaoStock history
        "beg": start_date,
        "end": end_date,
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": _FIELDS2,
    }
    resp = requests.get(_URL, params=params, timeout=timeout)
    payload = resp.json() or {}
    node = payload.get("data") or {}
    klines = node.get("klines") or []
    if not klines:
        return pd.DataFrame()

    records = []
    for line in klines:
        parts = str(line).split(",")
        if len(parts) < 11:
            continue
        date = parts[0].replace("-", "")
        open_, close, high, low = (_to_float(parts[1]), _to_float(parts[2]),
                                   _to_float(parts[3]), _to_float(parts[4]))
        volume = _to_float(parts[5])          # 手
        amount = _to_float(parts[6])          # 元
        pct_chg = _to_float(parts[8])
        change = _to_float(parts[9])
        if None in (open_, close, high, low):
            continue
        pre_close = None
        if change is not None:
            pre_close = round(close - change, 4)
        records.append(
            {
                "trade_date": date,
                "ts_code": ts_code,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "pre_close": pre_close,
                "change": change if change is not None else (close - pre_close if pre_close else None),
                "pct_chg": pct_chg,
                "vol": volume * 100.0 if volume is not None else None,  # 手 -> 股
                "amount": amount,
            }
        )
    return pd.DataFrame(records)


def _to_float(value: object) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
