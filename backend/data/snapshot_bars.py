"""Today's daily bars from the closing full-market snapshot.

The quote service already fetches the whole market (open, high, low, price,
pre-close, volume, amount) every few seconds. Once the session is over, that
snapshot *is* the day's daily bar for every stock, so one request covers what
a per-stock k-line backfill needs ~5,000 requests (hours) to download.

Rules:

* only a snapshot the quote service marks ``final`` -- taken at or after 15:00
  on the trading day it belongs to -- is used; an intraday snapshot is never
  stored as a daily bar;
* a stock with no trade that day (suspended: zero volume) gets no bar, the
  same as the exchange daily files;
* rows with inconsistent prices (high below close, non-positive prices, ...)
  are dropped rather than repaired;
* bars already stored for the day are left untouched (insert-only).

Units match the existing store: ``vol`` in shares, ``amount`` in yuan.
"""

from __future__ import annotations

import json
from typing import Any, Optional

import pandas as pd

SOURCE_STATE_PREFIX = "snapshot_bars:"


def _meta_value(meta: Any, name: str) -> Any:
    if isinstance(meta, dict):
        return meta.get(name)
    return getattr(meta, name, None)


def is_closing_snapshot(meta: Any, trading_day: str) -> bool:
    if _meta_value(meta, "status") != "final":
        return False
    if str(_meta_value(meta, "quote_date") or "") != str(trading_day):
        return False
    clock = str(_meta_value(meta, "quote_time") or "").replace(":", "")
    return len(clock) == 6 and clock.isdigit() and clock >= "150000"


def snapshot_to_bars(frame: pd.DataFrame, trading_day: str) -> pd.DataFrame:
    """Convert snapshot rows to daily-bar rows; drop untraded / inconsistent rows."""
    columns = ["ts_code", "trade_date", "open", "high", "low", "close",
               "pre_close", "change", "pct_chg", "vol", "amount"]
    if frame is None or frame.empty:
        return pd.DataFrame(columns=columns)
    num = lambda name: pd.to_numeric(frame.get(name), errors="coerce")  # noqa: E731
    bars = pd.DataFrame({
        "ts_code": frame["ts_code"].astype(str).str.upper(),
        "trade_date": str(trading_day),
        "open": num("open"),
        "high": num("high"),
        "low": num("low"),
        "close": num("price"),
        "pre_close": num("pre_close"),
        "change": num("change"),
        "pct_chg": num("pct_chg"),
        "vol": num("volume"),
        "amount": num("amount"),
    })
    prices = bars[["open", "high", "low", "close"]]
    valid = (
        bars["ts_code"].str.fullmatch(r"\d{6}\.(SH|SZ|BJ)")
        & prices.notna().all(axis=1)
        & (prices > 0).all(axis=1)
        & (bars["vol"] > 0)
        & (bars["high"] >= bars[["open", "close", "low"]].max(axis=1))
        & (bars["low"] <= bars[["open", "close", "high"]].min(axis=1))
    )
    bars = bars[valid.fillna(False)].drop_duplicates("ts_code")
    # Fill change / pct_chg from pre_close when the vendor left them blank.
    has_pre = bars["pre_close"] > 0
    change = bars["close"] - bars["pre_close"]
    bars["change"] = bars["change"].where(bars["change"].notna(), change.where(has_pre))
    bars["pct_chg"] = bars["pct_chg"].where(
        bars["pct_chg"].notna(), (change / bars["pre_close"] * 100).where(has_pre))
    return bars[columns].reset_index(drop=True)


def persist_closing_snapshot(history: Any, frame: pd.DataFrame, meta: Any,
                             trading_day: str) -> Optional[dict[str, Any]]:
    """Store the closing snapshot as ``trading_day`` bars. ``None`` when not final."""
    if not is_closing_snapshot(meta, trading_day):
        return None
    bars = snapshot_to_bars(frame, trading_day)
    inserted = history.insert_missing(bars)
    record = {
        "trade_date": str(trading_day),
        "source": f"{_meta_value(meta, 'source') or 'unknown'}_snapshot",
        "quote_time": _meta_value(meta, "quote_time"),
        "as_of": _meta_value(meta, "as_of"),
        "snapshot_rows": int(len(frame)) if frame is not None else 0,
        "valid_bars": int(len(bars)),
        "inserted": int(inserted),
    }
    history.set_state(SOURCE_STATE_PREFIX + str(trading_day),
                      json.dumps(record, ensure_ascii=False))
    return record
