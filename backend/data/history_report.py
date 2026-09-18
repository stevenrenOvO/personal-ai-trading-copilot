"""Data acceptance report for the full-market history store."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from backend.data.history import MarketHistory


def build_report(db_path: Path | None = None) -> dict:
    history = MarketHistory(db_path)
    frame = history.load()
    if frame.empty:
        return {"rows": 0, "note": "empty history"}

    codes = int(frame["ts_code"].nunique())
    dates = int(frame["trade_date"].nunique())
    rows = int(len(frame))
    per_stock = frame.groupby("ts_code").size()
    expected = codes * dates

    violations = {
        "ohlc_high_lt_low": int((frame["high"] < frame["low"]).sum()),
        "ohlc_high_lt_open_close": int(
            (frame["high"] < frame[["open", "close"]].max(axis=1)).sum()
        ),
        "ohlc_low_gt_open_close": int(
            (frame["low"] > frame[["open", "close"]].min(axis=1)).sum()
        ),
        "non_positive_close": int((frame["close"] <= 0).sum()),
        "negative_volume": int((frame["vol"] < 0).sum()),
        "negative_amount": int((frame["amount"] < 0).sum()),
        "abs_pct_chg_over_50": int((frame["pct_chg"].abs() > 50).sum()),
    }
    duplicates = int(frame.duplicated(subset=["ts_code", "trade_date"]).sum())

    return {
        "codes": codes,
        "dates": dates,
        "rows": rows,
        "start": str(frame["trade_date"].min()),
        "end": str(frame["trade_date"].max()),
        "rows_per_stock_min": int(per_stock.min()),
        "rows_per_stock_median": float(per_stock.median()),
        "rows_per_stock_max": int(per_stock.max()),
        "stocks_below_expected": int((per_stock < dates).sum()),
        "expected_rows_if_full": expected,
        "missing_rate": round(1 - rows / expected, 4) if expected else None,
        "duplicates": duplicates,
        "violations": violations,
        "db_mb": history.coverage()["db_mb"],
    }


def main() -> None:
    print(json.dumps(build_report(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
