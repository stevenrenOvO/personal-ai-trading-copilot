"""Fetch real A-share data into local storage.

Usage::

    python -m backend.data.ingest --universe hs300_zz500 --start 20240101 --end 20240131

Flow: BaoStockProvider -> DataPipeline -> quality check -> normalized -> CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from backend.data.config import get_settings
from backend.data.pipeline import DataPipeline
from backend.data.providers._a_share_utils import compute_limit_prices
from backend.data.providers.baostock import BaoStockProvider
from backend.data.storage import DataStorage


def _universe_codes(provider: BaoStockProvider, universe: str) -> list[str]:
    if universe == "all":
        basic = provider.stock_basic(list_status="L")
        return sorted(basic["ts_code"].astype(str).unique().tolist())
    if universe in ("hs300", "zz500", "hs300_zz500"):
        codes: set[str] = set()
        if "hs300" in universe:
            codes.update(provider.index_constituents("hs300"))
        if "zz500" in universe:
            codes.update(provider.index_constituents("zz500"))
        return sorted(codes)
    return sorted({c.strip().upper() for c in universe.split(",") if c.strip()})


def ingest(
    *,
    universe: str,
    start_date: str,
    end_date: str,
    root: Path | None = None,
) -> dict:
    provider = BaoStockProvider()
    pipeline = DataPipeline(provider, root=root or get_settings().db_path.parent)

    result: dict = {}
    result["stock_basic"] = pipeline.ingest_stock_basic(list_status="L").rows
    result["trade_cal"] = pipeline.ingest_trade_cal(
        start_date=start_date, end_date=end_date
    ).rows

    codes = _universe_codes(provider, universe)
    raw = provider.daily_raw(ts_codes=codes, start_date=start_date, end_date=end_date)

    if not raw.empty:
        daily_cols = [
            "trade_date",
            "ts_code",
            "open",
            "high",
            "low",
            "close",
            "pre_close",
            "change",
            "pct_chg",
            "vol",
            "amount",
        ]
        result["daily"] = pipeline._finish("daily", raw[daily_cols]).rows
        limits = compute_limit_prices(raw)
        if not limits.empty:
            result["stk_limit"] = pipeline._finish("stk_limit", limits).rows
    else:
        result["daily"] = 0
        result["stk_limit"] = 0

    result["codes"] = len(codes)
    result["date_range"] = [start_date, end_date]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest real A-share data")
    parser.add_argument("--universe", default="hs300_zz500")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    args = parser.parse_args()
    result = ingest(universe=args.universe, start_date=args.start, end_date=args.end)
    print(result)


if __name__ == "__main__":
    main()
