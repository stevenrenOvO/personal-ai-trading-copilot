"""Resumable full-market daily history ingestion.

Design goals (V1.0 phase 1):
  * resumable   - stocks that already have data are skipped
  * retryable   - each stock is retried on failure
  * idempotent  - SQLite primary key dedups (ts_code, trade_date)
  * honest      - missing data is left missing, never synthesised

The accepted CSV store (``data/daily.csv``) is not touched.

Index series live in the same store (the market page needs them for
指数表现). Fill or refresh them with::

    python -m backend.data.ingest_full --source backfill --start 20260601 \\
        --end 20260917 --codes 000001.SH,000300.SH,000905.SH,399001.SZ,399006.SZ
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from backend.data.config import get_settings
from backend.data.history import MarketHistory
from backend.data.providers.baostock import BaoStockProvider


def _universe_codes(limit: int | None = None, codes: str | None = None) -> list[str]:
    if codes:
        return sorted({c.strip().upper() for c in codes.split(",") if c.strip()})
    basic_path = get_settings().db_path.parent / "stock_basic.csv"
    if basic_path.exists():
        basic = pd.read_csv(basic_path, dtype=str)
        result = sorted(basic["ts_code"].dropna().astype(str).unique().tolist())
    else:
        provider = BaoStockProvider()
        basic = provider.stock_basic(list_status="L")
        result = sorted(basic["ts_code"].astype(str).unique().tolist())
    return result[:limit] if limit else result


def run(
    *,
    start_date: str,
    end_date: str,
    limit: int | None = None,
    codes: str | None = None,
    retries: int = 3,
    report_every: int = 100,
    db_path: Path | None = None,
    batch: int | None = None,
) -> dict:
    history = MarketHistory(db_path)
    targets = _universe_codes(limit=limit, codes=codes)
    already = history.codes_with_data()
    todo = [c for c in targets if c not in already]
    if batch is not None:
        todo = todo[:batch]

    print(
        f"universe={len(targets)} already={len(already & set(targets))} todo={len(todo)} "
        f"range={start_date}~{end_date}",
        file=sys.stderr,
    )
    if not todo:
        return {"done": 0, "failed": 0, "skipped": len(targets), "coverage": history.coverage()}

    provider = BaoStockProvider()
    done = 0
    failed: dict[str, str] = {}
    started = time.time()

    for index, code in enumerate(todo, start=1):
        last_error = ""
        for attempt in range(1, retries + 1):
            try:
                raw = provider.daily_raw(
                    ts_codes=[code], start_date=start_date, end_date=end_date
                )
                if raw is None or raw.empty:
                    last_error = "empty"
                    break
                history.ingest(raw)
                done += 1
                last_error = ""
                break
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                # A stalled or closed session must be re-established, otherwise
                # every following request fails the same way.
                try:
                    provider.reconnect()
                except Exception:  # noqa: BLE001
                    pass
                time.sleep(min(2.0 * attempt, 5.0))
        if last_error:
            failed[code] = last_error

        if index % report_every == 0 or index == len(todo):
            elapsed = time.time() - started
            rate = index / elapsed if elapsed > 0 else 0.0
            remaining = (len(todo) - index) / rate if rate > 0 else 0.0
            print(
                f"progress {index}/{len(todo)} ok={done} failed={len(failed)} "
                f"rate={rate:.2f}/s eta={remaining/60:.1f}min",
                file=sys.stderr,
            )

    history.set_state("last_start", start_date)
    history.set_state("last_end", end_date)
    history.set_state("last_run_failed", str(len(failed)))
    history.set_state("last_failed_codes", json.dumps(sorted(failed)))

    elapsed = time.time() - started
    return {
        "done": done,
        "failed": len(failed),
        "failed_codes": list(failed)[:20],
        "skipped": len(targets) - len(todo),
        "elapsed_sec": round(elapsed, 1),
        "rate_per_sec": round(len(todo) / elapsed, 3) if elapsed else 0.0,
        "coverage": history.coverage(),
    }


def run_backfill(
    *,
    start_date: str,
    end_date: str,
    source: str = "tencent",
    limit: int | None = None,
    codes: str | None = None,
    retries: int = 2,
    report_every: int = 300,
    batch: int | None = None,
    db_path: Path | None = None,
) -> dict:
    """Fill missing trading days fast, using Eastmoney's kline endpoint.

    A stock counts as done when it already has a bar for ``end_date``, so an
    interrupted backfill resumes from where it stopped.
    """
    if source == "eastmoney":
        from backend.data.providers.eastmoney import fetch_daily
    else:
        from backend.data.providers.tencent import fetch_daily

    history = MarketHistory(db_path)
    targets = _universe_codes(limit=limit, codes=codes)
    done = history.codes_with_date(end_date)
    todo = [c for c in targets if c not in done]
    if batch is not None:
        todo = todo[:batch]

    print(
        f"backfill universe={len(targets)} already={len(done & set(targets))} todo={len(todo)} "
        f"range={start_date}~{end_date}",
        file=sys.stderr,
    )
    if not todo:
        return {"done": 0, "failed": 0, "skipped": len(targets), "coverage": history.coverage()}

    import time as _time

    ok = 0
    failed: dict[str, str] = {}
    corporate_actions = 0
    started = _time.time()
    for index, code in enumerate(todo, start=1):
        last_error = ""
        for attempt in range(1, retries + 2):
            try:
                frame = fetch_daily(code, start_date, end_date)
                if frame.empty:
                    last_error = "empty"
                    break
                corporate_actions += int(frame.attrs.get("corporate_action_rows", 0))
                history.ingest(frame)
                ok += 1
                last_error = ""
                break
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                _time.sleep(min(1.0 * attempt, 3.0))
        if last_error:
            failed[code] = last_error
        if index % report_every == 0 or index == len(todo):
            elapsed = _time.time() - started
            rate = index / elapsed if elapsed else 0.0
            eta = (len(todo) - index) / rate if rate else 0.0
            print(
                f"progress {index}/{len(todo)} ok={ok} failed={len(failed)} "
                f"rate={rate:.2f}/s eta={eta/60:.1f}min",
                file=sys.stderr,
            )

    history.set_state("last_backfill_start", start_date)
    history.set_state("last_backfill_end", end_date)
    history.set_state("last_backfill_corporate_actions", str(corporate_actions))
    elapsed = _time.time() - started
    return {
        "done": ok,
        "failed": len(failed),
        "failed_codes": list(failed)[:20],
        "corporate_action_rows": corporate_actions,
        "skipped": len(targets) - len(todo),
        "elapsed_sec": round(elapsed, 1),
        "coverage": history.coverage(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Full-market daily history ingestion")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--codes", default=None)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--report-every", type=int, default=100)
    parser.add_argument("--batch", type=int, default=None, help="limit this run to N pending stocks")
    parser.add_argument(
        "--source",
        default="baostock",
        choices=["baostock", "backfill", "eastmoney"],
        help="baostock = per-stock full ingest; backfill = fast k-line gap fill",
    )
    parser.add_argument("--backfill-source", default="tencent", choices=["tencent", "eastmoney"])
    args = parser.parse_args()
    if args.source in ("backfill", "eastmoney"):
        result = run_backfill(
            start_date=args.start,
            end_date=args.end,
            source=args.backfill_source,
            limit=args.limit,
            codes=args.codes,
            retries=args.retries,
            report_every=args.report_every,
            batch=args.batch,
        )
    else:
        result = run(
            start_date=args.start,
            end_date=args.end,
            limit=args.limit,
            codes=args.codes,
            retries=args.retries,
            report_every=args.report_every,
            batch=args.batch,
        )
    print(json.dumps(result), file=sys.stderr)


if __name__ == "__main__":
    main()
