"""Local data ingestion pipeline.

Flow: provider -> raw fetch -> quality check -> normalized schema ->
local storage. Real (Tushare) and fixture data stay isolated because the
pipeline stores under the same configurable data root but is always driven by
the provider the caller chooses; the fixture provider never writes into the
production database and vice versa.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from backend.data.quality import run_all_checks
from backend.data.schemas import SCHEMAS, validate_dataframe
from backend.data.storage import DataStorage


@dataclass
class IngestResult:
    table: str
    rows: int
    quality_ok: bool
    warnings: list[str] = field(default_factory=list)


class DataPipeline:
    """Fetch, validate and persist normalized tables."""

    def __init__(self, provider, root: Path | None = None) -> None:
        self.provider = provider
        self.storage = DataStorage(root)

    def ingest_stock_basic(self, list_status: str | None = None) -> IngestResult:
        raw = self.provider.stock_basic(list_status=list_status)
        return self._finish("stock_basic", raw)

    def ingest_daily(
        self,
        ts_codes: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> IngestResult:
        raw = self.provider.daily(
            ts_codes=ts_codes, start_date=start_date, end_date=end_date
        )
        return self._finish("daily", raw)

    def ingest_daily_basic(
        self,
        ts_codes: list[str] | None = None,
        trade_date: str | None = None,
    ) -> IngestResult:
        raw = self.provider.daily_basic(ts_codes=ts_codes, trade_date=trade_date)
        return self._finish("daily_basic", raw)

    def ingest_stk_limit(
        self,
        ts_codes: list[str] | None = None,
        trade_date: str | None = None,
    ) -> IngestResult:
        raw = self.provider.stk_limit(ts_codes=ts_codes, trade_date=trade_date)
        return self._finish("stk_limit", raw)

    def ingest_trade_cal(
        self,
        exchange: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> IngestResult:
        raw = self.provider.trade_cal(
            exchange=exchange, start_date=start_date, end_date=end_date
        )
        return self._finish("trade_cal", raw)

    def ingest_adj_factor(
        self,
        ts_codes: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> IngestResult:
        raw = self.provider.adj_factor(
            ts_codes=ts_codes, start_date=start_date, end_date=end_date
        )
        return self._finish("adj_factor", raw)

    def _finish(self, table: str, raw: pd.DataFrame) -> IngestResult:
        if raw is None:
            raw = pd.DataFrame()

        warnings: list[str] = []
        if raw.empty:
            warnings.append("empty source data")
            return IngestResult(table=table, rows=0, quality_ok=True, warnings=warnings)

        if table not in SCHEMAS:
            raise ValueError(f"unknown table: {table}")

        # Quality first, then canonical normalization. Both can reject bad rows.
        run_all_checks(raw, table)
        normalized = validate_dataframe(raw, table)

        self.storage.save(normalized, table)
        return IngestResult(
            table=table,
            rows=len(normalized),
            quality_ok=True,
            warnings=warnings,
        )

    def load(self, table: str) -> pd.DataFrame:
        return self.storage.load(table)
