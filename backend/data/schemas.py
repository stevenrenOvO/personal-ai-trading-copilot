"""Unified internal data schemas for the A-share data foundation layer.

Every provider returns DataFrames that satisfy these contracts so upper-layer
strategy and signal code never depends on a specific vendor's column names.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd


class SchemaValidationError(ValueError):
    """Raised when a DataFrame does not conform to an internal schema."""


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    dtype: str
    required: bool = True


@dataclass(frozen=True)
class TableSchema:
    name: str
    columns: tuple[ColumnSpec, ...] = field(default_factory=tuple)
    allow_extra: bool = True

    @property
    def required_columns(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.columns if c.required)

    @property
    def all_columns(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.columns)


_TEXT = "string"
_DATE = "date"
_FLOAT = "float64"
_INT = "int64"


STOCK_BASIC = TableSchema(
    name="stock_basic",
    columns=(
        ColumnSpec("ts_code", _TEXT),
        ColumnSpec("symbol", _TEXT),
        ColumnSpec("name", _TEXT),
        ColumnSpec("area", _TEXT, required=False),
        ColumnSpec("industry", _TEXT, required=False),
        ColumnSpec("market", _TEXT, required=False),
        ColumnSpec("list_date", _DATE, required=False),
        ColumnSpec("list_status", _TEXT, required=False),
        ColumnSpec("delist_date", _DATE, required=False),
        ColumnSpec("is_st", _TEXT, required=False),
    ),
)


DAILY = TableSchema(
    name="daily",
    columns=(
        ColumnSpec("trade_date", _DATE),
        ColumnSpec("ts_code", _TEXT),
        ColumnSpec("open", _FLOAT),
        ColumnSpec("high", _FLOAT),
        ColumnSpec("low", _FLOAT),
        ColumnSpec("close", _FLOAT),
        ColumnSpec("pre_close", _FLOAT),
        ColumnSpec("change", _FLOAT),
        ColumnSpec("pct_chg", _FLOAT),
        ColumnSpec("vol", _FLOAT),
        ColumnSpec("amount", _FLOAT),
    ),
)


DAILY_BASIC = TableSchema(
    name="daily_basic",
    columns=(
        ColumnSpec("trade_date", _DATE),
        ColumnSpec("ts_code", _TEXT),
        ColumnSpec("turnover_rate", _FLOAT, required=False),
        ColumnSpec("turnover_rate_f", _FLOAT, required=False),
        ColumnSpec("volume_ratio", _FLOAT, required=False),
        ColumnSpec("pe", _FLOAT, required=False),
        ColumnSpec("pe_ttm", _FLOAT, required=False),
        ColumnSpec("pb", _FLOAT, required=False),
        ColumnSpec("total_share", _FLOAT, required=False),
        ColumnSpec("float_share", _FLOAT, required=False),
        ColumnSpec("total_mv", _FLOAT, required=False),
        ColumnSpec("circ_mv", _FLOAT, required=False),
    ),
)


STK_LIMIT = TableSchema(
    name="stk_limit",
    columns=(
        ColumnSpec("trade_date", _DATE),
        ColumnSpec("ts_code", _TEXT),
        ColumnSpec("up_limit", _FLOAT),
        ColumnSpec("down_limit", _FLOAT),
    ),
)


TRADE_CAL = TableSchema(
    name="trade_cal",
    columns=(
        ColumnSpec("exchange", _TEXT),
        ColumnSpec("cal_date", _DATE),
        ColumnSpec("is_open", _INT),
        ColumnSpec("pretrade_date", _DATE, required=False),
    ),
)


ADJ_FACTOR = TableSchema(
    name="adj_factor",
    columns=(
        ColumnSpec("ts_code", _TEXT),
        ColumnSpec("trade_date", _DATE),
        ColumnSpec("adj_factor", _FLOAT),
    ),
)


SUSPEND_D = TableSchema(
    name="suspend_d",
    columns=(
        ColumnSpec("ts_code", _TEXT),
        ColumnSpec("trade_date", _DATE),
        ColumnSpec("suspend_timing", _TEXT, required=False),
        ColumnSpec("suspend_type", _TEXT, required=False),
    ),
)


SCHEMAS: dict[str, TableSchema] = {
    schema.name: schema
    for schema in (
        STOCK_BASIC,
        DAILY,
        DAILY_BASIC,
        STK_LIMIT,
        TRADE_CAL,
        ADJ_FACTOR,
        SUSPEND_D,
    )
}


_TS_CODE_RE = re.compile(r"^\d{6}\.(SH|SZ|BJ|sh|sz|bj)$")
_TRADE_DATE_RE = re.compile(r"^\d{8}$")


def validate_dataframe(df: pd.DataFrame, schema_name: str) -> pd.DataFrame:
    """Validate and normalize a DataFrame to the canonical schema.

    Required columns must be present. Optional columns that are missing are
    filled with pd.NA. Extra columns are dropped when ``allow_extra`` is False.
    """

    if not isinstance(df, pd.DataFrame):
        raise SchemaValidationError(
            f"{schema_name}: expected DataFrame, got {type(df).__name__}"
        )

    schema = SCHEMAS.get(schema_name)
    if schema is None:
        raise SchemaValidationError(f"unknown schema: {schema_name}")

    missing = [c for c in schema.required_columns if c not in df.columns]
    if missing:
        raise SchemaValidationError(
            f"{schema_name}: missing required columns: {missing}"
        )

    if not schema.allow_extra:
        extra = [c for c in df.columns if c not in schema.all_columns]
        if extra:
            raise SchemaValidationError(
                f"{schema_name}: unexpected columns: {extra}"
            )

    result = pd.DataFrame(index=df.index)
    for spec in schema.columns:
        if spec.name in df.columns:
            series = df[spec.name]
        elif spec.required:
            raise SchemaValidationError(
                f"{schema_name}: required column disappeared: {spec.name}"
            )
        else:
            series = pd.Series(pd.NA, index=df.index, dtype="object")

        if spec.dtype in (_FLOAT, _INT):
            result[spec.name] = pd.to_numeric(series, errors="coerce")
        else:
            result[spec.name] = series.astype("string")

    return result.loc[:, list(schema.all_columns)].reset_index(drop=True)


def is_valid_ts_code(value: object) -> bool:
    return bool(_TS_CODE_RE.match(str(value).strip().upper()))


def is_valid_trade_date(value: object) -> bool:
    return bool(_TRADE_DATE_RE.match(str(value).strip()))


def normalize_ts_codes(codes: Iterable[object]) -> list[str]:
    """Validate, upper-case, and de-duplicate ts_codes."""

    seen: set[str] = set()
    result: list[str] = []
    for code in codes:
        normalized = str(code).strip().upper()
        if not _TS_CODE_RE.match(normalized):
            raise SchemaValidationError(f"invalid ts_code: {code!r}")
        if normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result
