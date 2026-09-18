"""Data quality checks for A-share market data."""

from __future__ import annotations

import pandas as pd

from backend.data.schemas import is_valid_trade_date, is_valid_ts_code


class DataQualityError(ValueError):
    """Raised when a data quality check fails."""


def check_ohlc(df: pd.DataFrame) -> None:
    """Ensure high >= low and high >= open/close, low <= open/close."""
    if "high" in df.columns and "low" in df.columns:
        invalid = df[df["high"] < df["low"]]
        if not invalid.empty:
            raise DataQualityError(f"High < Low in {len(invalid)} rows")
    if "high" in df.columns and "open" in df.columns:
        invalid = df[df["high"] < df["open"]]
        if not invalid.empty:
            raise DataQualityError(f"High < Open in {len(invalid)} rows")
    if "high" in df.columns and "close" in df.columns:
        invalid = df[df["high"] < df["close"]]
        if not invalid.empty:
            raise DataQualityError(f"High < Close in {len(invalid)} rows")
    if "low" in df.columns and "open" in df.columns:
        invalid = df[df["low"] > df["open"]]
        if not invalid.empty:
            raise DataQualityError(f"Low > Open in {len(invalid)} rows")
    if "low" in df.columns and "close" in df.columns:
        invalid = df[df["low"] > df["close"]]
        if not invalid.empty:
            raise DataQualityError(f"Low > Close in {len(invalid)} rows")


def check_nonnegative(df: pd.DataFrame, columns: list[str]) -> None:
    for col in columns:
        if col in df.columns:
            invalid = df[df[col] < 0]
            if not invalid.empty:
                raise DataQualityError(f"{col} has negative values in {len(invalid)} rows")


def check_date_format(df: pd.DataFrame, date_col: str = "trade_date") -> None:
    if date_col not in df.columns:
        return
    invalid = df[~df[date_col].astype(str).str.match(r"^\d{8}$", na=False)]
    if not invalid.empty:
        raise DataQualityError(f"{date_col} has invalid format in {len(invalid)} rows")


def check_ts_code_format(df: pd.DataFrame, code_col: str = "ts_code") -> None:
    if code_col not in df.columns:
        return
    invalid = df[~df[code_col].astype(str).str.match(r"^\d{6}\.(SH|SZ|BJ)$", na=False)]
    if not invalid.empty:
        raise DataQualityError(f"{code_col} has invalid format in {len(invalid)} rows")


def check_duplicates(df: pd.DataFrame, key_cols: list[str]) -> None:
    duplicates = df.duplicated(subset=key_cols)
    if duplicates.any():
        raise DataQualityError(f"Duplicate rows found on {key_cols}")


def run_all_checks(
    df: pd.DataFrame,
    table: str,
    date_col: str = "trade_date",
    code_col: str = "ts_code",
) -> None:
    """Run a battery of quality checks appropriate for the table type."""
    if table in ("daily", "daily_basic", "stk_limit", "adj_factor", "suspend_d"):
        check_date_format(df, date_col)
        check_ts_code_format(df, code_col)
        check_nonnegative(
            df,
            [
                "vol",
                "amount",
                "turnover_rate",
                "total_share",
                "float_share",
                "adj_factor",
            ],
        )
        if table == "daily":
            check_ohlc(df)
        if table == "adj_factor":
            check_duplicates(df, ["ts_code", "trade_date"])
    elif table == "trade_cal":
        check_date_format(df, "cal_date")
        if "is_open" in df.columns:
            invalid = df[~df["is_open"].astype(int).isin([0, 1])]
            if not invalid.empty:
                raise DataQualityError("is_open must be 0 or 1")
    if table == "stock_basic":
        check_ts_code_format(df, code_col)
    # Check duplicates based on common keys
    if table == "daily":
        check_duplicates(df, ["trade_date", "ts_code"])
    elif table == "daily_basic":
        check_duplicates(df, ["trade_date", "ts_code"])
    elif table == "stk_limit":
        check_duplicates(df, ["trade_date", "ts_code"])
    elif table == "suspend_d":
        check_duplicates(df, ["trade_date", "ts_code"])
    elif table == "stock_basic":
        check_duplicates(df, ["ts_code"])
