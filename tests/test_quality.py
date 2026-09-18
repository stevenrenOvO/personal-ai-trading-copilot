"""Tests for data quality checks."""

import pytest
import pandas as pd
from backend.data.quality import (
    DataQualityError,
    check_ohlc,
    check_nonnegative,
    check_date_format,
    check_ts_code_format,
    check_duplicates,
    run_all_checks,
)


def test_check_ohlc_valid(sample_daily):
    check_ohlc(sample_daily)  # should not raise


def test_check_ohlc_high_low():
    df = pd.DataFrame({"high": [5], "low": [10]})
    with pytest.raises(DataQualityError, match="High < Low"):
        check_ohlc(df)


def test_check_ohlc_high_open():
    df = pd.DataFrame({"high": [5], "open": [10]})
    with pytest.raises(DataQualityError, match="High < Open"):
        check_ohlc(df)


def test_check_ohlc_low_close():
    df = pd.DataFrame({"low": [15], "close": [10]})
    with pytest.raises(DataQualityError, match="Low > Close"):
        check_ohlc(df)


def test_check_nonnegative_valid():
    df = pd.DataFrame({"vol": [0, 10], "amount": [0, 100]})
    check_nonnegative(df, ["vol", "amount"])  # should not raise


def test_check_nonnegative_negative():
    df = pd.DataFrame({"vol": [-1, 10]})
    with pytest.raises(DataQualityError, match="vol has negative"):
        check_nonnegative(df, ["vol"])


def test_check_date_format_valid():
    df = pd.DataFrame({"trade_date": ["20240101", "20240102"]})
    check_date_format(df)


def test_check_date_format_invalid():
    df = pd.DataFrame({"trade_date": ["2024-01-01", "20240102"]})
    with pytest.raises(DataQualityError, match="invalid format"):
        check_date_format(df)


def test_check_ts_code_format_valid():
    df = pd.DataFrame({"ts_code": ["000001.SZ", "600000.SH"]})
    check_ts_code_format(df)


def test_check_ts_code_format_invalid():
    df = pd.DataFrame({"ts_code": ["000001", "600000.SH"]})
    with pytest.raises(DataQualityError, match="invalid format"):
        check_ts_code_format(df)


def test_check_duplicates():
    df = pd.DataFrame({"trade_date": ["20240101", "20240101"], "ts_code": ["000001.SZ", "000001.SZ"]})
    with pytest.raises(DataQualityError, match="Duplicate rows"):
        check_duplicates(df, ["trade_date", "ts_code"])


def test_run_all_checks_daily(sample_daily):
    run_all_checks(sample_daily, "daily")  # should not raise
