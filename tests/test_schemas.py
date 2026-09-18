"""Tests for schema validation and normalization."""

import pytest
import pandas as pd
from backend.data.schemas import (
    SchemaValidationError,
    validate_dataframe,
    is_valid_ts_code,
    is_valid_trade_date,
    normalize_ts_codes,
    SCHEMAS,
)


def test_validate_stock_basic_valid(sample_stock_basic):
    result = validate_dataframe(sample_stock_basic, "stock_basic")
    assert len(result) == 2
    assert list(result.columns) == list(SCHEMAS["stock_basic"].all_columns)


def test_validate_stock_basic_missing_required():
    df = pd.DataFrame({"ts_code": ["000001.SZ"]})
    with pytest.raises(SchemaValidationError, match="missing required columns"):
        validate_dataframe(df, "stock_basic")


def test_validate_daily_valid(sample_daily):
    result = validate_dataframe(sample_daily, "daily")
    assert len(result) == 2
    assert list(result.columns) == list(SCHEMAS["daily"].all_columns)


def test_validate_daily_basic_valid(sample_daily_basic):
    result = validate_dataframe(sample_daily_basic, "daily_basic")
    assert len(result) == 2


def test_validate_stk_limit_valid(sample_stk_limit):
    result = validate_dataframe(sample_stk_limit, "stk_limit")
    assert len(result) == 2


def test_is_valid_ts_code():
    assert is_valid_ts_code("000001.SZ")
    assert is_valid_ts_code("600000.SH")
    assert is_valid_ts_code("000001.sz")  # case insensitive
    assert not is_valid_ts_code("000001")
    assert not is_valid_ts_code("000001.SS")


def test_is_valid_trade_date():
    assert is_valid_trade_date("20240101")
    assert not is_valid_trade_date("2024-01-01")


def test_normalize_ts_codes():
    codes = ["000001.sz", "600000.sh", "000001.SZ"]
    result = normalize_ts_codes(codes)
    assert result == ["000001.SZ", "600000.SH"]
