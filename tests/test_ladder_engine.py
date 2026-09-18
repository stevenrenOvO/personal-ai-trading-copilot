"""Tests for the Step-2 limit-up ladder / leader structure engine."""

from __future__ import annotations

import pandas as pd
import pytest

from backend.board.ladder import LimitLadderEngine, bucket_label, level_label


def _bars(rows: list[tuple[str, str, float, float, float]]) -> pd.DataFrame:
    """(date, code, pre_close, close, pct_chg) -> normalized daily frame."""
    return pd.DataFrame(
        [
            {
                "trade_date": day,
                "ts_code": code,
                "open": pre,
                "high": max(pre, close),
                "low": min(pre, close),
                "close": close,
                "pre_close": pre,
                "change": round(close - pre, 4),
                "pct_chg": pct,
                "vol": 1000.0,
                "amount": 10000.0,
            }
            for day, code, pre, close, pct in rows
        ]
    )


class _Provider:
    """Deterministic provider: 10% board, no ST, no new listings."""

    def __init__(self, bars: pd.DataFrame, basic: pd.DataFrame) -> None:
        self._bars = bars
        self._basic = basic

    def stock_basic(self, *, ts_codes=None, list_status=None):
        return self._basic

    def daily(self, *, ts_codes=None, start_date=None, end_date=None):
        frame = self._bars.copy()
        frame["trade_date"] = frame["trade_date"].astype(str)
        if start_date:
            frame = frame[frame["trade_date"] >= start_date]
        if end_date:
            frame = frame[frame["trade_date"] <= end_date]
        if ts_codes:
            frame = frame[frame["ts_code"].isin(ts_codes)]
        return frame.reset_index(drop=True)

    def daily_basic(self, *, ts_codes=None, trade_date=None):
        return pd.DataFrame()

    def stk_limit(self, *, ts_codes=None, trade_date=None):
        frame = self._bars[self._bars["trade_date"].astype(str) == str(trade_date)]
        rows = [
            {
                "trade_date": str(row["trade_date"]),
                "ts_code": str(row["ts_code"]),
                "up_limit": round(float(row["pre_close"]) * 1.10, 2),
                "down_limit": round(float(row["pre_close"]) * 0.90, 2),
            }
            for _, row in frame.iterrows()
        ]
        return pd.DataFrame(rows)

    def trade_cal(self, *, exchange=None, start_date=None, end_date=None, is_open=None):
        days = sorted(self._bars["trade_date"].astype(str).unique())
        return pd.DataFrame(
            {"exchange": ["SSE"] * len(days), "cal_date": days, "is_open": [1] * len(days)}
        )


_DAYS = ["20240101", "20240102", "20240103"]


def _scenario() -> _Provider:
    """A 3-day market with a 3-board, a 2-board, a first board, a gap and a break.

    Letters are codes: A runs 1-2-3 (board), B runs 2 then -5% (broken run),
    C is a plain first board today, D limited up on day1 only (so no 2-board on
    day2 => a gap at 2 is NOT expected since A is at 2; see assertions).
    """
    rows: list[tuple[str, str, float, float, float]] = []

    def add(day, code, pre, close, pct):
        rows.append((day, code, pre, close, pct))

    # A (000001.SZ): limit up every day -> 3 board on 20240103
    add("20240101", "000001.SZ", 10.0, 11.0, 10.0)
    add("20240102", "000001.SZ", 11.0, 12.1, 10.0)
    add("20240103", "000001.SZ", 12.1, 13.31, 10.0)
    # B (600000.SH): limit up day1+day2 (2 board), then falls 5% today
    add("20240101", "600000.SH", 20.0, 22.0, 10.0)
    add("20240102", "600000.SH", 22.0, 24.2, 10.0)
    add("20240103", "600000.SH", 24.2, 22.99, -5.0)
    # C (000002.SZ): flat, then limit up today -> 首板
    add("20240101", "000002.SZ", 5.0, 5.0, 0.0)
    add("20240102", "000002.SZ", 5.0, 5.0, 0.0)
    add("20240103", "000002.SZ", 5.0, 5.5, 10.0)
    # D (600004.SH): limit up day1 only, then flat -> yesterday premium check
    add("20240101", "600004.SH", 8.0, 8.8, 10.0)
    add("20240102", "600004.SH", 8.8, 8.6, -2.27)
    add("20240103", "600004.SH", 8.6, 8.5, -1.16)
    basic = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "600000.SH", "000002.SZ", "600004.SH"],
            "symbol": ["000001", "600000", "000002", "600004"],
            "name": ["甲公司", "乙公司", "丙公司", "丁公司"],
            "industry": ["银行", "银行", "软件", "银行"],
            "list_date": ["19910101"] * 4,
            "list_status": ["L"] * 4,
            "is_st": ["N"] * 4,
        }
    )
    return _Provider(_bars(rows), basic)


def test_level_and_bucket_labels():
    assert [level_label(h) for h in (1, 2, 3, 4, 6)] == ["首板", "二板", "三板", "4板", "6板"]
    assert [bucket_label(h) for h in (1, 2, 3, 4, 6)] == ["首板", "二板", "三板", "四板及以上", "四板及以上"]


def test_ladder_heights_and_structure():
    snapshot = LimitLadderEngine(_scenario()).snapshot("20240103")
    assert snapshot.available
    # A is a 3-board, C is a first board.
    assert snapshot.max_height == 3
    assert snapshot.buckets == {"首板": 1, "二板": 0, "三板": 1, "四板及以上": 0}
    # No stock is on a 2-board today although one is on 3 -> 断层 at 2.
    assert snapshot.gaps == (2,)
    assert snapshot.is_complete is False
    assert [s.name for s in snapshot.highest] == ["甲公司"]
    assert snapshot.excluded_new_listing == 0


def test_yesterday_premium_and_promotion_rate():
    snapshot = LimitLadderEngine(_scenario()).snapshot("20240103")
    y = snapshot.yesterday
    assert y is not None
    assert y.date == "20240102"
    # Yesterday's limit-ups: A (2 board) and B (2 board) only.
    assert y.count == 2
    assert y.evaluated == 2
    assert y.promoted_count == 1          # only A limited up again
    assert y.promotion_rate == pytest.approx(0.5)
    assert y.up_count == 1 and y.down_count == 1
    assert y.avg_pct == pytest.approx((10.0 + (-5.0)) / 2)
    assert y.median_pct == pytest.approx(2.5)
    by_height = {p.height: p for p in y.promotions}
    assert by_height[2].count == 2 and by_height[2].promoted == 1


def test_industry_ladder_roles_are_structural():
    snapshot = LimitLadderEngine(_scenario()).snapshot("20240103")
    banks = next(i for i in snapshot.industries if i.name == "银行")
    # 银行 holds the 3-board (甲公司) and the 2-board-turned-down name is not
    # limit-up today, so the bank industry only has 甲公司 left.
    assert banks.max_height == 3
    assert banks.leader_candidates == ("000001.SZ",)
    assert snapshot.dominant_industry == "银行"


def test_unavailable_day_is_reported_not_invented():
    snapshot = LimitLadderEngine(_scenario()).snapshot("20240104")
    assert snapshot.available is False
    assert snapshot.limit_up_count == 0
    assert snapshot.notes


def test_stale_st_name_is_corrected_by_data():
    """A name that says ST but trades on a 10% band must not use a 5% cap."""
    provider = _scenario()
    provider._basic.loc[provider._basic["ts_code"] == "000002.SZ", "name"] = "ST丙公司"
    provider._basic.loc[provider._basic["ts_code"] == "000002.SZ", "is_st"] = "Y"
    # C moved -0%/0% then +10%: a 5% band would have made the +10% un-tradable.
    provider._bars.loc[
        (provider._bars["ts_code"] == "000002.SZ") & (provider._bars["trade_date"] == "20240102"),
        "close",
    ] = 5.4
    snapshot = LimitLadderEngine(provider).snapshot("20240103")
    assert snapshot.buckets["首板"] >= 1
    assert snapshot.excluded_corporate_action == 0
