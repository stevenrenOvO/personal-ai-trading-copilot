"""Tests for the Step-4 validation tooling (event study / control / robustness)."""
from __future__ import annotations

import pandas as pd

from backend.opportunity.validation import (
    CAPABILITIES,
    GroupStats,
    OpportunityValidator,
    Trade,
    _stats,
)

DAYS = ["20240101", "20240102", "20240103", "20240104", "20240105"]


def _row(day, code, o, h, l, c, pre, pct):
    return {
        "trade_date": day,
        "ts_code": code,
        "open": o,
        "high": h,
        "low": l,
        "close": c,
        "pre_close": pre,
        "pct_chg": pct,
        "change": round(c - pre, 4),
        "vol": 1000.0,
        "amount": None,
    }


class _Provider:
    """000001.SZ is fillable at the T+1 open; 000002.SZ opens limit-up."""

    def __init__(self) -> None:
        self._bars = pd.DataFrame(
            [
                _row("20240102", "000001.SZ", 10.0, 11.0, 10.0, 11.0, 10.0, 10.0),
                _row("20240103", "000001.SZ", 10.0, 11.0, 9.9, 10.8, 11.0, -1.8),
                _row("20240104", "000001.SZ", 11.0, 11.0, 10.9, 11.0, 10.8, 1.9),
                _row("20240102", "000002.SZ", 10.0, 11.0, 10.0, 11.0, 10.0, 10.0),
                _row("20240103", "000002.SZ", 12.1, 12.1, 12.1, 12.1, 11.0, 10.0),
                _row("20240104", "000002.SZ", 12.1, 12.1, 11.5, 11.8, 12.1, -2.5),
                _row("20240102", "600000.SH", 5.0, 5.1, 4.9, 5.0, 5.0, 0.0),
                _row("20240103", "600000.SH", 5.0, 5.1, 4.9, 5.0, 5.0, 0.0),
                _row("20240104", "600000.SH", 5.1, 5.2, 5.0, 5.1, 5.0, 2.0),
            ]
        )
        self._basic = pd.DataFrame(
            [
                ("000001.SZ", "000001", "甲", "银行", "19910101"),
                ("000002.SZ", "000002", "乙", "银行", "19910101"),
                ("600000.SH", "600000", "丙", "银行", "19910101"),
            ],
            columns=["ts_code", "symbol", "name", "industry", "list_date"],
        )

    def stock_basic(self, *, ts_codes=None, list_status=None):
        return self._basic.copy()

    def daily(self, *, ts_codes=None, start_date=None, end_date=None):
        frame = self._bars.copy()
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
        frame = self._bars[self._bars["trade_date"] == trade_date]
        return pd.DataFrame(
            [
                {
                    "trade_date": str(r["trade_date"]),
                    "ts_code": str(r["ts_code"]),
                    "up_limit": round(float(r["pre_close"]) * 1.10, 2),
                    "down_limit": round(float(r["pre_close"]) * 0.90, 2),
                }
                for _, r in frame.iterrows()
            ]
        )

    def trade_cal(self, *, exchange=None, start_date=None, end_date=None, is_open=None):
        return pd.DataFrame(
            {"exchange": ["SSE"] * len(DAYS), "cal_date": DAYS, "is_open": [1] * len(DAYS)}
        )


class _Plan:
    def __init__(self, code, type_="首板", height=1):
        self.ts_code = code
        self.name = code[:6]
        self.opportunity_type = type_
        self.tier = "A"
        self.sector = "银行"
        self.board_height = height


def _validator() -> OpportunityValidator:
    return OpportunityValidator(_Provider())


def test_stats_compute_win_rate_payoff_and_fill_rate():
    trades = [
        Trade("d", "a", "", "首板", "A", "", 1, entry_price=10.0, exit_price=11.0, ret=0.1, filled=True),
        Trade("d", "b", "", "首板", "A", "", 1, entry_price=10.0, exit_price=9.0, ret=-0.1, filled=True),
        Trade("d", "c", "", "首板", "A", "", 1, not_filled_reason="涨停价开盘"),
    ]
    stats = _stats("x", trades)
    assert (stats.signals, stats.filled, stats.not_filled) == (3, 2, 1)
    assert stats.win_rate == 0.5 and stats.avg_ret == 0.0 and stats.payoff == 1.0
    assert stats.to_dict()["fill_rate"] == round(2 / 3, 4)


def test_entry_is_t_plus_one_open_and_exit_is_t_plus_two_open():
    bars = _Provider().daily(start_date="20240102", end_date="20240105")
    trade = _validator()._evaluate("20240102", _Plan("000001.SZ"), bars)
    assert trade.filled
    assert trade.entry_date == "20240103" and trade.entry_price == 10.0
    assert trade.exit_date == "20240104" and trade.exit_price == 11.0
    assert abs(trade.ret - 0.1) < 1e-9
    assert trade.mfe is not None and trade.mae is not None


def test_limit_up_open_is_not_treated_as_filled():
    bars = _Provider().daily(start_date="20240102", end_date="20240105")
    trade = _validator()._evaluate("20240102", _Plan("000002.SZ", "连板接力", 2), bars)
    assert trade.filled is False and trade.ret is None
    assert "涨停" in trade.not_filled_reason


def test_missing_t_plus_two_data_is_reported_not_faked():
    bars = _Provider().daily(start_date="20240104", end_date="20240105")
    trade = _validator()._evaluate("20240104", _Plan("000001.SZ"), bars)
    assert trade.filled is False and "T+2" in trade.not_filled_reason


def test_control_group_excludes_signal_names():
    bars = _Provider().daily(start_date="20240101", end_date="20240105")
    stats = _validator()._control_group(["20240102", "20240103"], {"000001.SZ", "000002.SZ"}, bars)
    assert isinstance(stats, GroupStats) and stats.signals > 0 and "对照" in stats.name


def test_robustness_drops_best_five_percent_and_splits_halves():
    trades = [
        Trade("20240101", f"c{i}", "", "首板", "A", "", 1,
              entry_price=10.0, exit_price=10.0 + i, ret=i / 100.0, filled=True)
        for i in range(1, 21)
    ]
    report = _validator()._robustness(trades)
    assert report["dropped_best"] == 1
    assert report["avg_ret_after_dropping_best_5pct"] is not None
    assert report["first_half"] and report["second_half"]


def test_capability_table_states_what_cannot_be_backtested():
    verdicts = {item: verdict for item, verdict, _ in CAPABILITIES}
    assert verdicts["半路买点"] == "必须分钟数据"
    assert verdicts["打板能否成交"] == "当前无法回测"
    assert verdicts["首板 / 连板接力 的次日表现"] == "日线可验证"


def test_validation_reports_caveats_about_bias_and_execution():
    report = _validator().run("20240101", "20240105", tiers=("A", "B"))
    text = " ".join(report.caveats)
    assert "T+1" in text and "幸存者偏差" in text and "未来函数" in text
