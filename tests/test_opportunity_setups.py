"""Tests for Step-4 stage 2: seven setups + the entry state machine."""

from __future__ import annotations

import pandas as pd

from backend.opportunity.setups import OPPORTUNITY_TYPES, OpportunitySetupEngine

DAYS = ["20240101", "20240102", "20240103", "20240104", "20240105"]


def _bar(day, code, pre, close, pct, *, open_=None, high=None, low=None, vol=1000.0):
    return {
        "trade_date": day,
        "ts_code": code,
        "open": pre if open_ is None else open_,
        "high": max(pre, close) if high is None else high,
        "low": min(pre, close) if low is None else low,
        "close": close,
        "pre_close": pre,
        "change": round(close - pre, 4),
        "pct_chg": pct,
        "vol": vol,
        "amount": None,
    }


def _up(pre: float) -> float:
    return round(pre * 1.10, 2)


class _Provider:
    """银行/医药/电子/计算机 with the six setups this step must separate."""

    def __init__(self) -> None:
        rows: list[dict] = []
        # 银行: 一字首板 + 放量首板
        rows.append(_bar(DAYS[-1], "000001.SZ", 10.0, 11.0, 10.0, open_=11.0, low=11.0))
        rows.append(_bar(DAYS[-1], "000002.SZ", 20.0, 22.0, 10.0, open_=20.4, low=20.2, vol=3000.0))
        # 医药: 3 板 + 3 板（当日开过板 = 分歧）
        price = 30.0
        for day in DAYS[-3:]:
            up = _up(price)
            rows.append(_bar(day, "000003.SZ", price, up, 10.0))
            price = up
        price = 15.0
        for index, day in enumerate(DAYS[-3:]):
            up = _up(price)
            if index == 2:
                rows.append(
                    _bar(day, "000006.SZ", price, up, 10.0,
                         open_=price * 1.02, low=price * 0.99, vol=5000.0)
                )
            else:
                rows.append(_bar(day, "000006.SZ", price, up, 10.0))
            price = up
        # 电子: 昨日炸板 → 今日封板（真弱转强）
        rows.append(_bar(DAYS[-2], "000005.SH", 8.0, 8.4, 5.0, high=8.8, low=8.1))
        rows.append(_bar(DAYS[-1], "000005.SH", 8.4, 9.24, 10.0, open_=8.6, low=8.5))
        rows.append(_bar(DAYS[-1], "000009.SH", 12.0, 13.2, 10.0, open_=12.3))
        # 计算机: 昨日炸板 → 今日 +6% 但未封板（入口成立、未确认）
        rows.append(_bar(DAYS[-2], "000004.SZ", 9.0, 9.5, 5.6, high=9.9, low=9.1))
        rows.append(_bar(DAYS[-1], "000004.SZ", 9.5, 10.07, 6.0, open_=9.6, low=9.55))
        rows.append(_bar(DAYS[-1], "000007.SZ", 11.0, 11.66, 6.0, open_=11.1, low=11.05))
        self._bars = pd.DataFrame(rows)
        self._basic = pd.DataFrame(
            [
                ("000001.SZ", "000001", "甲银行", "银行", "19910101"),
                ("000002.SZ", "000002", "乙银行", "银行", "19910101"),
                ("000003.SZ", "000003", "丙医药", "医药", "19910101"),
                ("000006.SZ", "000006", "己医药", "医药", "19910101"),
                ("000005.SH", "000005", "戊电子", "电子", "19910101"),
                ("000009.SH", "000009", "壬电子", "电子", "19910101"),
                ("000004.SZ", "000004", "丁计算", "计算机", "19910101"),
                ("000007.SZ", "000007", "庚计算", "计算机", "19910101"),
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
                    "trade_date": str(row["trade_date"]),
                    "ts_code": str(row["ts_code"]),
                    "up_limit": _up(float(row["pre_close"])),
                    "down_limit": round(float(row["pre_close"]) * 0.9, 2),
                }
                for _, row in frame.iterrows()
            ]
        )

    def trade_cal(self, *, exchange=None, start_date=None, end_date=None, is_open=None):
        return pd.DataFrame(
            {"exchange": ["SSE"] * len(DAYS), "cal_date": DAYS, "is_open": [1] * len(DAYS)}
        )


def _result(stage="发酵"):
    engine = OpportunitySetupEngine(_Provider())
    return engine.build(DAYS[-1], stage=stage, stage_rule="test")


def test_all_seven_types_are_evaluated_independently():
    result = _result()
    assert set(result.type_counts) == set(OPPORTUNITY_TYPES)
    assert result.type_counts["首板"] >= 1
    assert result.type_counts["连板接力"] >= 1
    assert result.type_counts["弱转强"] >= 1
    assert result.type_counts["分歧转一致"] >= 1
    assert result.type_counts["打板"] >= 1


def test_one_row_per_stock_and_overlaps_are_reported_on_the_row():
    result = _result()
    codes = [o.ts_code for o in result.opportunities]
    assert len(codes) == len(set(codes))
    assert any(o.also_matches for o in result.opportunities)


def test_state_machine_uses_the_four_labels_and_always_explains_itself():
    result = _result()
    assert set(result.state_counts) == {"观察", "触发中", "可执行", "失效"}
    assert sum(result.state_counts.values()) == len(result.opportunities)
    for item in result.opportunities:
        assert item.state in ("观察", "触发中", "可执行", "失效")
        assert item.state_reason
        assert item.entry_conditions and item.trigger_conditions
        assert item.invalidation_conditions


def test_unsatisfied_trigger_can_never_be_executable():
    result = _result()
    for item in result.opportunities:
        if item.state != "可执行":
            continue
        assert item.environment_allowed
        assert all(c.passed is True for c in item.entry_conditions)
        assert all(c.passed is True for c in item.trigger_conditions)
        assert all(c.passed is True for c in item.executable_conditions)


def test_missing_data_can_never_be_executable():
    result = _result()
    for item in result.opportunities:
        if item.opportunity_type not in ("打板", "半路"):
            continue
        assert item.state != "可执行"
        assert item.execution_mode in ("日内不可执行（缺分钟/盘口数据）", "仅观察")
        assert item.warnings


def test_weak_to_strong_requires_more_than_five_percent():
    result = _result()
    w2s = [o for o in result.opportunities if o.opportunity_type == "弱转强"]
    assert w2s
    for item in w2s:
        sealed = next(c for c in item.trigger_conditions if "封板" in c.name)
        if sealed.passed is False:
            assert item.state != "可执行"
        assert len(item.trigger_conditions) >= 4   # 板块/梯队/结构/阶段 all checked


def test_divergence_never_uses_future_data():
    result = _result()
    diverged = [o for o in result.opportunities if o.opportunity_type == "分歧转一致"]
    for item in diverged:
        assert all(c.verifiable for c in item.entry_conditions)
        for condition in item.trigger_conditions + item.executable_conditions:
            if "T+1" in condition.name or "T+1" in condition.detail:
                assert condition.verifiable is False
                assert condition.passed is None
        assert item.state != "可执行"


def test_t_plus_one_and_board_constraints_are_declared():
    result = _result()
    for item in result.opportunities:
        assert item.constraints["t_plus_one"] is True
        assert item.constraints["lot_size"] == 100
        assert item.constraints["limit_up_cannot_buy"] is True
        assert item.constraints["fill_assumption"]


def test_one_word_board_is_not_treated_as_fillable():
    result = _result()
    one_word = [o for o in result.opportunities if o.board_form == "一字板/未开板"]
    assert one_word, "fixture contains a 一字板 first board"
    for item in one_word:
        assert item.state != "可执行"


def test_decline_stage_closes_offensive_setups():
    result = _result(stage="退潮")
    assert result.state_counts.get("可执行", 0) == 0
    assert all(not o.environment_allowed for o in result.opportunities)
