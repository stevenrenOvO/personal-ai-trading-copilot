"""Tests for Step-4 stage 1: candidate pool + four-layer hard elimination."""

from __future__ import annotations

import pandas as pd

from backend.opportunity.candidates import CandidatePoolEngine, LAYERS

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
    """Synthetic market: two hot sectors, one cold sector, plus eliminations."""

    def __init__(self) -> None:
        rows: list[dict] = []
        # --- 银行 (hot): 000001.SZ runs 5 days (leader), 600000.SH 2 days,
        #     000002.SZ first board today (补涨)
        price = 10.0
        for day in DAYS:
            up = _up(price)
            rows.append(_bar(day, "000001.SZ", price, up, 10.0, open_=up, low=up))
            price = up
        for day in DAYS[-2:]:
            rows.append(_bar(day, "600000.SH", 20.0, 22.0, 10.0, open_=20.5, low=20.2))
        rows.append(_bar(DAYS[-1], "000002.SZ", 5.0, 5.5, 10.0, open_=5.1))
        # --- 医药: 000006.SZ 4 boards, 600005.SH 2 boards (squeezed)
        price = 30.0
        for day in DAYS[-4:]:
            up = _up(price)
            rows.append(_bar(day, "000006.SZ", price, up, 10.0))
            price = up
        for day in DAYS[-2:]:
            rows.append(_bar(day, "600005.SH", 8.0, 8.8, 10.0))
        # --- 软件 (cold): a single first board
        rows.append(_bar(DAYS[-1], "600004.SH", 12.0, 13.2, 10.0))
        # A falling name in the same sector so its relative strength is negative
        # (a lone limit-up would otherwise make the sector look "hot").
        rows.append(_bar(DAYS[-1], "600013.SH", 10.0, 9.5, -5.0))
        # --- eliminations
        rows.append(_bar(DAYS[-1], "600007.SH", 6.0, 6.6, 10.0))      # ST name
        rows.append(_bar(DAYS[-1], "600008.SH", 9.0, 9.9, 10.0))      # 次新
        # 地天板: opens limit-down (一字跌停) then closes at the up limit. It is
        # therefore a candidate, and the instrument layer must still drop it.
        rows.append(_bar(DAYS[-1], "600009.SH", 7.0, 7.7, 10.0, open_=6.3, low=6.3))
        price = 15.0
        for day in DAYS:
            up = _up(price)
            rows.append(_bar(day, "600010.SH", price, up, 10.0))
            price = up
        self._bars = pd.DataFrame(rows)
        self._basic = pd.DataFrame(
            [
                ("000001.SZ", "000001", "甲银行", "银行", "19910101", "N"),
                ("600000.SH", "600000", "乙银行", "银行", "19910101", "N"),
                ("000002.SZ", "000002", "丙银行", "银行", "19910101", "N"),
                ("000006.SZ", "000006", "己医药", "医药", "19910101", "N"),
                ("600005.SH", "600005", "戊医药", "医药", "19910101", "N"),
                ("600004.SH", "600004", "丁软件", "软件", "19910101", "N"),
                ("600007.SH", "600007", "ST庚公司", "银行", "19910101", "N"),
                ("600008.SH", "600008", "辛次新", "银行", DAYS[-3], "N"),
                ("600009.SH", "600009", "壬跌停", "银行", "19910101", "N"),
                ("600010.SH", "600010", "癸高位", "银行", "19910101", "N"),
                ("600013.SH", "600013", "甲软件", "软件", "19910101", "N"),
            ],
            columns=["ts_code", "symbol", "name", "industry", "list_date", "is_st"],
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
        rows = [
            {
                "trade_date": str(row["trade_date"]),
                "ts_code": str(row["ts_code"]),
                "up_limit": _up(float(row["pre_close"])),
                "down_limit": round(float(row["pre_close"]) * 0.9, 2),
            }
            for _, row in frame.iterrows()
        ]
        return pd.DataFrame(rows)

    def trade_cal(self, *, exchange=None, start_date=None, end_date=None, is_open=None):
        return pd.DataFrame(
            {"exchange": ["SSE"] * len(DAYS), "cal_date": DAYS, "is_open": [1] * len(DAYS)}
        )


def _pool(stage="发酵"):
    engine = CandidatePoolEngine(_Provider())
    return engine.build(DAYS[-1], stage=stage, stage_rule="test")


def test_layers_run_in_order_and_report_counts():
    pool = _pool()
    assert list(pool.layer_counts) == list(LAYERS)
    for layer in LAYERS:
        stats = pool.layer_counts[layer]
        assert stats["considered"] == stats["eliminated"] + stats["passed"]
    # Each layer receives exactly what the previous one passed.
    for previous, current in zip(LAYERS, LAYERS[1:]):
        assert pool.layer_counts[current]["considered"] == pool.layer_counts[previous]["passed"]


def test_instrument_layer_eliminates_st_new_listing_and_limit_down_open():
    pool = _pool()
    rules = {e.ts_code: e.rule for e in pool.eliminations}
    assert rules["600007.SH"] == "ST/*ST 标的"
    assert rules["600008.SH"] == "次新股"
    assert rules["600009.SH"] == "一字跌停开盘"
    assert all(e.layer == "标的层" for e in pool.eliminations if e.ts_code in
               {"600007.SH", "600008.SH", "600009.SH"})


def test_sector_layer_drops_the_cold_sector_and_keeps_the_ladder():
    pool = _pool()
    rules = {e.ts_code: e.rule for e in pool.eliminations}
    # A lone limit-up in a sector whose relative strength is negative is either
    # "脱节" or "孤立板" -- both are layer-2 rules.
    assert rules.get("600004.SH") in ("与热点板块脱节", "孤立板")
    survivors = {c.ts_code for c in pool.candidates}
    # The hot sector's leader (5板) and its first-board 补涨 survive; the 2-board
    # follower is squeezed by the higher board and is dropped in layer 3.
    assert {"000001.SZ", "000002.SZ"} <= survivors
    assert rules.get("600000.SH") == "同板块已被更高板压制"


def test_ladder_layer_drops_high_board_in_decline_and_squeezed_middle():
    pool = _pool(stage="退潮")
    rules = {e.ts_code: e.rule for e in pool.eliminations}
    assert rules.get("000001.SZ") == "高位板在退潮/分化阶段"   # 5 板 in 退潮
    # 600005.SH is a 2-board name in a sector that already has a 4-board.
    assert rules.get("600005.SH") == "同板块已被更高板压制"
    # The first board in the same hot sector is a 补涨 position and must stay.
    assert "000002.SZ" in {c.ts_code for c in pool.candidates}


def test_hard_elimination_cannot_be_compensated():
    """A strong-looking name that fails a layer is out, regardless of anything."""
    pool = _pool(stage="退潮")
    eliminated = {e.ts_code for e in pool.eliminations}
    candidates = {c.ts_code for c in pool.candidates}
    assert not (eliminated & candidates)


def test_pool_reports_no_high_quality_when_everything_needs_minute_data():
    pool = _pool()
    for candidate in pool.candidates:
        assert candidate.requires_minute_data is False or pool.has_high_quality is False


def test_unavailable_day_returns_no_pool():
    engine = CandidatePoolEngine(_Provider())
    pool = engine.build("20240108", stage="发酵")
    assert pool.available is False
    assert pool.candidates == ()
    assert pool.unavailable
