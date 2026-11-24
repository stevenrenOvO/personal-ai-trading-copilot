"""V1.2 daily loop: band rules, adjusted bars, real ledger, plans, pre-trade check."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from backend.personal import RiskProfile
from backend.personal import adjusted as adjusted_mod
from backend.personal.adjusted import AdjustedBarStore
from backend.personal.band_rules import (
    BELOW_MA, BUY, CROSS_NO_VOLUME, NO_DATA, WAIT_CROSS, WEEKLY_FAIL, evaluate,
)
from backend.personal.plans import PlanStore
from backend.personal.real_account import RealTradeStore
from tests.test_api_v1 import _call, _make_runtime


def bars(closes, vols=None, start="2025-01-06"):
    dates = pd.bdate_range(start, periods=len(closes)).strftime("%Y%m%d")
    vols = vols if vols is not None else [1e6] * len(closes)
    return pd.DataFrame({"trade_date": dates, "open": closes, "high": [c * 1.01 for c in closes],
                         "low": [c * 0.99 for c in closes], "close": closes, "vol": vols})


# Uptrend, shallow pullback (DIF back near zero, green bars), then a bounce.
BASE = list(np.linspace(10, 14, 140)) + list(np.linspace(14, 13.6, 12)) + [13.62, 13.9]


# -- band rules -------------------------------------------------------------------

def test_macd_matches_an_independent_ema_recurrence():
    closes = BASE + [14.4]

    def ema(values, span):
        k, out = 2 / (span + 1), [values[0]]
        for v in values[1:]:
            out.append(out[-1] + k * (v - out[-1]))
        return out

    dif = [a - b for a, b in zip(ema(closes, 12), ema(closes, 26))]
    dea = ema(dif, 9)
    r = evaluate(bars(closes), last_week_complete=True)
    assert r["macd"]["dif"] == pytest.approx(dif[-1], abs=1e-4)
    assert r["macd"]["dea"] == pytest.approx(dea[-1], abs=1e-4)
    assert r["macd"]["hist_last3"][-1] == pytest.approx(2 * (dif[-1] - dea[-1]), abs=1e-4)


def test_buy_point_requires_volume_on_the_cross():
    closes = BASE + [14.4]
    r = evaluate(bars(closes, [1e6] * (len(closes) - 1) + [2e6]), last_week_complete=True)
    assert r["entry"] == BUY
    assert r["macd"]["cross"]["volume_ratio"] == 2.0 and r["macd"]["cross"]["near_zero"]
    assert all(c["passed"] for c in r["conditions"] if c["key"] != "green_shrinking")
    quiet = evaluate(bars(closes, [1e6] * (len(closes) - 1) + [1.1e6]), last_week_complete=True)
    assert quiet["entry"] == CROSS_NO_VOLUME


def test_waiting_for_cross_while_green_bars_shrink():
    r = evaluate(bars(BASE), last_week_complete=True)
    assert r["entry"] == WAIT_CROSS
    assert r["macd"]["green_shrinking"] and r["macd"]["near_zero"] and r["macd"]["cross"] is None


def test_weekly_trend_gates_everything():
    assert evaluate(bars(list(np.linspace(14, 10, 150))), last_week_complete=True)["entry"] == WEEKLY_FAIL


def test_effective_break_of_ma21():
    r = evaluate(bars(list(np.linspace(10, 14, 150)) + [13.5]), last_week_complete=True)
    assert r["entry"] == BELOW_MA
    ma = r["ma"]
    assert ma["effective_break"] and ma["close"] < ma["ma21"] * 0.99
    assert ma["suggested_stop"] == pytest.approx(ma["ma21"] * 0.99, abs=0.01)


def test_in_progress_week_is_not_used_as_weekly_close():
    closes = BASE + [14.4]
    done = evaluate(bars(closes), last_week_complete=True)["weekly"]
    partial = evaluate(bars(closes), last_week_complete=False)["weekly"]
    assert partial["week_end"] < done["week_end"]
    assert "in_progress" in partial


def test_too_few_bars_gives_no_verdict():
    r = evaluate(bars(list(np.linspace(10, 11, 60))))
    assert r["entry"] == NO_DATA and "60" in r["reason"]


def test_exit_switches_to_trailing_after_15_percent():
    closes = list(np.linspace(10, 12, 120))
    frame = bars(closes)
    entry_date, cost = frame["trade_date"].iloc[100], closes[100]
    # +20% then a fall of 11% from the highest close.
    closes += list(np.linspace(cost * 1.02, cost * 1.20, 8)) + [cost * 1.20 * 0.89]
    r = evaluate(bars(closes), last_week_complete=True,
                 position={"entry_price": cost, "entry_date": entry_date})
    ex = r["exit"]
    assert ex["trailing_mode"] and ex["max_profit_pct"] == pytest.approx(0.20, abs=1e-3)
    assert ex["verdict"] == "离场：自最高回撤"
    assert ex["stop"] == pytest.approx(cost * 1.20 * 0.90, abs=0.01)


def test_exit_below_15_percent_uses_ma21_and_adjusts_cost_for_dividends():
    closes = list(np.linspace(10, 12, 140))
    frame = bars(closes)
    entry_date, adj_close = frame["trade_date"].iloc[130], closes[130]
    closes += [closes[-1] * 0.97]  # >1% under MA21
    position = {"entry_price": adj_close * 1.1, "entry_date": entry_date, "entry_close_raw": adj_close * 1.1}
    ex = evaluate(bars(closes), last_week_complete=True, position=position)["exit"]
    assert not ex["trailing_mode"] and ex["verdict"] == "离场：有效跌破21日线"
    assert ex["cost_adjusted"] == pytest.approx(adj_close, abs=1e-3)
    assert "复权系数" in ex["note"]


# -- adjusted bars ----------------------------------------------------------------

class _History:
    def __init__(self, frame):
        self.frame = frame

    def load(self, ts_codes=None, start_date=None, end_date=None):
        f = self.frame
        if end_date:
            f = f[f["trade_date"] <= end_date]
        return f.copy()


def _qfq(n=120, last="20261113"):
    dates = pd.bdate_range(end=pd.Timestamp(last), periods=n).strftime("%Y%m%d")
    closes = np.linspace(10, 12, n)
    return pd.DataFrame({"trade_date": dates, "open": closes, "high": closes, "low": closes,
                         "close": closes, "vol": 1e6})


@pytest.fixture(autouse=True)
def _clear_backoff():
    adjusted_mod._FAILED.clear()
    yield
    adjusted_mod._FAILED.clear()


def test_adjusted_series_fetched_once_per_day_and_raw_today_appended(tmp_path):
    calls = []
    store = AdjustedBarStore(tmp_path, fetcher=lambda code: calls.append(code) or _qfq())
    raw_today = pd.DataFrame([{"trade_date": "20261116", "ts_code": "000001.SZ", "open": 12.0, "high": 12.3,
                               "low": 11.9, "close": 12.2, "pre_close": 12.0, "vol": 2e6}])
    frame, basis = store.bars("000001.SZ", "20261116", _History(raw_today))
    assert calls == ["000001.SZ"] and basis["adjusted"] and basis["appended_raw_dates"] == ["20261116"]
    assert frame["trade_date"].iloc[-1] == "20261116" and not basis["warnings"]
    store.bars("000001.SZ", "20261116", _History(raw_today))
    assert len(calls) == 1  # cached for the day


def test_ex_dividend_day_is_not_appended_to_the_adjusted_series(tmp_path):
    store = AdjustedBarStore(tmp_path, fetcher=lambda code: _qfq())
    raw_today = pd.DataFrame([{"trade_date": "20261116", "ts_code": "000001.SZ", "open": 11.0, "high": 11.1,
                               "low": 10.9, "close": 11.0, "pre_close": 11.5, "vol": 1e6}])
    frame, basis = store.bars("000001.SZ", "20261116", _History(raw_today))
    assert frame["trade_date"].iloc[-1] == "20261113" and not basis["appended_raw_dates"]
    assert any("除权" in w for w in basis["warnings"])


def test_fetch_failure_falls_back_to_cache_then_backs_off(tmp_path):
    good = AdjustedBarStore(tmp_path, fetcher=lambda code: _qfq())
    good.bars("000001.SZ", "20261113")
    calls = []

    def broken(code):
        calls.append(code)
        raise ConnectionError("offline")

    store = AdjustedBarStore(tmp_path, fetcher=broken)
    frame, basis = store.bars("000001.SZ", "20261116")
    assert not frame.empty and basis["adjusted"] and any("沿用" in w for w in basis["warnings"])
    store.bars("000001.SZ", "20261116")
    assert len(calls) == 1  # no retry storm


def test_no_cache_and_no_network_uses_unadjusted_and_says_so(tmp_path):
    raw = _qfq().assign(ts_code="000001.SZ", pre_close=None)
    store = AdjustedBarStore(tmp_path, fetcher=lambda code: (_ for _ in ()).throw(TimeoutError()))
    frame, basis = store.bars("000001.SZ", "20261113", _History(raw))
    assert len(frame) == 120 and basis["adjusted"] is False
    assert any("未复权" in w for w in basis["warnings"])


# -- real ledger ------------------------------------------------------------------

def _ledger(tmp_path):
    store, p = RealTradeStore(tmp_path), RiskProfile()
    store.record({"side": "BUY", "trade_date": "20261110", "ts_code": "000001.SZ", "price": 10, "shares": 200}, p,
                 today="20261117")
    store.record({"side": "BUY", "trade_date": "20261111", "ts_code": "000001.SZ", "price": 11, "shares": 100}, p,
                 today="20261117")
    store.record({"side": "SELL", "trade_date": "20261112", "ts_code": "000001.SZ", "price": 12, "shares": 100}, p,
                 today="20261117")
    return store, p


def test_ledger_replay_matches_hand_calculation(tmp_path):
    # Buy 200@10: 2,000 + 5 min commission. Buy 100@11: 1,100 + 5.
    # Sell 100@12: 1,200 - (5 commission + 0.6 stamp); released cost 3,110/3 = 1,036.67.
    store, p = _ledger(tmp_path)
    acct = store.account(p)
    pos = acct["positions"][0]
    assert pos["shares"] == 200 and pos["cost"] == pytest.approx(2073.33, abs=0.01)
    assert pos["avg_cost"] == pytest.approx(10.3667, abs=1e-4) and pos["entry_date"] == "20261110"
    assert acct["realized_pnl"] == pytest.approx(157.73, abs=0.01)
    assert acct["cash"] == pytest.approx(12000 - 2005 - 1105 + 1194.4, abs=0.01)
    sell = store.list(date="20261112")[0]
    assert sell["fee"] == 5.6 and sell["fee_source"] == "estimated"


def test_ledger_rejects_bad_trades(tmp_path):
    store, p = _ledger(tmp_path)
    before = store.path.read_text(encoding="utf-8")
    cases = [
        ({"side": "SELL", "trade_date": "20261113", "ts_code": "000001.SZ", "price": 12, "shares": 300}, "持有 200"),
        ({"side": "SELL", "trade_date": "20261111", "ts_code": "000001.SZ", "price": 12, "shares": 300}, "超过当时持仓"),
        ({"side": "BUY", "trade_date": "20261113", "ts_code": "000001.SZ", "price": 12, "shares": 150}, "整数倍"),
        ({"side": "BUY", "trade_date": "20261120", "ts_code": "000001.SZ", "price": 12, "shares": 100}, "未来"),
        ({"side": "HOLD", "trade_date": "20261113", "ts_code": "000001.SZ", "price": 12, "shares": 100}, "方向"),
    ]
    for body, message in cases:
        with pytest.raises(ValueError, match=message):
            store.record(body, p, today="20261117")
    assert store.path.read_text(encoding="utf-8") == before


def test_void_keeps_the_record_and_cannot_orphan_a_sell(tmp_path):
    store, p = _ledger(tmp_path)
    store.record({"side": "SELL", "trade_date": "20261113", "ts_code": "000001.SZ", "price": 12, "shares": 100},
                 p, today="20261117")  # 300 bought, 200 sold
    first = store.list()[0]
    with pytest.raises(ValueError, match="原因"):
        store.void(first["id"], "")
    before = store.path.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="超过当时持仓"):
        store.void(first["id"], "录错")  # 100 bought vs 200 sold
    assert store.path.read_text(encoding="utf-8") == before
    sell = store.list(date="20261112")[0]
    voided = store.void(sell["id"], "重复录入")
    assert voided["voided"] and voided["void_reason"] == "重复录入"
    assert len(store.list()) == 4 and store.account(p)["positions"][0]["shares"] == 200


# -- plans ------------------------------------------------------------------------

def test_plan_limits_and_revisions(tmp_path):
    store = PlanStore(tmp_path)
    cand = lambda code: {"ts_code": code, "entry_price": 10, "stop_price": 9.5}  # noqa: E731
    with pytest.raises(ValueError, match="最多 3"):
        store.save({"date": "20261117", "candidates": [cand(f"00000{i}.SZ") for i in range(1, 5)]})
    with pytest.raises(ValueError, match="重复"):
        store.save({"date": "20261117", "candidates": [cand("000001.SZ"), cand("000001.SZ")]})
    with pytest.raises(ValueError, match="止损价须低于"):
        store.save({"date": "20261117", "candidates": [{"ts_code": "000001.SZ", "entry_price": 10, "stop_price": 10}]})
    first = store.save({"date": "20261117", "candidates": [cand("000001.SZ")],
                        "holdings": [{"ts_code": "600519.SH", "action": "持有"}]})
    assert first["revision"] == 1
    second = store.save({"date": "20261117", "candidates": [cand("000002.SZ")]})
    assert second["revision"] == 2 and second["created_at"] == first["created_at"]
    assert second["revisions"][0]["candidates"][0]["ts_code"] == "000001.SZ"
    assert store.dates() == ["20261117"]


# -- API --------------------------------------------------------------------------

@pytest.fixture
def offline_qfq(monkeypatch):
    closes = BASE + [14.4]
    frame = bars(closes, [1e6] * (len(closes) - 1) + [2e6], start="2026-04-01")
    monkeypatch.setattr("backend.data.providers.tencent.fetch_qfq_daily", lambda code, **kw: frame.copy())
    monkeypatch.setattr("backend.personal.service.evaluation_day", lambda runtime: frame["trade_date"].iloc[-1])
    return frame


def test_daily_loop_api(tmp_path, offline_qfq):
    runtime = _make_runtime(tmp_path)
    assert _call("POST", "/personal/watchlist", runtime, {"ts_code": "000001.SZ", "name": "平安银行"})[0] == 200

    code, data = _call("GET", "/personal/watchlist/status", runtime)
    assert code == 200
    item = data["data"]["items"][0]
    assert item["entry"] == BUY and item["basis"]["source"] == "tencent_qfq"
    assert item["evaluation_day"] == offline_qfq["trade_date"].iloc[-1]

    code, data = _call("GET", "/personal/pretrade?ts_code=000001.SZ&entry=14.4&stop=13.8", runtime)
    assert code == 200
    checks = {c["key"]: c["level"] for c in data["data"]["checks"]}
    assert checks["rules"] == "pass" and checks["watchlist"] == "pass"
    assert checks["plan"] == "warn"  # no plan saved yet: flagged as unplanned
    assert data["data"]["verdict"] == "有警告"

    plan = {"date": "20261117", "candidates": [{"ts_code": "000001.SZ", "entry_price": 14.4, "stop_price": 13.8}]}
    assert _call("POST", "/personal/plan", runtime, plan)[0] == 200
    code, data = _call("GET", "/personal/plan?date=20261117", runtime)
    assert data["data"]["candidates"][0]["ts_code"] == "000001.SZ" and data["meta"]["dates"] == ["20261117"]

    day = offline_qfq["trade_date"].iloc[-1]
    trade = {"side": "BUY", "trade_date": day, "ts_code": "000001.SZ", "price": 14.4, "shares": 100,
             "plan_date": "20261117", "followed_plan": True}
    code, data = _call("POST", "/personal/trades", runtime, trade)
    assert code == 200 and data["data"]["id"].startswith(f"R{day}-")
    code, data = _call("GET", f"/personal/trades?date={day}", runtime)
    assert len(data["data"]["trades"]) == 1

    code, data = _call("GET", "/personal/account", runtime)
    pos = data["data"]["positions"][0]
    assert pos["shares"] == 100 and pos["exit"]["verdict"] == "持有"
    assert pos["exit"]["stop"] == pytest.approx(item["ma"]["suggested_stop"], abs=0.01)

    code, data = _call("POST", "/personal/trades/void", runtime, {"id": "nope", "reason": "x"})
    assert code == 404
    code, data = _call("POST", "/personal/trades", runtime, {**trade, "shares": 50})
    assert code == 400
    saved = json.loads((tmp_path / "personal" / "real_trades.json").read_text(encoding="utf-8"))
    assert len(saved["trades"]) == 1


# -- 1.2.1 fixes --------------------------------------------------------------------

def test_prefetch_downloads_in_parallel_and_reports_failures(tmp_path):
    import threading
    import time as _time

    active, peak, lock = [0], [0], threading.Lock()

    def slow(code):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        _time.sleep(0.2)
        with lock:
            active[0] -= 1
        if code == "000004.SZ":
            raise TimeoutError("vendor down")
        return _qfq()

    codes = [f"00000{i}.SZ" for i in range(1, 7)]
    store = AdjustedBarStore(tmp_path, fetcher=slow)
    started = _time.monotonic()
    store.prefetch(codes, "20261113")
    assert _time.monotonic() - started < 0.2 * len(codes)  # faster than sequential; overlap checked below
    assert peak[0] > 1

    calls = []
    store._fetch = lambda code: calls.append(code) or _qfq()
    frame, basis = store.bars("000001.SZ", "20261113")
    assert not frame.empty and basis["adjusted"] and calls == []  # served from the prefetch
    _, failed = store.bars("000004.SZ", "20261113")
    assert any("TimeoutError" in w for w in failed["warnings"]) and calls == []


def test_default_fetcher_uses_short_timeout(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr("backend.data.providers.tencent.fetch_qfq_daily",
                        lambda code, **kw: seen.update(kw) or _qfq())
    AdjustedBarStore(tmp_path).bars("000001.SZ", "20261113")
    assert seen["timeout"] <= 5


def test_limit_checks_by_board_and_session():
    from types import SimpleNamespace

    from backend.personal.service import limit_checks, limit_rate

    assert limit_rate("600519.SH") == 0.10 and limit_rate("600001.SH", "*ST某某") == 0.05
    assert limit_rate("300750.SZ", "ST创业") == 0.20 and limit_rate("830799.BJ") == 0.30
    intraday, final = SimpleNamespace(status="ok"), SimpleNamespace(status="final")
    sealed = {"price": 11.0, "pre_close": 10.0, "name": "某股"}
    items = limit_checks("600000.SH", sealed, intraday, 11.0)
    assert [lvl for lvl, _ in items] == ["warn"] and "封涨停" in items[0][1]
    items = limit_checks("600000.SH", {"price": 10.2, "pre_close": 10.0}, intraday, 11.5)
    assert items == [("fail", "买入价 11.50 高于今日涨停价 11.00，无法成交")]
    # After the close the next session's limit comes from today's close (10.2 -> 11.22).
    assert limit_checks("600000.SH", {"price": 10.2, "pre_close": 10.0}, final, 11.2) == []
    items = limit_checks("600000.SH", {"price": 10.2, "pre_close": 10.0}, final, 11.3)
    assert items[0][0] == "fail" and "下一交易日涨停价 11.22" in items[0][1]
    st = limit_checks("600001.SH", {"price": 10.5, "pre_close": 10.0, "name": "ST某某"}, intraday, 10.5)
    assert st and st[0][0] == "warn"
    assert limit_checks("600000.SH", None, intraday, 10) == []


# -- 1.2.2 execution list (close-based, no intraday watching) ------------------------

def _session(monkeypatch, phase, day="20261117"):
    monkeypatch.setattr("backend.personal.service._session_info",
                        lambda: {"phase": phase, "current_trading_day": day})


def test_after_close_list_is_for_the_next_session_with_placeable_orders(tmp_path, offline_qfq, monkeypatch):
    day = offline_qfq["trade_date"].iloc[-1]
    _session(monkeypatch, "盘后", day)
    runtime = _make_runtime(tmp_path)
    _call("POST", "/personal/watchlist", runtime, {"ts_code": "000001.SZ", "name": "平安银行"})
    PlanStore(tmp_path).save({"date": day, "candidates": [
        {"ts_code": "000001.SZ", "entry_price": 14.4, "stop_price": 13.8},
        {"ts_code": "600000.SH", "entry_condition": "回踩21日线", "entry_price": 8, "stop_price": 7.6, "shares": 200}]})
    today = _call("GET", "/personal/today", runtime)[1]["data"]
    assert today["mode"] == "next" and today["title"].startswith("执行单 · ") and "的执行单" in today["notice"]
    kinds = [a["kind"] for a in today["actions"]]
    assert kinds == ["buy", "plan"]  # plan for tomorrow already saved: no reminder
    buy, plan = today["actions"]
    limit = round(14.4 * 1.02, 2)  # last close 14.4 + 2% chase limit
    assert buy["order"]["limit"] == limit and buy["order"]["side"] == "buy"
    assert f"下一交易日限价 ≤ {limit}" in buy["next_step"] and f"开盘高于 {limit} 就放弃" in buy["next_step"]
    assert buy["order"]["shares"] % 100 == 0 and buy["order"]["shares"] > 0
    assert "计划内" in buy["detail"] and buy["key"] == "buy:000001.SZ" and buy["mark"] is None
    assert plan["ts_code"] == "600000.SH" and "限价 ≤ 8" in plan["next_step"] and "200 股" in plan["next_step"]


def test_intraday_list_uses_last_close_and_reminds_after_close_only(tmp_path, offline_qfq, monkeypatch):
    day = offline_qfq["trade_date"].iloc[-1]
    _session(monkeypatch, "盘中", "20261118")
    runtime = _make_runtime(tmp_path)
    today = _call("GET", "/personal/today", runtime)[1]["data"]
    assert today["mode"] == "today" and today["title"] == "今日执行单"
    assert "盘中不必盯盘" in today["notice"] and today["basis_date"] == day
    assert [a["kind"] for a in today["actions"]] == ["reminder"]  # never planned
    _session(monkeypatch, "盘后", day)
    PlanStore(tmp_path).save({"date": "20260101", "candidates": []})
    actions = _call("GET", "/personal/today", runtime)[1]["data"]["actions"]
    assert [a["title"] for a in actions] == ["保存明日计划"]


def test_exit_signal_comes_first_and_holdings_get_stop_orders(tmp_path, monkeypatch):
    closes = list(np.linspace(10, 12, 140))
    frame = bars(closes + [closes[-1] * 0.97], start="2026-05-01")
    day = frame["trade_date"].iloc[-1]
    monkeypatch.setattr("backend.data.providers.tencent.fetch_qfq_daily", lambda code, **kw: frame.copy())
    monkeypatch.setattr("backend.personal.service.evaluation_day", lambda runtime: day)
    _session(monkeypatch, "盘中", day)
    runtime = _make_runtime(tmp_path)
    entry_day = frame["trade_date"].iloc[130]
    code, data = _call("POST", "/personal/watchlist/hold", runtime,
                       {"ts_code": "000001.SZ", "cost": 11.9, "entry_date": entry_day, "shares": 100})
    assert code == 200 and data["data"]["holding"]["shares"] == 100
    today = _call("GET", "/personal/today", runtime)[1]["data"]
    first = today["actions"][0]
    assert first["kind"] == "sell" and "有效跌破21日线" in first["detail"] and "100 股" in first["next_step"]
    assert today["counts"]["sell"] == 1 and today["account"]["cash"] == 12000 - 1190


def test_holding_far_from_stop_gets_conditional_stop_order(tmp_path, offline_qfq, monkeypatch):
    day = offline_qfq["trade_date"].iloc[-1]
    _session(monkeypatch, "盘后", day)
    runtime = _make_runtime(tmp_path)
    PlanStore(tmp_path).save({"date": day, "candidates": []})
    _call("POST", "/personal/watchlist/hold", runtime,
          {"name": "平安银行 000001.SZ", "cost": 12.0, "entry_date": offline_qfq["trade_date"].iloc[-60]})
    actions = _call("GET", "/personal/today", runtime)[1]["data"]["actions"]
    kinds = [a["kind"] for a in actions]
    assert kinds[-2:] == ["hold", "none"] or kinds[-1] == "hold"
    hold = next(a for a in actions if a["kind"] in ("hold", "near_stop"))
    assert hold["next_step"].startswith("止损条件单：价格 ≤") and "（全部）" in hold["next_step"]


def test_mark_execution_and_show_it_on_the_list(tmp_path, offline_qfq, monkeypatch):
    day = offline_qfq["trade_date"].iloc[-1]
    _session(monkeypatch, "盘后", day)
    runtime = _make_runtime(tmp_path)
    _call("POST", "/personal/watchlist", runtime, {"ts_code": "000001.SZ"})
    code, data = _call("POST", "/personal/today/mark", runtime,
                       {"basis": day, "key": "buy:000001.SZ", "status": "skipped", "note": "高开 4%"})
    assert code == 200 and data["data"]["status"] == "skipped"
    buy = next(a for a in _call("GET", "/personal/today", runtime)[1]["data"]["actions"] if a["kind"] == "buy")
    assert buy["mark"]["status"] == "skipped" and buy["mark"]["note"] == "高开 4%"
    assert _call("POST", "/personal/today/mark", runtime, {"basis": day, "key": "rm -rf", "status": "done"})[0] == 400
    _call("POST", "/personal/today/mark", runtime, {"basis": day, "key": "buy:000001.SZ", "status": None})
    buy = next(a for a in _call("GET", "/personal/today", runtime)[1]["data"]["actions"] if a["kind"] == "buy")
    assert buy["mark"] is None


# -- 1.2.2 names instead of codes, holdings in the watchlist ---------------------------

class _Basic:
    def __init__(self, rows):
        self.rows = rows

    def stock_basic(self):
        return pd.DataFrame(self.rows, columns=["ts_code", "name"])


def test_search_and_resolve_by_name():
    from types import SimpleNamespace

    from backend.personal.stocks import resolve_stock, search_stocks

    rt = SimpleNamespace(provider=_Basic([("601318.SH", "中国平安"), ("000001.SZ", "平安银行"),
                                          ("001359.SZ", "平安电工"), ("600519.SH", "贵州茅台")]))
    assert [s["name"] for s in search_stocks(rt, "平安")] == ["平安银行", "平安电工", "中国平安"]
    assert resolve_stock(rt, "茅台") == ("600519.SH", "贵州茅台")
    assert resolve_stock(rt, "平安银行") == ("000001.SZ", "平安银行")
    assert resolve_stock(rt, "中国平安 601318.SH") == ("601318.SH", "中国平安")
    assert resolve_stock(rt, "600519") == ("600519.SH", "贵州茅台")
    with pytest.raises(ValueError, match="匹配到多只"):
        resolve_stock(rt, "平安")
    with pytest.raises(ValueError, match="没有找到"):
        resolve_stock(rt, "不存在的股")


def test_hold_and_sell_keep_history_and_guard_removal(tmp_path):
    from backend.personal import WatchlistStore

    store = WatchlistStore(tmp_path)
    store.hold("600519.SH", cost=1400, entry_date="20261110", shares=100, name="贵州茅台", today="20261118")
    with pytest.raises(ValueError, match="先标记卖出"):
        store.remove("600519.SH")
    with pytest.raises(ValueError, match="不能早于买入日期"):
        store.sell("600519.SH", exit_date="20261101", today="20261118")
    assert store.holdings()[0]["holding"]["cost"] == 1400
    record = store.sell("600519.SH", exit_date="20261117", exit_price=1470, today="20261118")
    assert record["return_pct"] == pytest.approx(0.05) and store.holdings() == []
    assert store.closed()[0]["exit_date"] == "20261117" and store.list()[0]["ts_code"] == "600519.SH"
    with pytest.raises(ValueError, match="未来"):
        store.hold("600519.SH", cost=10, entry_date="20261120", today="20261118")
    with pytest.raises(ValueError, match="正整数"):
        store.hold("600519.SH", cost=10, entry_date="20261118", shares=1.5, today="20261118")


def test_pretrade_accepts_a_name_and_defaults_to_next_day_limit(tmp_path, offline_qfq, monkeypatch):
    monkeypatch.setattr("backend.personal.stocks._basic",
                        lambda runtime: pd.DataFrame([("000001.SZ", "平安银行")], columns=["ts_code", "name"]))
    runtime = _make_runtime(tmp_path)
    code, data = _call("GET", "/personal/pretrade?ts_code=%E5%B9%B3%E5%AE%89%E9%93%B6%E8%A1%8C", runtime)
    assert code == 200 and data["data"]["ts_code"] == "000001.SZ" and data["data"]["name"] == "平安银行"
    assert data["data"]["inputs"]["entry_price"] == round(14.4 * 1.02, 2)
    assert "收盘价 + 追高上限 2%" in data["data"]["inputs"]["entry_source"]


def test_next_session_skips_weekends_and_is_named():
    from types import SimpleNamespace

    from backend.personal.service import _session_label, next_session

    class Cal:
        def trade_cal(self, start_date, end_date, is_open=None):
            days = ["20261120", "20261123", "20261124"]
            return pd.DataFrame({"cal_date": [d for d in days if start_date <= d <= end_date]})

    runtime = SimpleNamespace(provider=Cal())
    assert next_session(runtime, "20261120") == "20261123"
    assert _session_label("20261123") == "11-23 周一"
    assert next_session(SimpleNamespace(provider=None), "20261120") is None
    assert _session_label(None) == "下一交易日"


def test_usage_record_counts_named_events_only(tmp_path):
    from backend.personal.usage import UsageStore

    store = UsageStore(tmp_path)
    store.record("pane:watch", day="20261118")
    store.record("pane:watch", day="20261120")
    store.record("action:pretrade", day="20261120")
    summary = store.summary(days=30, today="20261120")
    assert summary["active_days"] == 2 and summary["events"] == {"pane:watch": 2, "action:pretrade": 1}
    assert store.summary(days=1, today="20261120")["events"] == {"pane:watch": 1, "action:pretrade": 1}
    for bad in ("600519.SH", "pane:Watch", "action:" + "x" * 40, "", None, "price:1450"):
        with pytest.raises(ValueError):
            store.record(bad)


def test_usage_api_roundtrip(tmp_path):
    runtime = _make_runtime(tmp_path)
    assert _call("POST", "/personal/usage", runtime, {"event": "tab:account"})[0] == 200
    assert _call("POST", "/personal/usage", runtime, {"event": "贵州茅台"})[0] == 400
    code, data = _call("GET", "/personal/usage?days=7", runtime)
    assert code == 200 and data["data"]["events"] == {"tab:account": 1} and "不联网" in data["data"]["note"]


# -- review fixes ----------------------------------------------------------------------

def test_market_state_falls_back_to_the_basis_days_stored_bars(tmp_path):
    from types import SimpleNamespace

    from backend.data.history import MarketHistory
    from backend.personal.service import market_state_for

    history = MarketHistory(tmp_path / "h.db")
    rows = [{"ts_code": f"{600000 + i}.SH", "trade_date": "20261119", "open": 10, "high": 10.2, "low": 9.6,
             "close": 9.7, "pre_close": 10, "change": -0.3, "pct_chg": -3.0, "vol": 1000, "amount": 1e6}
            for i in range(1200)]
    history.ingest(pd.DataFrame(rows))
    runtime = SimpleNamespace(provider=SimpleNamespace(history=history))
    state = market_state_for(runtime, "20261119")   # no closing snapshot of that day is cached
    assert state["state"] and state["source"] == "本地日线（收盘）"
    assert '"date": "20261119"' in history.get_state("market_state:20261119")
    assert market_state_for(runtime, "20261118")["state"] is None   # no bars: unknown, not guessed


def test_unknown_share_count_blocks_buy_sizing_and_asks_for_it(tmp_path, offline_qfq, monkeypatch):
    day = offline_qfq["trade_date"].iloc[-1]
    _session(monkeypatch, "盘后", day)
    runtime = _make_runtime(tmp_path)
    PlanStore(tmp_path).save({"date": day, "candidates": []})
    _call("POST", "/personal/watchlist", runtime, {"ts_code": "000001.SZ"})
    _call("POST", "/personal/watchlist/hold", runtime,
          {"ts_code": "600000.SH", "cost": 12, "entry_date": offline_qfq["trade_date"].iloc[-30]})
    actions = _call("GET", "/personal/today", runtime)[1]["data"]["actions"]
    buy = next(a for a in actions if a["kind"] == "buy")
    assert buy["level"] == "warn" and "补填股数" in buy["next_step"] and buy["order"] is None
    reminder = next(a for a in actions if a["title"].startswith("补填股数"))
    assert reminder["ts_code"] == "600000.SH" and reminder["key"] == "reminder:600000.SH"


def test_plan_candidate_with_prices_carries_an_order(tmp_path, offline_qfq, monkeypatch):
    day = offline_qfq["trade_date"].iloc[-1]
    _session(monkeypatch, "盘后", day)
    runtime = _make_runtime(tmp_path)
    PlanStore(tmp_path).save({"date": day, "candidates": [
        {"ts_code": "600000.SH", "entry_price": 8, "stop_price": 7.6, "shares": 200}]})
    plan = next(a for a in _call("GET", "/personal/today", runtime)[1]["data"]["actions"] if a["kind"] == "plan")
    assert plan["order"] == {"side": "buy", "limit": 8, "shares": 200, "stop": 7.6}


def test_exit_note_when_bought_before_adjusted_series():
    closes = list(np.linspace(10, 12, 140))
    frame = bars(closes)
    ex = evaluate(frame, last_week_complete=True,
                  position={"entry_price": 9.5, "entry_date": "20240101"})["exit"]
    assert ex["verdict"] in ("持有", "离场：有效跌破21日线", "离场：自最高回撤")
    assert "早于复权数据起点" in ex["note"]


# -- 1.2.3 plain language ------------------------------------------------------------

def test_bought_after_last_close_gets_a_stop_line_not_no_data():
    closes = list(np.linspace(10, 12, 140))
    frame = bars(closes)
    ex = evaluate(frame, last_week_complete=True,
                  position={"entry_price": 12.1, "entry_date": "20991231"})["exit"]
    assert ex["verdict"] == "持有" and ex["fresh"] and ex["stop"] is not None
    assert ex["max_profit_pct"] == 0.0 and not ex["trailing_mode"]
    broken = evaluate(bars(closes + [closes[-1] * 0.95]), last_week_complete=True,
                      position={"entry_price": 11.5, "entry_date": "20991231"})["exit"]
    assert broken["fresh"] and "不该买" in broken["note"]


def test_watch_words_are_plain():
    from backend.personal.service import watch_word

    assert watch_word({"entry": BUY}) == ("可以买", "ok", "买入信号出现了")
    assert watch_word({"entry": WAIT_CROSS})[0] == "快了"
    assert watch_word({"entry": WEEKLY_FAIL}) == ("等待", "muted", "趋势不对，先不看")
    held = {"held": True, "exit": {"verdict": "持有", "stop": 10.0, "close": 11.0, "profit_pct": 0.064}}
    assert watch_word(held) == ("安全", "muted", "赚 6.4% · 离止损还有 10%")
    held["exit"].update(close=10.2)
    assert watch_word(held) == ("留意", "warn", "赚 6.4% · 离止损只差 2.0%")
    held["exit"].update(close=9.8, fresh=True)
    assert watch_word(held) == ("留意", "warn", "今天刚买 · 已在止损线下方")
    held["exit"].update(verdict="离场：有效跌破21日线", fresh=False, profit_pct=-0.021)
    assert watch_word(held) == ("该卖了", "danger", "亏 2.1% · 跌破止损线")


def test_today_has_a_one_line_conclusion_and_plain_instructions(tmp_path, offline_qfq, monkeypatch):
    day = offline_qfq["trade_date"].iloc[-1]
    _session(monkeypatch, "盘后", day)
    runtime = _make_runtime(tmp_path)
    _call("POST", "/personal/watchlist", runtime, {"ts_code": "000001.SZ", "name": "平安银行"})
    PlanStore(tmp_path).save({"date": day, "candidates": []})
    today = _call("GET", "/personal/today", runtime)[1]["data"]
    hero = today["hero"]
    assert hero["headline"] == "要做 1 件事" and "不用盯盘" in hero["day_line"]
    assert hero["basis_line"].startswith("依据 ") and not today["setup"]
    buy = today["actions"][0]
    limit = round(14.4 * 1.02, 2)
    assert buy["headline"] == "买入 平安银行"
    assert buy["instruction"] == f"挂限价 {limit:,.2f} 元，买 {buy['order']['shares']} 股"
    assert buy["sub"].startswith(f"开盘高于 {limit:,.2f} 就不买")
    empty = _call("GET", "/personal/today", _make_runtime(tmp_path / "fresh"))[1]["data"]
    assert empty["setup"] is True
