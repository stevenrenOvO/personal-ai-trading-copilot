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
    assert _time.monotonic() - started < 0.2 * len(codes) / 2  # well under the sequential 1.2 s
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
