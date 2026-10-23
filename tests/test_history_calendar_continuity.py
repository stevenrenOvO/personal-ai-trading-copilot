"""Prevent missing exchange sessions and first-day listings fabricating boards."""

import sqlite3
from contextlib import closing

import pandas as pd
import pytest

from backend.board.engine import BoardEngine
from backend.board.ladder import LimitLadderEngine
from backend.data.history import MarketHistory
from backend.data.providers.history import HistoryProvider
from backend.market.engine import MarketEngine
from backend.market.limit_ladder import consecutive_limit_up_heights


def _provider(root, dates, *, listed="19910101"):
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "ts_code": "000001.SZ", "symbol": "000001", "name": "测试股",
        "industry": "银行", "list_date": listed, "list_status": "L", "is_st": "N",
    }]).to_csv(root / "stock_basic.csv", index=False)
    rows = []
    pre = 10.0
    for day in dates:
        close = round(pre * 1.1, 2)
        rows.append({
            "trade_date": day, "ts_code": "000001.SZ", "open": pre,
            "high": close, "low": pre, "close": close, "pre_close": pre,
            "change": close - pre, "pct_chg": 10.0, "vol": 1000.0, "amount": 10000.0,
        })
        pre = close
    MarketHistory(root / "history" / "market_history.db").ingest(pd.DataFrame(rows))
    return HistoryProvider(base_dir=root)


def _session_calendar(root, rows):
    path = root / "cache" / "session.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS calendar (cal_date TEXT PRIMARY KEY, is_open INTEGER)")
        conn.executemany("INSERT OR REPLACE INTO calendar VALUES (?, ?)", rows)
        conn.commit()


def test_cached_exchange_calendar_keeps_missing_open_day_and_closed_days(tmp_path):
    provider = _provider(tmp_path, ["20240102", "20240104"])
    _session_calendar(tmp_path, [("20240102", 1), ("20240103", 1), ("20240104", 1), ("20240106", 0)])
    assert provider.trade_cal(is_open=True)["cal_date"].tolist() == ["20240102", "20240103", "20240104"]
    assert provider.trade_cal(is_open=False)["cal_date"].tolist() == ["20240106"]
    assert provider.trade_cal(start_date="20240103", end_date="20240104")["cal_date"].tolist() == ["20240103", "20240104"]


def test_calendar_cache_notices_session_calendar_created_after_initial_read(tmp_path):
    provider = _provider(tmp_path, ["20240102", "20240104"])
    assert provider.trade_cal()["cal_date"].tolist() == ["20240102", "20240104"]
    _session_calendar(tmp_path, [("20240103", 1)])
    assert provider.trade_cal()["cal_date"].tolist() == ["20240102", "20240103", "20240104"]


def test_session_calendar_closed_day_overrides_observed_and_csv_dates(tmp_path):
    provider = _provider(tmp_path, ["20240105", "20240106"])
    pd.DataFrame({"exchange": ["SSE"], "cal_date": ["20240106"], "is_open": [1]}).to_csv(tmp_path / "trade_cal.csv", index=False)
    _session_calendar(tmp_path, [("20240105", 1), ("20240106", 0)])
    assert provider.trade_cal(is_open=True)["cal_date"].tolist() == ["20240105"]


def test_absent_session_calendar_is_never_created_by_history_reader(tmp_path):
    provider = _provider(tmp_path, ["20240102"])
    provider.trade_cal()
    assert not (tmp_path / "cache" / "session.db").exists()


def test_missing_whole_open_day_breaks_market_board_and_ladder_streaks(tmp_path):
    provider = _provider(tmp_path, ["20240102", "20240104"])
    _session_calendar(tmp_path, [("20240102", 1), ("20240103", 1), ("20240104", 1)])
    assert MarketEngine(provider).calculate("20240104").max_consecutive_up == 1
    assert BoardEngine(provider).ladder("20240104").max_height == 1
    ladder = LimitLadderEngine(provider).snapshot("20240104")
    assert ladder.available
    assert ladder.max_height == 1
    assert ladder.yesterday is None
    assert ladder.recent[-1].promotion_rate is None
    assert ladder.recent[-1].avg_premium is None
    assert [item.date for item in ladder.recent] == ["20240102", "20240104"]
    assert any("20240103" in note for note in ladder.notes)


def test_weekend_does_not_break_a_streak(tmp_path):
    provider = _provider(tmp_path, ["20240105", "20240108"])
    _session_calendar(tmp_path, [("20240105", 1), ("20240106", 0), ("20240107", 0), ("20240108", 1)])
    assert MarketEngine(provider).calculate("20240108").max_consecutive_up == 2
    assert BoardEngine(provider).ladder("20240108").max_height == 2
    assert LimitLadderEngine(provider).snapshot("20240108").max_height == 2


@pytest.mark.parametrize("index", [0, 4, 5])
def test_listing_at_calendar_first_day_is_new_for_five_sessions(tmp_path, index):
    days = ["20240102", "20240103", "20240104", "20240105", "20240108", "20240109"]
    provider = _provider(tmp_path, days, listed=days[0])
    _session_calendar(tmp_path, [(day, 1) for day in days])
    ladder = LimitLadderEngine(provider).snapshot(days[index])
    assert ladder.excluded_new_listing == (1 if index < 5 else 0)
    assert ladder.max_height == (0 if index < 5 else 1)


def test_helper_uses_explicit_open_sessions_without_fetching_missing_limits(tmp_path):
    provider = _provider(tmp_path, ["20240102", "20240104"])
    calls = []
    def limits(day):
        calls.append(day)
        return provider.stk_limit(trade_date=day)
    assert consecutive_limit_up_heights(provider.daily(), "20240104", limits,
        trading_days=["20240102", "20240103", "20240104"]) == {"000001.SZ": 1}
    assert calls == ["20240102", "20240104"]
