"""The dashboard must open on usable data and preserve its true basis."""
from datetime import datetime
from types import SimpleNamespace

import pandas as pd
import pytest

from backend.api import v1
from backend.api.data_status import history_dates
from backend.data import auto_ingest
from backend.data.providers.history import HistoryProvider
from backend.market.quote_service import QuoteService, get_quote_service, set_quote_service
from backend.market.session import TradingSession, set_trading_session
from tests.test_api_v1 import _call, _make_runtime


@pytest.fixture
def history_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(auto_ingest, "ensure_today_bars", lambda *args, **kwargs: False)
    runtime = _make_runtime(tmp_path)
    codes = [f"{i + 1:06}.SZ" for i in range(1000)]
    pd.DataFrame({"ts_code": codes, "name": codes, "industry": "银行", "symbol": codes,
        "list_date": "19900101", "list_status": "L", "is_st": "N"}).to_csv(tmp_path / "stock_basic.csv", index=False)
    runtime.provider = HistoryProvider(base_dir=tmp_path)
    def bars(day, n):
        return pd.DataFrame({"ts_code": codes[:n], "trade_date": day,
            "open": 10., "high": 10.5, "low": 9.5, "close": 10.1,
            "pre_close": 10., "pct_chg": 1., "vol": 1000., "amount": 10000.})
    runtime.provider.history.ingest(pd.concat([bars("20240102", 1000), bars("20240103", 400)]))
    session = TradingSession(db_path=tmp_path / "cache" / "session.db",
        calendar_fetcher=lambda start, end: [("20240102", 1), ("20240103", 1), ("20240104", 1)],
        now_fn=lambda: datetime(2024, 1, 4, 8, 0))
    set_trading_session(session)
    session.ensure_calendar()
    return runtime


def test_before_open_defaults_to_complete_day_and_separates_trade_day(history_runtime):
    status, response = _call("GET", "/data/latest", history_runtime)
    data = response["data"]
    assert status == 200 and data["date"] == data["latest_complete_date"] == "20240102"
    assert data["current_trading_day"] == "20240104"
    dates = {item["date"]: item for item in data["history_dates"]}
    assert dates["20240102"]["complete"] and not dates["20240103"]["complete"]
    assert dates["20240103"]["coverage_ratio"] == .4


def test_partial_historical_market_and_board_have_no_scores(history_runtime):
    for endpoint in ("/market/state", "/board/environment"):
        status, response = _call("GET", endpoint + "?date=20240103", history_runtime)
        assert status == 200 and response["meta"]["status"] == "no_data"
        assert response["data"]["score"] is None and response["data"]["recommended_exposure"] is None
    assert v1._history_board_extras(history_runtime, "20240103") is None


def test_completed_current_history_remains_visible_when_snapshot_is_missing(history_runtime):
    provider = history_runtime.provider
    codes = provider.stock_basic()["ts_code"].tolist()
    provider.history.ingest(pd.DataFrame({"ts_code": codes, "trade_date": "20240104",
        "open": 10., "high": 11., "low": 9., "close": 10.1,
        "pre_close": 10., "pct_chg": 1., "vol": 1000., "amount": 10000.}))
    history_runtime.analysis_cache.clear()
    for endpoint in ("/market/state", "/sectors", "/board/environment"):
        status, payload = _call("GET", endpoint + "?date=20240104", history_runtime)
        assert status == 200 and payload["meta"]["status"] == "ok"
        assert payload["data"]


def test_previous_closing_snapshot_is_displayed_on_its_actual_date(history_runtime, monkeypatch):
    codes = history_runtime.provider.stock_basic()["ts_code"].tolist()
    frame = pd.DataFrame({"ts_code": codes, "name": codes, "price": 10., "pre_close": 9.9,
        "pct_chg": 1., "change": .1, "open": 9.9, "high": 10., "low": 9.9,
        "volume": 1000., "amount": 10000.})
    service = QuoteService(db_path=history_runtime.root / "q.db",
        fetcher=lambda: (frame, {"quote_date": "20240103", "quote_time": "15:00:01", "source": "tencent"}))
    set_quote_service(service)
    assert service.refresh().status == "final"
    status, response = _call("GET", "/market/state?date=20240103", history_runtime)
    assert status == 200 and response["data"]["total_count"] == 1000
    assert response["meta"]["quote_date"] == "20240103"
    assert response["meta"]["status"] == "final"
    status, data = _call("GET", "/sectors?date=20240103", history_runtime)
    assert status == 200 and data["data"]


def test_tiny_snapshot_does_not_claim_to_cover_the_market(history_runtime, monkeypatch):
    from backend.market.quote_service import QuoteMeta
    monkeypatch.setattr(get_quote_service(), "get", lambda: (pd.DataFrame({"ts_code": ["000001.SZ"], "price": [10]}),
        QuoteMeta(quote_date="20240104", status="ok", is_stale=False)))
    _, response = _call("GET", "/market/state?date=20240104", history_runtime)
    assert response["data"]["available"] is False
    assert response["meta"]["status"] == "no_data"


def test_history_fallback_plan_cannot_be_executed(history_runtime, monkeypatch):
    monkeypatch.setattr(v1, "_trade_plan", lambda *args: SimpleNamespace(to_dict=lambda: {
        "date": "20240102", "available": True, "can_attack": True, "max_new_positions": 2,
        "available_exposure": .6, "plans": [{"position": {"shares": 100, "lots": 1, "amount": 1000}}]}))
    _, response = _call("GET", "/opportunities/plan?date=20240103", history_runtime)
    plan = response["data"]
    assert plan["historical_only"] and not plan["can_attack"] and plan["max_new_positions"] == 0
    assert plan["basis_date"] == "20240102" and plan["requested_date"] == "20240103"
    assert plan["plans"][0]["position"]["shares"] == 0


def test_invalid_prices_and_indexes_do_not_count_toward_coverage(history_runtime):
    provider = history_runtime.provider
    codes = provider.stock_basic()["ts_code"].tolist()
    provider.history.ingest(pd.DataFrame({"ts_code": codes, "trade_date": "20240105",
        "open": 10., "high": 9., "low": 8., "close": 10.}))
    provider.history.ingest(pd.DataFrame([{"ts_code": "000300.SH", "trade_date": "20240105",
        "open": 10., "high": 11., "low": 9., "close": 10.}]))
    history_runtime.analysis_cache.clear()
    status = next(item for item in history_dates(history_runtime) if item["date"] == "20240105")
    assert status["row_count"] == 1000 and status["valid_rows"] == 0 and not status["complete"]


def test_snapshot_board_uses_confirmed_structural_emotion(history_runtime, monkeypatch):
    from dataclasses import replace
    from tests.test_quote_service import _frame

    env = replace(v1._board_state(history_runtime, "20240102"), emotion_cycle="高潮")
    monkeypatch.setattr(v1, "_board_state", lambda *args: env)
    extras = v1._history_board_extras(history_runtime, "20240102")
    assert extras["emotion_cycle"] == "高潮"
    payload = v1._full_market_board("20240102", _frame(), history_runtime)
    assert payload["emotion_cycle"] == "高潮"
    assert payload["emotion_basis_date"] == "20240102"
    assert payload["emotion_source"] == "local_history"


def test_snapshot_without_structure_does_not_guess_emotion(history_runtime, monkeypatch):
    from tests.test_quote_service import _frame

    monkeypatch.setattr(v1, "_history_board_extras", lambda *args: None)
    payload = v1._full_market_board("20240102", _frame(), history_runtime)
    assert payload["emotion_cycle"] is None
    assert payload["emotion_basis_date"] is None
    assert payload["emotion_source"] == "unavailable"
