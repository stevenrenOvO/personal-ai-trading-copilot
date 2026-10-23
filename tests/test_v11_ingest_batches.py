import pandas as pd

from backend.data import auto_ingest, ingest_full
from backend.data.history import MarketHistory
from backend.data.providers.history import HistoryProvider
from tests.test_auto_ingest import _CalendarSession, _Runtime


def test_range_backfill_repairs_an_older_gap_even_when_end_exists(tmp_path, monkeypatch):
    path = tmp_path / "history.db"
    history = MarketHistory(path)
    history.ingest(pd.DataFrame([{"ts_code": "000001.SZ", "trade_date": "20260918", "close": 10}]))
    calls = []
    def fetch(code, start, end):
        calls.append((code, start, end))
        return pd.DataFrame([{"ts_code": code, "trade_date": "20260917", "close": 10}])
    monkeypatch.setattr("backend.data.providers.tencent.fetch_daily", fetch)
    result = ingest_full.run_backfill(start_date="20260917", end_date="20260918",
        codes="000001.SZ", db_path=path, required_dates=["20260917", "20260918"])
    assert calls == [("000001.SZ", "20260917", "20260918")]
    assert result["failures_by_date"] == {"20260917": {"failed": 0, "failed_codes": []},
                                         "20260918": {"failed": 0, "failed_codes": []}}
    calls.clear()
    ingest_full.run_backfill(start_date="20260917", end_date="20260918",
        codes="000001.SZ", db_path=path, required_dates=["20260917", "20260918"])
    assert not calls


def test_partial_range_is_not_reported_as_complete(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.data.providers.tencent.fetch_daily", lambda code, start, end:
        pd.DataFrame([{"ts_code": code, "trade_date": end, "close": 10}]))
    result = ingest_full.run_backfill(start_date="20260917", end_date="20260918",
        codes="000001.SZ", db_path=tmp_path / "h.db", required_dates=["20260917", "20260918"])
    assert result["failures_by_date"]["20260917"]["failed"] == 1
    assert result["failures_by_date"]["20260918"]["failed"] == 0
    assert result["failed"] == 1 and result["done"] == 0


def test_automatic_missing_days_share_a_single_download_window(tmp_path, monkeypatch):
    auto_ingest.reset_state()
    provider = HistoryProvider(base_dir=tmp_path)
    monkeypatch.setattr(provider, "stock_basic", lambda: pd.DataFrame({"ts_code": ["000001.SZ"]}))
    calls = []
    def fill(**kwargs):
        calls.append(kwargs)
        return {"failures_by_date": {day: {"failed": 0} for day in kwargs["required_dates"]}}
    monkeypatch.setattr(ingest_full, "run_backfill", fill)
    try:
        assert auto_ingest.ensure_today_bars(_Runtime(provider),
            session=_CalendarSession(["20260916", "20260917", "20260918"]),
            rows_on=lambda day: 0, run_sync=True)
        assert len(calls) == 1
        assert calls[0]["required_dates"] == ["20260916", "20260917", "20260918"]
        assert calls[0]["db_path"] == provider.history.db_path
    finally:
        auto_ingest.reset_state()
