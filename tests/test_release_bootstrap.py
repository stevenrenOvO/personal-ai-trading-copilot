from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from backend.api.data_status import history_dates
from backend.api.demo import configure_demo_services
from backend.api.runtime import Runtime
from backend.data.config import clear_settings_cache, get_settings
from backend.data.history import MarketHistory
from backend.data.providers.fixture import FixtureProvider
from backend.data.providers.history import HistoryProvider
from tests.test_api_v1 import _call


def test_explicit_demo_has_data_and_never_fetches_network(monkeypatch):
    monkeypatch.setattr("requests.sessions.Session.request", lambda *a, **kw: pytest.fail("offline demo attempted network"))
    configure_demo_services()
    runtime = Runtime.create()
    assert runtime.provider.name == "fixture"
    status, latest = _call("GET", "/data/latest", runtime)
    assert status == 200 and latest["data"]["date"] == runtime.provider.latest_trade_date()
    status, market = _call("GET", "/market/state", runtime)
    assert status == 200 and market["data"]["total_count"] > 0
    assert market["meta"]["is_fixture"]
    assert get_settings().db_path.parent == runtime.root


def test_demo_services_require_explicit_opt_in(monkeypatch):
    monkeypatch.setenv("COPILOT_DEMO_MODE", "0")
    clear_settings_cache()
    with pytest.raises(ValueError, match="explicitly"):
        configure_demo_services()


def test_new_full_catalog_selects_empty_history_for_background_ingestion(tmp_path):
    codes = [f"{i+1:06}.SZ" for i in range(1000)]
    pd.DataFrame({"ts_code": codes, "symbol": codes, "name": codes}).to_csv(tmp_path / "stock_basic.csv", index=False)
    MarketHistory(tmp_path / "history" / "market_history.db")
    assert Runtime._history_provider(tmp_path).name == "history"
    # A small basket must not become a claimed full market on a fresh install.
    pd.DataFrame({"ts_code": ["000001.SZ"], "symbol": ["000001"], "name": ["sample"]}).to_csv(tmp_path / "stock_basic.csv", index=False)
    assert Runtime._history_provider(tmp_path) is None


def test_metadata_only_initializes_without_requesting_daily_bars(tmp_path, monkeypatch):
    from backend.data import ingest
    fixture = FixtureProvider(get_settings().fixture_dir)
    monkeypatch.setattr(ingest, "BaoStockProvider", lambda: fixture)
    monkeypatch.setattr(ingest, "_universe_codes", lambda *args: ["000001.SZ"])
    monkeypatch.setattr(fixture, "daily_raw", lambda **kwargs: pytest.fail("metadata initialization fetched bars"), raising=False)
    result = ingest.ingest(universe="all", start_date="20240101", end_date="20240103", root=tmp_path, metadata_only=True)
    assert result["metadata_only"] and (tmp_path / "stock_basic.csv").exists()
    assert MarketHistory(tmp_path / "history" / "market_history.db").coverage()["rows"] == 0


def test_unchanged_history_reuses_scan_and_commit_refreshes_it(tmp_path, monkeypatch):
    pd.DataFrame({"ts_code": ["000001.SZ"], "symbol": ["000001"], "name": ["sample"]}).to_csv(tmp_path / "stock_basic.csv", index=False)
    provider = HistoryProvider(base_dir=tmp_path)
    runtime = Runtime.create(root=tmp_path / "runtime")
    runtime.provider = provider
    original = provider.history._connect
    scans = []
    @contextmanager
    def count():
        scans.append(True)
        with original() as conn:
            yield conn
    monkeypatch.setattr(provider.history, "_connect", count)
    assert history_dates(runtime) == []
    first = len(scans)
    assert history_dates(runtime) == [] and len(scans) == first
    provider.history.ingest(pd.DataFrame([{"ts_code": "000001.SZ", "trade_date": "20240102",
        "open": 10, "high": 11, "low": 9, "close": 10}]))
    assert history_dates(runtime)[0]["row_count"] == 1


def test_date_picker_never_requests_unbounded_remote_history(tmp_path):
    runtime = Runtime.create(root=tmp_path)
    def fail(*args, **kwargs):
        pytest.fail("date picker must not query remote universe")
    runtime.provider = SimpleNamespace(name="baostock+akshare", daily=fail, stock_basic=fail)
    assert history_dates(runtime) == []
