"""Cross-layer regressions for the V1.1 audit findings (offline, isolated state)."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from io import BytesIO
import json
from threading import Barrier, Event
from types import SimpleNamespace

import pandas as pd
import pytest

from backend.api import app, v1
from backend.api.memo import TTLCache
from backend.api.runtime import Runtime
from backend.ai.agent import AICopilot
from backend.backtest.ashare_engine import AShareBacktestEngine
from backend.backtest.metrics import calculate_metrics
from backend.data import auto_ingest
from backend.data.config import Settings, get_settings
from backend.data.providers.history import HistoryProvider
from backend.journal.trades import TradeJournal
from backend.opportunity.ranking import OpportunityRankingEngine
from backend.opportunity.setups import SetupResult
from backend.paper.engine import PaperTradingEngine
from backend.profile.engine import ProfileEngine
from backend.strategies.base import TradeSignal
from tests.test_api_v1 import _make_runtime, _call
from tests.test_auto_ingest import _Runtime, _HistoryProviderStub, _Session


def signal(action="BUY", day="20240102", price=10, quantity=100, signal_id=""):
    return TradeSignal(ts_code="000001.SZ", trade_date=day, action=action,
                       price=price, quantity=quantity, signal_id=signal_id)


@pytest.mark.parametrize("quantity", [0, -1, float("nan"), float("inf")])
def test_invalid_quantity_does_not_trade(quantity):
    paper = PaperTradingEngine(initial_cash=10000)
    with pytest.raises(ValueError):
        paper.execute_signal(signal(quantity=quantity))
    assert paper.cash == 10000 and not paper.positions and not paper.orders


@pytest.mark.parametrize("price", [0, -1, float("nan"), float("inf")])
def test_invalid_sell_price_cannot_corrupt_cash(price):
    paper = PaperTradingEngine(initial_cash=10000)
    paper.execute_signal(signal())
    cash = paper.cash
    with pytest.raises(ValueError):
        paper.execute_signal(signal("SELL", "20240103", price))
    assert paper.cash == cash and paper.positions["000001.SZ"].quantity == 100


def test_old_shares_can_sell_after_today_add_on_and_restart(tmp_path):
    path = tmp_path / "state.json"
    paper = PaperTradingEngine(state_path=path, t_plus_one=True)
    paper.execute_signal(signal())
    paper.execute_signal(signal(day="20240103"))
    paper = PaperTradingEngine(state_path=path, t_plus_one=True)
    assert paper.execute_signal(signal("SELL", "20240103")).status == "FILLED"
    assert paper.execute_signal(signal("SELL", "20240103")).status == "REJECTED"
    assert paper.positions["000001.SZ"].quantity == 100


def test_account_rejects_backdated_execution():
    paper = PaperTradingEngine(t_plus_one=True)
    paper.execute_signal(signal(day="20240103"))
    with pytest.raises(ValueError, match="日期"):
        paper.execute_signal(signal("SELL", "20240102"))


def test_state_failure_rolls_back_and_keeps_previous_disk_state(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    paper = PaperTradingEngine(state_path=path)
    paper.execute_signal(signal())
    previous = path.read_bytes()
    cash = paper.cash
    with monkeypatch.context() as patch:
        patch.setattr("backend.data.atomic.os.replace", lambda *args: (_ for _ in ()).throw(OSError("disk failure")))
        with pytest.raises(OSError):
            paper.execute_signal(signal(day="20240103"))
    assert paper.cash == cash and paper.positions["000001.SZ"].quantity == 100
    assert path.read_bytes() == previous
    assert not list(tmp_path.glob("*.tmp"))
    restored = PaperTradingEngine(state_path=path)
    assert restored.cash == cash and len(restored.orders) == 1


def test_corrupt_existing_account_does_not_reset_to_million(tmp_path):
    path = tmp_path / "state.json"
    path.write_text('{"cash":', encoding="utf-8")
    with pytest.raises(RuntimeError, match="恢复"):
        PaperTradingEngine(state_path=path)


def test_journal_projection_recovers_once_after_failure(tmp_path, monkeypatch):
    journal = TradeJournal(tmp_path)
    path = tmp_path / "state.json"
    paper = PaperTradingEngine(state_path=path, journal=journal)
    with monkeypatch.context() as patch:
        patch.setattr(journal, "record", lambda record: (_ for _ in ()).throw(OSError("journal disk")))
        assert paper.execute_signal(signal()).status == "FILLED"
        assert paper.journal_pending
    restored = PaperTradingEngine(state_path=path, journal=journal)
    assert not restored.journal_pending
    restored.sync_journal()
    assert len(journal.list()) == 1


def test_duplicate_order_is_idempotent_after_restart(tmp_path):
    path = tmp_path / "state.json"
    paper = PaperTradingEngine(state_path=path)
    request = signal(signal_id="manual-1")
    first = paper.execute_signal(request)
    paper = PaperTradingEngine(state_path=path)
    assert paper.execute_signal(request) == first
    assert paper.positions["000001.SZ"].quantity == 100 and len(paper.orders) == 1
    with pytest.raises(ValueError):
        paper.execute_signal(replace(request, quantity=200))


def test_concurrent_duplicate_order_fills_once():
    paper = PaperTradingEngine()
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: paper.execute_signal(signal(signal_id="same")), range(4)))
    assert len(paper.orders) == 1 and paper.positions["000001.SZ"].quantity == 100


def test_profile_counts_realized_sells_not_decision_events(tmp_path):
    paper = PaperTradingEngine(journal=TradeJournal(tmp_path))
    paper.execute_signal(signal())
    paper.execute_signal(signal("SELL", "20240103", 11))
    profile = ProfileEngine(tmp_path)
    stats = profile.refresh_stats_from_trades(profile.load_or_create()).stats
    assert stats.total_trades == 1 and stats.winning_trades == 1
    assert stats.total_pnl == pytest.approx(paper.snapshot().realized_pnl)
    assert stats.win_rate == 1 and not stats.sample_sufficient


def test_zero_cash_does_not_create_new_position_capacity():
    plan = OpportunityRankingEngine(object()).build("20240103",
        setups=SetupResult(date="20240103", available=True, stage="发酵"),
        state={"equity": 10000, "cash": 0})
    assert plan.current_exposure == 1 and plan.available_exposure == 0
    assert plan.max_new_positions == 0


def test_trade_win_rate_is_not_positive_day_ratio():
    curve = pd.DataFrame({"date": ["20240102", "20240103", "20240104"], "total": [100, 101, 102]})
    metrics = calculate_metrics(curve, trade_log=[SimpleNamespace(exit_price=9, pnl=-100, holding_days=1)])
    assert metrics["win_rate"] == 0 and metrics["positive_day_ratio"] == 1
    assert calculate_metrics(curve, trade_log=[])["win_rate"] is None


def test_historical_limits_unchanged_when_future_bars_added(tmp_path):
    pd.DataFrame([dict(ts_code="000001.SZ", symbol="000001", name="ST demo", is_st="1")]).to_csv(tmp_path / "stock_basic.csv", index=False)
    provider = HistoryProvider(base_dir=tmp_path)
    provider.history.ingest(pd.DataFrame([dict(ts_code="000001.SZ", trade_date="20240102", pre_close=10, pct_chg=5)]))
    before = provider.stk_limit(trade_date="20240102").iloc[0].up_limit
    provider.history.ingest(pd.DataFrame([dict(ts_code="000001.SZ", trade_date="20240103", pre_close=10.5, pct_chg=10)]))
    provider.clear_cache()
    assert before == 10.5 == provider.stk_limit(trade_date="20240102").iloc[0].up_limit


def test_failed_ingest_retries_after_backoff(monkeypatch):
    auto_ingest.reset_state()
    runtime = _Runtime(_HistoryProviderStub())
    clock = [1000.0]
    monkeypatch.setattr(auto_ingest.time, "monotonic", lambda: clock[0])
    attempts = []
    def fail(day):
        attempts.append(day)
        raise RuntimeError("source failed")
    args = dict(session=_Session("盘后"), rows_on=lambda day: 0, backfill=fail, run_sync=True)
    assert auto_ingest.ensure_today_bars(runtime, **args)
    assert not auto_ingest.ensure_today_bars(runtime, **args)
    assert "20260918" not in auto_ingest._state["done"]
    clock[0] += 301
    assert auto_ingest.ensure_today_bars(runtime, **args) and len(attempts) == 2
    auto_ingest.reset_state()


def test_partial_market_ingest_is_not_complete(monkeypatch):
    auto_ingest.reset_state()
    provider = _HistoryProviderStub()
    provider.stock_basic = lambda: pd.DataFrame({"ts_code": range(5000)})
    attempts = []
    assert auto_ingest.ensure_today_bars(_Runtime(provider), session=_Session("盘后"),
        rows_on=lambda day: 200, backfill=lambda day: attempts.append(day) or {"failed": 2}, run_sync=True)
    assert attempts and "20260918" not in auto_ingest._state["done"]
    auto_ingest.reset_state()


def test_provider_cache_is_invalidated_after_ingest():
    provider = _HistoryProviderStub()
    cleared = []
    provider.clear_cache = lambda: cleared.append(True)
    auto_ingest._invalidate(_Runtime(provider))
    assert cleared


def test_production_chain_does_not_contain_fixture(tmp_path):
    settings = Settings(None, tmp_path / "market.db", get_settings().fixture_dir)
    provider = Runtime._choose_provider(settings, tmp_path)
    assert all(p.name != "fixture" for p in provider.providers)
    assert Runtime._choose_provider(replace(settings, demo_mode=True), tmp_path).name == "fixture"


@pytest.mark.parametrize("data", [[], {"quantity": 0}, {"quantity": -100}, {"price": -1}, {"ts_code": "000300"}, {"action": "HACK"}])
def test_bad_manual_orders_return_400_without_state_change(tmp_path, data):
    runtime = _make_runtime(tmp_path)
    payload = dict(ts_code="000001.SZ", action="BUY", price=10, quantity=100)
    if isinstance(data, dict):
        payload.update(data)
    else:
        payload = data
    code, result = _call("POST", "/paper/trade", runtime, payload)
    assert code == 400 and not runtime.paper.orders and not runtime.journal.list()


def test_unknown_strategy_does_not_run_baseline(tmp_path):
    runtime = _make_runtime(tmp_path)
    code, _ = _call("POST", "/backtest/runs", runtime, {"strategy": "missing"})
    assert code == 400 and not runtime.backtest_runs


def test_manual_limit_up_is_rejected(tmp_path):
    runtime = _make_runtime(tmp_path)
    upper = runtime.provider.stk_limit(ts_codes=["000001.SZ"], trade_date="20240103").iloc[0].up_limit
    code, result = _call("POST", "/paper/trade", runtime, dict(ts_code="000001.SZ", price=upper, quantity=100))
    assert code == 200 and result["data"]["order"]["status"] == "REJECTED"
    assert not runtime.paper.orders


def test_trade_plan_recomputes_after_account_changes(tmp_path, monkeypatch):
    runtime = _make_runtime(tmp_path)
    monkeypatch.setattr(v1, "_typed_opportunities", lambda *args: object())
    monkeypatch.setattr(v1.OpportunityRankingEngine, "build", lambda self, *args, **kwargs: kwargs["state"])
    first = v1._trade_plan(runtime, "20240103")
    runtime.paper.execute_signal(signal())
    second = v1._trade_plan(runtime, "20240103")
    assert first["cash"] > second["cash"]


def test_account_and_ai_use_same_real_positions(tmp_path):
    runtime = _make_runtime(tmp_path)
    runtime.paper.execute_signal(signal())
    snapshot = runtime.account_snapshot("20240103")
    assert snapshot.equity == pytest.approx(runtime.paper.cash + 100 * 11)
    copilot = AICopilot(provider=runtime.provider, runtime=runtime)
    positions = copilot.get_positions("20240103").data
    assert positions["cash"] == snapshot.cash
    assert positions["positions"]["000001.SZ"]["quantity"] == 100
    assert positions["equity"] == snapshot.equity


def test_runtime_initialization_is_once_even_under_concurrency(monkeypatch):
    barrier = Barrier(2)
    created = []
    def create():
        value = object()
        created.append(value)
        return value
    monkeypatch.setattr(app, "_RUNTIME", None)
    monkeypatch.setattr(app.Runtime, "create", create)
    def get(_):
        barrier.wait(timeout=3)
        return app._get_runtime()
    with ThreadPoolExecutor(max_workers=2) as pool:
        values = list(pool.map(get, range(2)))
    assert values[0] is values[1] and len(created) == 1


def test_static_page_and_health_do_not_initialize_runtime(monkeypatch):
    monkeypatch.setattr(app, "_get_runtime", lambda: (_ for _ in ()).throw(RuntimeError("vendor offline")))
    for path in ("/", "/health"):
        responses = []
        output = app.application({"PATH_INFO": path, "REQUEST_METHOD": "GET"}, lambda status, headers: responses.append(status))
        assert responses == ["200 OK"] and output


def test_single_flight_cache_and_clear_during_compute():
    cache = TTLCache(60)
    entered, release = Event(), Event()
    calls = []
    def compute():
        calls.append(True)
        entered.set()
        assert release.wait(3)
        return 1
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(cache.get_or_compute, "key", compute)
        assert entered.wait(3)
        second = pool.submit(cache.get_or_compute, "key", compute)
        release.set()
        assert first.result() == second.result() == 1
    assert len(calls) == 1
    cache.clear()
    entered.clear()
    release.clear()
    with ThreadPoolExecutor(max_workers=1) as pool:
        old = pool.submit(cache.get_or_compute, "key", compute)
        assert entered.wait(3)
        cache.clear()
        release.set()
        assert old.result() == 1
    assert cache.get("key") is None


class ResearchProvider:
    name = "research-test"
    def __init__(self, prices):
        self.prices = prices
    def daily(self, **kwargs):
        return pd.DataFrame([dict(ts_code="000001.SZ", trade_date=f"2024010{i + 2}",
            open=price, high=price, low=price, close=price, pre_close=10, vol=100, amount=1000)
            for i, price in enumerate(self.prices)])
    def stock_basic(self, **kwargs):
        return pd.DataFrame({"ts_code": ["000001.SZ"]})
    def stk_limit(self, trade_date, **kwargs):
        return pd.DataFrame([dict(ts_code="000001.SZ", trade_date=trade_date, up_limit=11, down_limit=9)])
    def suspend_d(self, **kwargs):
        return pd.DataFrame()


def test_stop_loss_waits_for_executable_next_open_and_costs_reconcile():
    provider = ResearchProvider([10, 10, 9, 9, 9.5])
    strategy = SimpleNamespace(generate_signals=lambda frame: [signal()])
    engine = AShareBacktestEngine(provider, strategy, initial_cash=100000)
    result = engine.run()
    assert len(result.trade_log) == 1
    assert result.trade_log[0].exit_date == "20240106"
    assert result.trade_log[0].exit_price == 9.5
    assert sum(t.pnl for t in result.trade_log) == pytest.approx(result.final_total_value - 100000)


def test_index_only_data_cannot_fall_back_to_tradable(tmp_path):
    provider = ResearchProvider([10, 10])
    provider.stock_basic = lambda **kwargs: pd.DataFrame({"ts_code": ["600000.SH"]})
    with pytest.raises(ValueError, match="tradable"):
        AShareBacktestEngine(provider, SimpleNamespace(generate_signals=lambda frame: [signal()])).run()


def test_backtest_add_on_costs_reconcile():
    provider = ResearchProvider([10, 10, 9.5, 10])
    strategy = SimpleNamespace(generate_signals=lambda frame: [
        signal(quantity=None), signal(day="20240103", quantity=None),
        signal("SELL", "20240104", quantity=None)])
    result = AShareBacktestEngine(provider, strategy, initial_cash=100000,
        stop_loss_pct=0, take_profit_pct=0).run()
    assert len(result.orders) == 3 and len(result.trade_log) == 1
    assert result.positions["000001.SZ"].quantity == 0
    assert result.trade_log[0].pnl == pytest.approx(result.final_total_value - 100000)


def test_legacy_account_migrates_to_sellable_lots(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"cash": 9000, "initial_cash": 10000,
        "positions": {"000001.SZ": {"quantity": 100, "avg_price": 10,
        "buy_date": "20240102"}}, "orders": []}), encoding="utf-8")
    paper = PaperTradingEngine(state_path=path, t_plus_one=True)
    assert paper.execute_signal(signal("SELL", "20240103")).status == "FILLED"


def test_pending_ingest_survives_restart_even_if_coverage_threshold_met(tmp_path):
    provider = HistoryProvider(base_dir=tmp_path)
    provider.stock_basic = lambda: pd.DataFrame({"ts_code": range(5000)})
    provider.history.set_state("auto_ingest_pending", '{"20260918": {"failed": 2}}')
    auto_ingest.reset_state()
    calls = []
    assert auto_ingest.ensure_today_bars(_Runtime(provider), session=_Session("盘后"),
        rows_on=lambda day: 5000, backfill=lambda day: calls.append(day) or {"failed": 0}, run_sync=True)
    assert calls == ["20260918"]
    assert json.loads(provider.history.get_state("auto_ingest_pending")) == {}
    auto_ingest.reset_state()


def test_intraday_can_backfill_previous_confirmed_days():
    from tests.test_auto_ingest import _CalendarSession
    auto_ingest.reset_state()
    calls = []
    assert auto_ingest.ensure_today_bars(_Runtime(_HistoryProviderStub()),
        session=_CalendarSession(["20260917", "20260918"], phase="盘中"),
        rows_on=lambda day: 0, backfill=lambda day: calls.append(day) or {}, run_sync=True)
    assert calls == ["20260917"]
    auto_ingest.reset_state()


def test_real_quote_date_survives_cache_restart(tmp_path):
    from backend.market.quote_service import QuoteService
    from tests.test_quote_service import _frame
    service = QuoteService(db_path=tmp_path / "quotes.db", fetcher=lambda: (
        _frame(), {"source": "tencent", "quote_time": "15:00:00", "quote_date": "20240103"}))
    service.refresh()
    restored = QuoteService(db_path=tmp_path / "quotes.db")
    _, meta = restored.get()
    assert meta.quote_date == "20240103"
    assert v1._snapshot_matches(_frame(), meta, "20240103")
    assert not v1._snapshot_matches(_frame(), meta, "20240104")


def test_snapshot_errors_do_not_dump_transport_details():
    from backend.market.quote_service import QuoteMeta
    payload = v1._snapshot_meta(QuoteMeta(error="HTTPConnectionPool(url=huge internal request)"))
    assert "HTTPConnectionPool" not in payload["error"]


def test_json_array_and_non_finite_numbers_return_400(tmp_path):
    runtime = _make_runtime(tmp_path)
    for raw in (b'[]', b'{"price": NaN}', b'{broken'):
        env = {"PATH_INFO": "/paper/trade", "REQUEST_METHOD": "POST",
               "wsgi.input": BytesIO(raw), "CONTENT_LENGTH": str(len(raw))}
        responses = []
        payload, handled = v1.dispatch(env, lambda status, headers: responses.append(status), runtime)
        assert handled and responses == ["400 Bad Request"]
        assert not runtime.paper.orders


def test_decision_record_preserves_trade_date_and_numeric_results(tmp_path):
    runtime = _make_runtime(tmp_path)
    code, _ = _call("POST", "/trades", runtime, dict(ts_code="000001.SZ", action="SELL",
        trade_date="20240103", result={"pnl": "100", "return_pct": "0.1"}))
    assert code == 201
    record = runtime.journal.list()[0]
    assert record.trade_date == "20240103" and record.result["pnl"] == 100
    _, profile = _call("GET", "/profile", runtime)
    assert profile["data"]["stats"]["total_pnl"] == 100


def test_ai_cannot_recommend_current_attack_using_historical_emotion():
    copilot = AICopilot()
    assert not copilot._can_trade_today({"date": "20240103", "market": {"state": "偏强", "score": 80},
        "emotion": {"date": "20240102", "emotion_cycle": "发酵"}})
    assert not copilot._can_trade_today({"date": "20240103", "market": {"state": "偏强", "score": 80},
        "emotion": {"date": "20240103", "emotion_cycle": "发酵"},
        "trade_plan": {"date": "20240103", "can_attack": False}})
