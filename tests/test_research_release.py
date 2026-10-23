"""V1.1 execution, research accounting, and reproducibility regressions."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from backend.backtest.ashare_engine import AShareBacktestEngine
from backend.backtest.metrics import calculate_metrics
from backend.data.providers.fixture import FixtureProvider
from backend.research.release_report import sha256, verify_snapshot
from backend.strategies.base import Strategy, TradeSignal


class ScheduledStrategy(Strategy):
    def __init__(self, signals=()):
        self.signals = list(signals)

    def generate_signals(self, daily):
        # Deliberately independent of bars so executor guarantees are tested.
        return self.signals


def provider(tmp_path, *, volumes=None, changes=None):
    dates = pd.bdate_range("2024-01-01", periods=8).strftime("%Y%m%d").tolist()
    prices = changes or [10.] * 8
    data = pd.DataFrame({"trade_date": dates, "ts_code": ["000001.SZ"] * 8,
        "open": prices, "high": [p + 0.2 for p in prices], "low": [p - 0.2 for p in prices],
        "close": prices, "pre_close": [10.] + prices[:-1], "change": [0.] * 8,
        "pct_chg": [0.] * 8, "vol": volumes or [10000.] * 8, "amount": [100000.] * 8})
    data.to_csv(tmp_path / "daily.csv", index=False)
    return FixtureProvider(base_dir=tmp_path), dates


@pytest.mark.parametrize("key", ["train_days", "test_days", "step_days"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_walk_forward_rejects_nonpositive_or_fractional_windows(tmp_path, key, value):
    source, dates = provider(tmp_path)
    with pytest.raises(ValueError, match=key):
        AShareBacktestEngine(source, ScheduledStrategy()).walk_forward(
            start_date=dates[0], end_date=dates[-1], **{key: value})


@pytest.mark.parametrize("volume", [0., -1., float("nan")])
def test_price_only_halt_bars_do_not_fill(tmp_path, volume):
    source, dates = provider(tmp_path, volumes=[volume] * 8)
    strategy = ScheduledStrategy([TradeSignal("000001.SZ", dates[0], "BUY")])
    result = AShareBacktestEngine(source, strategy).run()
    assert result.orders == []
    assert result.final_total_value == 1_000_000


def test_test_fold_warms_on_train_bars_but_starts_in_cash(tmp_path):
    source, dates = provider(tmp_path)
    strategy = ScheduledStrategy([TradeSignal("000001.SZ", dates[1], "BUY")])
    engine = AShareBacktestEngine(source, strategy, commission_rate=0.01,
                                  stamp_duty_rate=0, stop_loss_pct=0, take_profit_pct=0)
    result = engine.walk_forward(start_date=dates[0], end_date=dates[3],
                                  train_days=2, test_days=2, step_days=2)
    fold = result.folds[0]
    assert fold.train_start == dates[0] and fold.train_end == dates[1]
    assert fold.train_metrics["total_return"] == 0  # signal can't fill until test starts
    assert fold.metrics["total_return"] < 0  # test first-day commission is counted
    assert result.in_sample["fold_count"] == result.out_of_sample["fold_count"] == 1


def test_first_test_day_cost_included_in_return_and_drawdown(tmp_path):
    source, dates = provider(tmp_path)
    strategy = ScheduledStrategy([TradeSignal("000001.SZ", dates[0], "BUY")])
    result = AShareBacktestEngine(source, strategy, commission_rate=0.01).run(
        start_date=dates[1], end_date=dates[1], warmup_start_date=dates[0])
    metrics = calculate_metrics(result.equity_curve)
    expected = result.final_total_value / 1_000_000 - 1
    assert metrics["total_return"] == pytest.approx(expected)
    assert metrics["max_drawdown"] == pytest.approx(expected)
    assert result.drawdown_curve.drawdown.iloc[0] == pytest.approx(expected)
    assert expected < 0


def test_independent_fold_win_rate_pools_closed_trades(tmp_path, monkeypatch):
    source, dates = provider(tmp_path)
    engine = AShareBacktestEngine(source, ScheduledStrategy())
    calls = []

    def fake_run(**kwargs):
        calls.append(kwargs)
        test = "warmup_start_date" in kwargs
        # Test folds have 1 winning trade then 3 losing trades, not 50% wins.
        pnls = [1.] if len(calls) == 2 else [-1., -1., -1.]
        curve = pd.DataFrame({"date": ["20240101", "20240102"], "total": [100., 100.]})
        return SimpleNamespace(equity_curve=curve, trade_log=[SimpleNamespace(
            exit_price=10, pnl=pnl, holding_days=1) for pnl in pnls] if test else [])

    monkeypatch.setattr(engine, "run", fake_run)
    result = engine.walk_forward(start_date=dates[0], end_date=dates[5],
                                  train_days=2, test_days=2, step_days=2)
    assert result.out_of_sample["win_rate"] == 0.25
    assert result.out_of_sample["trade_count"] == 4
    assert result.in_sample["fold_count"] == 2
    assert calls[1]["warmup_start_date"] == dates[0]
    assert calls[3]["warmup_start_date"] == dates[2]


@pytest.mark.parametrize("prices", [[-1.] * 8, [float("nan")] * 8])
def test_invalid_prices_abort_instead_of_producing_metrics(tmp_path, prices):
    source, _ = provider(tmp_path, changes=prices)
    with pytest.raises(ValueError, match="无效价格"):
        AShareBacktestEngine(source, ScheduledStrategy()).run()


def test_duplicate_dates_abort_backtest(tmp_path):
    source, _ = provider(tmp_path)
    path = tmp_path / "daily.csv"
    frame = pd.read_csv(path)
    pd.concat([frame, frame.iloc[:1]]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="重复"):
        AShareBacktestEngine(source, ScheduledStrategy()).run()


def test_invalid_ohlc_aborts_backtest(tmp_path):
    source, _ = provider(tmp_path)
    path = tmp_path / "daily.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "high"] = 9
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="OHLC"):
        AShareBacktestEngine(source, ScheduledStrategy()).run()


def test_null_only_stock_list_cannot_restore_unverified_universe(tmp_path, monkeypatch):
    source, _ = provider(tmp_path)
    monkeypatch.setattr(source, "stock_basic", lambda: pd.DataFrame({"ts_code": [pd.NA]}))
    with pytest.raises(ValueError, match="不含有效代码"):
        AShareBacktestEngine(source, ScheduledStrategy()).run()


def test_single_observation_preserves_closed_trade_statistics():
    metrics = calculate_metrics(pd.DataFrame({"date": ["20240101"], "total": [100.]}),
        trade_log=[SimpleNamespace(exit_price=10, pnl=1., holding_days=1)])
    assert metrics["win_rate"] == 1
    assert metrics["trade_count"] == 1
    assert metrics["positive_day_ratio"] is None


def test_frozen_input_hash_detects_edits_and_path_traversal(tmp_path):
    sample = tmp_path / "sample.csv"
    sample.write_text("original", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"files": {"sample.csv": sha256(sample)}}))
    verify_snapshot(tmp_path)
    sample.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="校验失败"):
        verify_snapshot(tmp_path)
    outside = tmp_path.parent / "outside.csv"
    outside.write_text("outside", encoding="utf-8")
    manifest.write_text(json.dumps({"files": {"../outside.csv": sha256(outside)}}))
    with pytest.raises(ValueError, match="校验失败"):
        verify_snapshot(tmp_path)
