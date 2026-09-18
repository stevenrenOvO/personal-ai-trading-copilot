"""Tests for the risk checking layer."""

from backend.risk import (
    MaxDrawdownRule,
    MaxPositionWeightRule,
    PositionInfo,
    RiskContext,
    RiskManager,
    StopLossRule,
)
from backend.strategies.base import TradeSignal


def make_signal(action: str = "BUY") -> TradeSignal:
    return TradeSignal(
        ts_code="000001.SZ",
        trade_date="20240103",
        action=action,
        price=10.0,
    )


def make_context(**overrides) -> RiskContext:
    values = {
        "equity": 100_000.0,
        "cash": 100_000.0,
        "peak_equity": 100_000.0,
    }
    values.update(overrides)
    return RiskContext(**values)


def test_max_position_weight_blocks_overweight_buy():
    signal = make_signal("BUY")
    context = make_context(
        positions={"000001.SZ": PositionInfo("000001.SZ", 8_000.0, 10.0)},
        prices={"000001.SZ": 10.0},
    )
    manager = RiskManager([MaxPositionWeightRule(max_weight=0.2)])

    result = manager.check_signal(signal, context)

    assert not result.allowed
    assert result.violations[0].rule == "max_position_weight"
    assert result.violations[0].ts_code == "000001.SZ"


def test_max_position_weight_allows_sell():
    signal = make_signal("SELL")
    context = make_context(
        positions={"000001.SZ": PositionInfo("000001.SZ", 8_000.0, 10.0)},
        prices={"000001.SZ": 10.0},
    )
    manager = RiskManager([MaxPositionWeightRule(max_weight=0.2)])

    result = manager.check_signal(signal, context)

    assert result.allowed
    assert result.violations == ()


def test_max_drawdown_blocks_new_buy():
    signal = make_signal("BUY")
    context = make_context(equity=79_000.0, peak_equity=100_000.0)
    manager = RiskManager([MaxDrawdownRule(max_drawdown=0.20)])

    result = manager.check_signal(signal, context)

    assert not result.allowed
    assert result.violations[0].rule == "max_drawdown"


def test_stop_loss_flags_losing_position():
    context = make_context(
        positions={"000001.SZ": PositionInfo("000001.SZ", 1_000.0, 10.0)},
        prices={"000001.SZ": 9.0},
    )
    manager = RiskManager([StopLossRule(stop_loss_pct=0.08)])

    result = manager.check_portfolio(context)

    assert not result.allowed
    assert result.violations[0].rule == "stop_loss"
    assert result.violations[0].ts_code == "000001.SZ"


def test_risk_manager_allows_healthy_buy():
    signal = make_signal("BUY")
    context = make_context(
        positions={"000001.SZ": PositionInfo("000001.SZ", 1_000.0, 10.0)},
        prices={"000001.SZ": 10.0},
    )
    manager = RiskManager()

    result = manager.check_signal(signal, context)

    assert result.allowed
    assert result.violations == ()
