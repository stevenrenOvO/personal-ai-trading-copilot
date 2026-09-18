"""Tests for the offline paper trading engine."""

from __future__ import annotations

import pytest

from backend.paper import PaperTradingEngine
from backend.strategies.base import TradeSignal


def _signal(
    *,
    ts_code: str = "000001.SZ",
    action: str = "BUY",
    price: float = 10.0,
    trade_date: str = "20240103",
) -> TradeSignal:
    return TradeSignal(
        ts_code=ts_code,
        trade_date=trade_date,
        action=action,
        price=price,
        reason="test",
    )


def test_paper_engine_starts_with_cash():
    engine = PaperTradingEngine(initial_cash=100_000.0)
    assert engine.cash == 100_000.0
    assert engine.positions == {}
    assert engine.snapshot({"000001.SZ": 10.0}).equity == 100_000.0


def test_paper_buy_rounds_to_lot():
    # cash = 1000, price=10, lot=100 -> max 100 shares
    engine = PaperTradingEngine(
        initial_cash=1000.0,
        lot_size=100,
        commission_rate=0.0,
    )
    order = engine.execute_signal(_signal(price=10.0))
    assert order.status == "FILLED"
    assert order.quantity == 100
    assert order.gross_value == 1000.0
    assert engine.cash == 0.0
    assert engine.positions["000001.SZ"].quantity == 100
    assert engine.positions["000001.SZ"].avg_price == 10.0


def test_paper_buy_honours_requested_quantity():
    """A manual simulated order must be able to be smaller than a full position."""
    engine = PaperTradingEngine(initial_cash=1_000_000.0, commission_rate=0.0)
    order = engine.execute_signal(
        TradeSignal(
            ts_code="000001.SZ",
            trade_date="20240103",
            action="BUY",
            price=10.0,
            quantity=500,
            reason="manual size",
        )
    )
    assert order.status == "FILLED"
    assert order.quantity == 500
    assert engine.cash == 1_000_000.0 - 5000.0


def test_paper_buy_rounds_requested_quantity_and_caps_it():
    engine = PaperTradingEngine(initial_cash=100_000.0, commission_rate=0.0)
    partial = engine.execute_signal(
        TradeSignal(
            ts_code="600000.SH",
            trade_date="20240103",
            action="BUY",
            price=10.0,
            quantity=550,  # not a whole lot
        )
    )
    assert partial.quantity == 500

    too_big = engine.execute_signal(
        TradeSignal(
            ts_code="600000.SH",
            trade_date="20240103",
            action="BUY",
            price=10.0,
            quantity=10_000_000,  # more than the account can pay for
        )
    )
    assert too_big.quantity <= 9_500
    assert engine.cash >= 0.0


def test_paper_buy_without_quantity_still_sizes_by_cash():
    engine = PaperTradingEngine(initial_cash=100_000.0, commission_rate=0.0)
    order = engine.execute_signal(_signal(price=10.0))
    assert order.quantity == 10_000


def test_paper_partial_sell_honours_the_requested_quantity():
    """Selling a part of the position must not silently liquidate everything."""
    engine = PaperTradingEngine(initial_cash=100_000.0, commission_rate=0.0)
    engine.execute_signal(
        TradeSignal(ts_code="000001.SZ", trade_date="20240103", action="BUY",
                    price=10.0, quantity=2000)
    )
    order = engine.execute_signal(
        TradeSignal(ts_code="000001.SZ", trade_date="20240104", action="SELL",
                    price=11.0, quantity=500)
    )
    assert order.status == "FILLED" and order.quantity == 500
    assert engine.positions["000001.SZ"].quantity == 1500


def test_paper_partial_sell_of_odd_lot_is_rejected_not_resized():
    engine = PaperTradingEngine(initial_cash=100_000.0, commission_rate=0.0)
    engine.execute_signal(
        TradeSignal(ts_code="000001.SZ", trade_date="20240103", action="BUY",
                    price=10.0, quantity=2000)
    )
    order = engine.execute_signal(
        TradeSignal(ts_code="000001.SZ", trade_date="20240104", action="SELL",
                    price=11.0, quantity=50)
    )
    assert order.status == "REJECTED"
    assert engine.positions["000001.SZ"].quantity == 2000


def test_paper_sell_more_than_held_closes_without_going_negative():
    engine = PaperTradingEngine(initial_cash=100_000.0, commission_rate=0.0)
    engine.execute_signal(
        TradeSignal(ts_code="000001.SZ", trade_date="20240103", action="BUY",
                    price=10.0, quantity=1000)
    )
    order = engine.execute_signal(
        TradeSignal(ts_code="000001.SZ", trade_date="20240104", action="SELL",
                    price=11.0, quantity=99999)
    )
    assert order.status == "FILLED" and order.quantity == 1000
    assert engine.positions["000001.SZ"].quantity == 0
    assert engine.cash > 0


def test_paper_sell_without_position_is_rejected():
    engine = PaperTradingEngine(initial_cash=100_000.0)
    order = engine.execute_signal(
        TradeSignal(ts_code="600000.SH", trade_date="20240104", action="SELL",
                    price=11.0, quantity=100)
    )
    assert order.status == "REJECTED"
    assert engine.positions.get("600000.SH") is None


def test_paper_buy_applies_commission():
    engine = PaperTradingEngine(
        initial_cash=1000.0,
        lot_size=100,
        commission_rate=0.001,
    )
    order = engine.execute_signal(_signal(price=10.0))
    # buy 100 shares at 10 => gross 1000, commission 1, total cost 1001 > cash
    # so it should reduce to 0 shares and reject
    # Actually our logic: per_share_cost = 10 * 1.001 = 10.01, affordable = 1000//10.01 = 99, floor to lot -> 0, so reject
    # But we want to test commission applied on a filled order, so we need more cash
    # Let's set cash=10000, price=10, lot=100 -> affordable = 10000//10.01 = 999, floor to 900 shares, commission = 900*10*0.001=9
    # So we adjust
    engine2 = PaperTradingEngine(
        initial_cash=10000.0,
        lot_size=100,
        commission_rate=0.001,
    )
    order2 = engine2.execute_signal(_signal(price=10.0))
    assert order2.status == "FILLED"
    assert order2.quantity == 900
    assert order2.commission == 9.0
    assert engine2.cash == 10000 - (900*10 + 9) == 10000 - 9009 == 991.0


def test_paper_buy_rejects_when_cash_insufficient_for_lot():
    engine = PaperTradingEngine(
        initial_cash=99.0,
        lot_size=100,
        commission_rate=0.0,
    )
    order = engine.execute_signal(_signal(price=10.0))
    assert order.status == "REJECTED"
    assert order.quantity == 0
    assert engine.cash == 99.0
    assert engine.positions == {}


def test_paper_sell_reduces_position_and_updates_cash():
    engine = PaperTradingEngine(
        initial_cash=1000.0,
        lot_size=100,
        commission_rate=0.0,
    )
    engine.execute_signal(_signal(price=10.0))  # buys 100 shares, cash 0
    before_cash = engine.cash
    sell = engine.execute_signal(_signal(action="SELL", price=11.0))
    assert sell.status == "FILLED"
    assert sell.quantity == 100
    assert engine.positions["000001.SZ"].quantity == 0
    assert engine.cash == before_cash + 1100.0  # 1100
    assert engine.positions["000001.SZ"].realized_pnl == 100.0


def test_paper_sell_without_position_is_rejected():
    engine = PaperTradingEngine()
    order = engine.execute_signal(_signal(action="SELL", price=11.0))
    assert order.status == "REJECTED"
    assert order.quantity == 0
    assert "no position" in order.reason


def test_paper_total_equity_uses_market_prices():
    engine = PaperTradingEngine(
        initial_cash=1000.0,
        lot_size=100,
        commission_rate=0.0,
    )
    engine.execute_signal(_signal(price=10.0))  # 100 shares, cash 0
    # market price 12 => value 1200
    assert engine.total_equity({"000001.SZ": 12.0}) == 1200.0


def test_paper_engine_invalid_configuration():
    with pytest.raises(ValueError):
        PaperTradingEngine(initial_cash=0)
    with pytest.raises(ValueError):
        PaperTradingEngine(lot_size=0)
    with pytest.raises(ValueError):
        PaperTradingEngine(commission_rate=-0.1)
    with pytest.raises(ValueError):
        PaperTradingEngine(slippage=-0.1)
