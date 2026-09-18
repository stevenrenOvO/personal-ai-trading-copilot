import pandas as pd

from backend.strategies import StrongSectorBreakoutStrategy


def _daily(*, breakout: bool = True, volume_ratio: float = 1.5, n: int = 25, ts_code: str = "000001.SZ") -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq="D").strftime("%Y%m%d").tolist()
    close = [10.0] * n
    high = [10.0] * n
    vol = [100.0] * n

    if breakout:
        high[-1] = 11.5
        close[-1] = 11.5
    else:
        close[-1] = 9.5

    vol[-1] = 100 * volume_ratio

    return pd.DataFrame({
        "ts_code": [ts_code] * n,
        "trade_date": dates,
        "open": [10.0] * n,
        "high": high,
        "low": [9.5] * n,
        "close": close,
        "pre_close": [10.0] * n,
        "change": [c - 10.0 for c in close],
        "pct_chg": [(c - 10.0) / 10.0 * 100 for c in close],
        "vol": vol,
        "amount": [100000.0] * n,
    })


def test_breakout_emits_buy_signal():
    strategy = StrongSectorBreakoutStrategy(lookback=5, min_volume_ratio=1.2, max_signals=2)
    signals = strategy.generate_signals(_daily(breakout=True, volume_ratio=1.5))

    assert len(signals) == 1
    assert signals[0].action == "BUY"
    assert signals[0].ts_code == "000001.SZ"


def test_no_signal_without_breakout():
    strategy = StrongSectorBreakoutStrategy(lookback=5)

    assert strategy.generate_signals(_daily(breakout=False)) == []


def test_sector_rank_filters_low_ranked_sector():
    strategy = StrongSectorBreakoutStrategy(
        lookback=5,
        max_sector_rank=3,
        max_signals=2,
        sector_by_code={"000001.SZ": "银行"},
        sector_rank={"银行": 4},
    )

    assert strategy.generate_signals(_daily(breakout=True, volume_ratio=1.5)) == []


def test_sector_rank_allows_top_sector():
    strategy = StrongSectorBreakoutStrategy(
        lookback=5,
        max_sector_rank=3,
        max_signals=2,
        sector_by_code={"000001.SZ": "银行"},
        sector_rank={"银行": 1},
    )
    signals = strategy.generate_signals(_daily(breakout=True, volume_ratio=1.5))

    assert len(signals) == 1
