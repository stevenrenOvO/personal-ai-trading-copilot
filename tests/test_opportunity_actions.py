"""Tests for how an opportunity is turned into an actionable verdict.

The regression these pin: on a weak day the engine used to mark *every* row
"AVOID" (the ranking score was crushed by a blanket penalty), so the list told
the user nothing. Environment problems must read "wait", and only
stock-specific problems may read "avoid".
"""

from __future__ import annotations

import pandas as pd

from backend.opportunity.engine import OpportunityEngine
from backend.sector import SectorEngine
from backend.strategies.base import Strategy, TradeSignal


class _FixedSignalStrategy(Strategy):
    name = "fixed"
    strategy_id = "fixed"
    version = "1.0"

    def __init__(self, ts_code: str = "000001.SZ") -> None:
        self._ts_code = ts_code
        self.sector_context: dict | None = None

    def set_sector_context(self, sector_by_code, sector_rank):
        self.sector_context = {"codes": dict(sector_by_code), "ranks": dict(sector_rank)}

    def generate_signals(self, daily: pd.DataFrame) -> list[TradeSignal]:
        dates = sorted(str(d) for d in daily["trade_date"].unique())
        return [
            self.make_signal(
                ts_code=self._ts_code,
                trade_date=dates[-1],
                action="BUY",
                price=10.0,
                strength=1.0,
                reason="fixed test signal",
            )
        ]


class _StubSector:
    """Minimal stand-in for a ranked sector score."""

    def __init__(self, name: str, rank: int, score: float, codes: list[str]) -> None:
        self.name = name
        self.rank = rank
        self.score = score
        self.stock_codes = tuple(codes)
        self.leader = codes[0] if codes else None


class _StubSectorEngine:
    def __init__(self, sectors) -> None:
        self._sectors = sectors
        self.seen_dates: list[str] = []

    def ranked(self, date, *args, **kwargs):
        self.seen_dates.append(date)
        return list(self._sectors)


class _StubMarket:
    def __init__(self, score: float, state: str = "偏弱") -> None:
        self.score = score
        self.state = state


class _StubEmotion:
    def __init__(self, cycle: str, score: float = 50.0, exposure: float = 0.4) -> None:
        self.emotion_cycle = cycle
        self.emotion_score = score
        self.recommended_exposure = exposure


class _StubBoard:
    def __init__(self, grade: str, score: float = 45.0) -> None:
        self.grade = grade
        self.score = score


class _StubProvider:
    """Just enough provider for one signal to be evaluated."""

    def stock_basic(self, *, ts_codes=None, list_status=None):
        return pd.DataFrame(
            {
                "ts_code": ["000001.SZ", "600000.SH"],
                "symbol": ["000001", "600000"],
                "name": ["平安银行", "浦发银行"],
            }
        )

    def daily(self, *, ts_codes=None, start_date=None, end_date=None):
        return pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "trade_date": ["20240103"],
                "open": [10.0],
                "high": [10.2],
                "low": [9.9],
                "close": [10.1],
                "pre_close": [10.0],
                "change": [0.1],
                "pct_chg": [1.0],
                "vol": [1000.0],
                "amount": [10000.0],
            }
        )


def _evaluate(*, market_score: float, cycle: str, grade: str, sector_score: float, sector_rank: int):
    provider = _StubProvider()
    strategy = _FixedSignalStrategy()
    sectors = [_StubSector("银行", sector_rank, sector_score, ["000001.SZ"])]
    engine = OpportunityEngine(provider, strategy, _StubSectorEngine(sectors))
    return engine.scan(
        "20240103",
        market=_StubMarket(market_score),
        emotion=_StubEmotion(cycle),
        board=_StubBoard(grade),
        sectors=sectors,
    )


def test_weak_environment_reads_wait_not_avoid():
    """A good name in a bad tape is something to watch, not something to shun."""
    opps = _evaluate(
        market_score=44.0, cycle="分化", grade="C", sector_score=80.0, sector_rank=1
    )
    assert len(opps) == 1
    opp = opps[0]
    assert opp.suggested_action == "WAIT"
    assert opp.buy_ready is False
    assert opp.environment_blockers          # the tape is the problem
    assert opp.stock_blockers == ()          # the name itself is fine


def test_strong_environment_marks_buy_ready():
    opps = _evaluate(
        market_score=85.0, cycle="发酵", grade="A", sector_score=95.0, sector_rank=1
    )
    assert len(opps) == 1
    opp = opps[0]
    assert opp.buy_ready is True
    assert opp.suggested_action in ("BUY WATCH", "WATCH")
    assert opp.blockers == ()


def test_stock_specific_problem_still_reads_avoid():
    """A name in a weak, low-ranked sector is genuinely not a candidate."""
    opps = _evaluate(
        market_score=85.0, cycle="发酵", grade="A", sector_score=20.0, sector_rank=9
    )
    assert len(opps) == 1
    opp = opps[0]
    assert opp.suggested_action == "AVOID"
    assert opp.buy_ready is False
    assert any("板块" in reason for reason in opp.stock_blockers)


def test_scan_attaches_sector_context_to_a_copy_of_the_strategy():
    provider = _StubProvider()
    strategy = _FixedSignalStrategy()
    sectors = [_StubSector("银行", 1, 90.0, ["000001.SZ"])]
    engine = OpportunityEngine(provider, strategy, _StubSectorEngine(sectors))

    engine.scan(
        "20240103",
        market=_StubMarket(80.0),
        emotion=_StubEmotion("发酵"),
        board=_StubBoard("A"),
        sectors=sectors,
    )

    # The shared instance must stay untouched (two concurrent scans for
    # different dates would otherwise overwrite each other's context).
    assert strategy.sector_context is None


def test_signal_ids_stay_unique_across_strategy_copies():
    strategy = _FixedSignalStrategy()
    import copy

    first = copy.copy(strategy)
    second = copy.copy(strategy)
    frame = _StubProvider().daily()
    id_a = first.generate_signals(frame)[0].signal_id
    id_b = second.generate_signals(frame)[0].signal_id
    assert id_a != id_b
