"""Emotion-cycle replay over real history (Step 3 validation tool).

Run it to print, for every trading day in a range, the stage the rule table
produces together with the structure behind it::

    python -m backend.emotion.replay --start 20260901 --end 20260917

It reads only the local full-market history store; nothing is simulated and no
parameter is fitted to the output. Use it to check that a stage matches the
market structure of that day before trusting it.
"""

from __future__ import annotations

import argparse
import io
import sys

from backend.board.ladder import LimitLadderEngine
from backend.data.config import get_settings
from backend.data.providers.history import HistoryProvider
from backend.emotion.engine import EmotionEngine
from backend.market.engine import MarketEngine


def replay(start: str, end: str, provider=None) -> list[dict]:
    provider = provider or HistoryProvider()
    calendar = MarketEngine(provider).calendar
    days = [
        d
        for d in calendar.trading_days()
        if (not start or d >= start) and (not end or d <= end)
    ]
    rows: list[dict] = []
    for day in days:
        ladder = LimitLadderEngine(provider).snapshot(day)
        market = MarketEngine(provider).calculate(day)
        state = EmotionEngine(provider).calculate(day, market=market, ladder=ladder)
        rows.append(
            {
                "date": day,
                "stage": state.emotion_cycle,
                "strength": state.emotion_score,
                "exposure": state.recommended_exposure,
                "risk": state.risk_level,
                "limit_up": state.limit_up_count,
                "max_height": state.max_continuous_up,
                "broken_ratio": state.broken_ratio,
                "promotion": state.promotion_rate,
                "premium": state.yesterday_limit_up_avg_pct,
                "high_board": state.high_board_avg_pct,
                "advance_ratio": state.advance_decline_ratio,
                "rule": state.matched_rule,
                "confidence": state.confidence,
            }
        )
    return rows


def _fmt(value, digits=2, suffix="") -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}{suffix}"
    return f"{value}{suffix}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay the emotion cycle")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument(
        "--provider",
        default="history",
        choices=["history", "fixture"],
        help="history = local full-market store (default)",
    )
    args = parser.parse_args()
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    if args.provider == "fixture":
        from backend.data.providers.fixture import FixtureProvider

        provider = FixtureProvider(base_dir=get_settings().fixture_dir)
    else:
        provider = HistoryProvider()

    print(
        f"{'日期':<10}{'阶段':<8}{'强度':>6}{'仓位':>7}{'涨停':>6}{'最高':>5}"
        f"{'炸板率':>9}{'晋级率':>9}{'溢价':>9}{'高位板':>9}{'涨占比':>8}"
    )
    for row in replay(args.start, args.end, provider):
        print(
            f"{row['date']:<10}{row['stage']:<8}{_fmt(row['strength']):>6}"
            f"{_fmt(row['exposure'], 2):>7}{_fmt(row['limit_up']):>6}"
            f"{_fmt(row['max_height']):>5}{_fmt(row['broken_ratio'], 3):>9}"
            f"{_fmt(row['promotion'], 3):>9}{_fmt(row['premium']):>9}"
            f"{_fmt(row['high_board']):>9}{_fmt(row['advance_ratio'], 3):>8}"
        )
        print(f"           └ {row['rule']}")


if __name__ == "__main__":
    main()
