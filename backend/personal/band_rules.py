"""The user's own swing (波段) rules, evaluated on one stock's adjusted daily bars.

Rules as stated by the user (2026-11):

Entry
  1. 周线：周收盘在 13 周均线之上，且 13 周均线向上。
  2. 日线 MACD：DIF 回到零轴附近；绿柱缩短；金叉且放量。
Exit
  3. 有效跌破 21 日均线离场。
  4. 盈利 < 15% 时按 21 日线；盈利曾达到 15% 后，改为自最高收盘回撤 10% 离场。

Every number a status depends on is returned next to it, so a status can be
checked by hand. Interpretation choices that the stated rules leave open are
parameters (``BandParams``) with the defaults below, and are reported with
each result:

* "周线收盘" uses the last *completed* week; the week in progress is shown
  for reference only.
* "零轴附近" means ``|DIF| <= 2%`` of the close.
* "放量" means cross-day volume >= 1.2x the mean of the 5 sessions before it.
* "有效跌破" means a close more than 1% below MA21, or 2 consecutive closes
  below it.
* The 15% switch uses the highest close since entry (盈利曾达到 15%).

MACD uses the common A-share convention: EMA12 - EMA26 = DIF, EMA9(DIF) = DEA,
histogram = 2 x (DIF - DEA).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Optional

import pandas as pd


@dataclass(frozen=True)
class BandParams:
    weekly_ma: int = 13
    exit_ma: int = 21
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    zero_axis_band: float = 0.02
    cross_lookback: int = 3
    volume_ratio: float = 1.2
    volume_window: int = 5
    break_buffer: float = 0.01
    break_days: int = 2
    profit_switch: float = 0.15
    trailing_drawdown: float = 0.10
    min_daily_bars: int = 100

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Entry verdicts, most to least actionable.
BUY = "买点：金叉放量"
WAIT_CROSS = "等待金叉"
CROSS_NO_VOLUME = "金叉未放量"
BELOW_MA = "跌破21日线"
WAIT_DAILY = "周线符合，等待日线信号"
WEEKLY_FAIL = "周线不符合"
NO_DATA = "数据不足"


def _r(value: Any, digits: int = 4) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(number) else round(number, digits)


def _weeks(dates: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(dates.astype(str), format="%Y%m%d")
    iso = parsed.dt.isocalendar()
    return iso["year"].astype(int) * 100 + iso["week"].astype(int)


def evaluate(
    bars: pd.DataFrame,
    *,
    params: BandParams = BandParams(),
    last_week_complete: Optional[bool] = None,
    position: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Evaluate entry (and, with ``position``, exit) rules on daily bars.

    ``bars`` needs trade_date (YYYYMMDD), close, vol; ascending dates.
    ``last_week_complete`` says whether the latest bar closes its week; when
    unknown, a Friday bar is taken as the week's close.
    ``position``: {"entry_price", "entry_date", "entry_close_raw"(optional)}.
    """
    p = params
    frame = bars.sort_values("trade_date").reset_index(drop=True)
    out: dict[str, Any] = {"params": p.to_dict(), "bars": int(len(frame)),
                           "as_of": str(frame["trade_date"].iloc[-1]) if len(frame) else None}
    if len(frame) < p.min_daily_bars:
        out.update(entry=NO_DATA, conditions=[],
                   reason=f"日线 {len(frame)} 根，少于 {p.min_daily_bars} 根，无法可靠计算 13 周均线与 MACD")
        if position:
            out["exit"] = {"verdict": NO_DATA, "reason": out["reason"]}
        return out

    close = frame["close"].astype(float)
    vol = frame["vol"].astype(float)
    last_close = float(close.iloc[-1])

    # -- weekly ---------------------------------------------------------
    week = _weeks(frame["trade_date"])
    weekly = frame.assign(week=week).groupby("week", sort=True).agg(
        close=("close", "last"), end=("trade_date", "last"))
    weekly["ma"] = weekly["close"].rolling(p.weekly_ma).mean()
    if last_week_complete is None:
        last_day = datetime.strptime(str(frame["trade_date"].iloc[-1]), "%Y%m%d")
        last_week_complete = last_day.weekday() == 4
    done = weekly if last_week_complete else weekly.iloc[:-1]
    weekly_info: dict[str, Any] = {"last_week_complete": bool(last_week_complete)}
    if len(done) < p.weekly_ma + 1 or pd.isna(done["ma"].iloc[-2]):
        weekly_ok = None
    else:
        w_close, w_ma, w_ma_prev = (float(done["close"].iloc[-1]), float(done["ma"].iloc[-1]),
                                    float(done["ma"].iloc[-2]))
        above, rising = w_close > w_ma, w_ma > w_ma_prev
        weekly_ok = above and rising
        weekly_info.update(week_end=str(done["end"].iloc[-1]), week_close=_r(w_close, 3),
                           ma=_r(w_ma, 3), ma_prev=_r(w_ma_prev, 3), above=above, rising=rising)
    if not last_week_complete and len(weekly) and not pd.isna(weekly["ma"].iloc[-1]):
        weekly_info["in_progress"] = {"week_end_so_far": str(weekly["end"].iloc[-1]),
                                      "close": _r(weekly["close"].iloc[-1], 3), "ma": _r(weekly["ma"].iloc[-1], 3)}

    # -- daily MACD -----------------------------------------------------
    ema_fast = close.ewm(span=p.macd_fast, adjust=False).mean()
    ema_slow = close.ewm(span=p.macd_slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=p.macd_signal, adjust=False).mean()
    hist = 2 * (dif - dea)
    near_zero = abs(float(dif.iloc[-1])) <= p.zero_axis_band * last_close
    shrinking = bool(hist.iloc[-1] < 0 and hist.iloc[-1] > hist.iloc[-2])
    shrink_days = 0
    for i in range(len(hist) - 1, 0, -1):
        if hist.iloc[i] < 0 and hist.iloc[i] > hist.iloc[i - 1]:
            shrink_days += 1
        else:
            break
    cross = None
    for i in range(len(frame) - 1, max(len(frame) - 1 - p.cross_lookback, 0), -1):
        if dif.iloc[i - 1] <= dea.iloc[i - 1] and dif.iloc[i] > dea.iloc[i]:
            prior = vol.iloc[max(0, i - p.volume_window):i]
            ratio = float(vol.iloc[i] / prior.mean()) if len(prior) and prior.mean() > 0 else None
            cross = {"date": str(frame["trade_date"].iloc[i]), "dif": _r(dif.iloc[i]), "dea": _r(dea.iloc[i]),
                     "volume_ratio": _r(ratio, 2),
                     "volume_ok": ratio is not None and ratio >= p.volume_ratio,
                     "near_zero": abs(float(dif.iloc[i])) <= p.zero_axis_band * float(close.iloc[i])}
            break
    macd_info = {"dif": _r(dif.iloc[-1]), "dea": _r(dea.iloc[-1]),
                 "hist_last3": [_r(x) for x in hist.iloc[-3:]],
                 "dif_pct_of_close": _r(float(dif.iloc[-1]) / last_close, 4),
                 "near_zero": near_zero, "green_shrinking": shrinking, "shrink_days": shrink_days,
                 "cross": cross}

    # -- MA21 -------------------------------------------------------------
    ma21 = close.rolling(p.exit_ma).mean()
    ma_last = float(ma21.iloc[-1])
    below_streak = 0
    for c, m in zip(close.iloc[::-1], ma21.iloc[::-1]):
        if pd.notna(m) and c < m:
            below_streak += 1
        else:
            break
    broken = last_close < ma_last * (1 - p.break_buffer) or below_streak >= p.break_days
    ma_info = {"ma21": _r(ma_last, 3), "close": _r(last_close, 3),
               "distance_pct": _r(last_close / ma_last - 1, 4), "below_streak": below_streak,
               "effective_break": broken,
               "suggested_stop": _r(ma_last * (1 - p.break_buffer), 2)}

    conditions = [
        {"key": "weekly", "label": "周收盘在上升的13周均线之上", "passed": weekly_ok,
         "detail": (f"周收盘 {weekly_info.get('week_close')}，13周均线 {weekly_info.get('ma')}"
                    f"（上周 {weekly_info.get('ma_prev')}）" if weekly_ok is not None else "周线数据不足")},
        {"key": "zero_axis", "label": "DIF 在零轴附近", "passed": near_zero,
         "detail": f"DIF {macd_info['dif']}，占收盘 {macd_info['dif_pct_of_close']:.2%}（阈值 ±{p.zero_axis_band:.0%}）"},
        {"key": "green_shrinking", "label": "绿柱缩短", "passed": shrinking,
         "detail": f"最近三根柱 {macd_info['hist_last3']}，连续缩短 {shrink_days} 天"},
        {"key": "golden_cross", "label": f"{p.cross_lookback} 日内金叉且放量",
         "passed": bool(cross and cross["volume_ok"]),
         "detail": (f"{cross['date']} 金叉，量比 {cross['volume_ratio']}（需 ≥{p.volume_ratio}）"
                    if cross else f"近 {p.cross_lookback} 日无金叉")},
        {"key": "ma21", "label": "未有效跌破21日线", "passed": not broken,
         "detail": f"收盘 {ma_info['close']}，21日线 {ma_info['ma21']}，偏离 {ma_info['distance_pct']:.2%}，"
                   f"连续 {below_streak} 天在下方"},
    ]

    if weekly_ok is None:
        entry = NO_DATA
    elif not weekly_ok:
        entry = WEEKLY_FAIL
    elif broken:
        entry = BELOW_MA
    elif cross and cross["volume_ok"] and cross["near_zero"]:
        entry = BUY
    elif cross and cross["near_zero"]:
        entry = CROSS_NO_VOLUME
    elif shrinking and near_zero:
        entry = WAIT_CROSS
    else:
        entry = WAIT_DAILY
    out.update(entry=entry, conditions=conditions, weekly=weekly_info, macd=macd_info, ma=ma_info)

    if position:
        out["exit"] = _exit(frame, close, ma_info, position, p)
    return out


def _exit(frame: pd.DataFrame, close: pd.Series, ma_info: dict, position: dict, p: BandParams) -> dict:
    entry_date = str(position.get("entry_date") or "")
    entry_price = float(position["entry_price"])
    since = frame["trade_date"].astype(str) >= entry_date
    if not since.any():
        return {"verdict": NO_DATA, "reason": f"复权日线未覆盖买入日 {entry_date}"}
    # Bring the raw entry price onto the forward-adjusted scale.
    factor, note = 1.0, None
    raw_close = position.get("entry_close_raw")
    entry_rows = frame[frame["trade_date"].astype(str) == entry_date]
    if raw_close and not entry_rows.empty:
        factor = float(entry_rows["close"].iloc[0]) / float(raw_close)
        if abs(factor - 1) > 1e-4:
            note = f"买入后有除权，成本按复权系数 {factor:.4f} 换算"
    cost = entry_price * factor
    last = float(close.iloc[-1])
    high = float(close[since].max())
    profit, max_profit = last / cost - 1, high / cost - 1
    trailing = max_profit >= p.profit_switch
    trail_stop = high * (1 - p.trailing_drawdown)
    if trailing:
        hit = last <= trail_stop
        rule = f"盈利曾达 {max_profit:.1%}（≥{p.profit_switch:.0%}），自最高收盘 {high:.2f} 回撤 {p.trailing_drawdown:.0%} 离场"
        verdict = "离场：自最高回撤" if hit else "持有"
        stop = trail_stop
    else:
        hit = bool(ma_info["effective_break"])
        rule = f"盈利未达 {p.profit_switch:.0%}（最高 {max_profit:.1%}），按21日线"
        verdict = "离场：有效跌破21日线" if hit else "持有"
        stop = ma_info["suggested_stop"]
    return {"verdict": verdict, "rule": rule, "cost_adjusted": _r(cost, 3), "close": _r(last, 3),
            "profit_pct": _r(profit, 4), "max_profit_pct": _r(max_profit, 4),
            "highest_close": _r(high, 3), "trailing_mode": trailing, "stop": _r(stop, 2),
            "trailing_stop": _r(trail_stop, 2), "note": note}
