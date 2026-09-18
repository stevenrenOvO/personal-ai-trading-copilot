"""Limit-up ladder, leader structure and yesterday's limit-up premium.

Scope (Step 2): describe **what happened**, never **what to buy**. Nothing in
this module produces a buy/sell verdict; the 角色 labels are structural
positions in a ladder, not recommendations.

Calculation definitions (口径)
------------------------------
涨停 (limit up)
    ``close >= up_limit``, where ``up_limit`` comes from the provider's
    ``stk_limit`` table. For the local history store it is *derived* from the
    stored ``pre_close`` and the exchange rules (主板 10%, 创业/科创 20%,
    北交所 30%, ST 5%), half-up rounded to 0.01 — not read from a vendor field.
连板高度 (height)
    Number of consecutive trading days, ending on the requested date, on which
    the stock closed at its up limit. Days come from the local trading calendar,
    so a suspension (no bar that day) breaks the run instead of being skipped.
    首板 = 1, 二板 = 2, 三板 = 3, 四板及以上 = >=4.
断层 (gap)
    A height between 1 and the day's maximum that no stock reached, e.g. a day
    with 首板 and 三板 but no 二板 has a gap at 2.
炸板 (broken board)
    ``high >= up_limit`` but ``close < up_limit``.
回封 (reseal, daily-bar approximation)
    ``high >= up_limit`` and ``low < up_limit`` and ``close >= up_limit``.
    Daily bars cannot tell "opened the limit then resealed" from "touched the
    limit then closed there", so this is labelled as an approximation.
昨日涨停溢价 (yesterday's limit-up premium)
    For the previous trading day's limit-up stocks: their pct_chg on the
    requested date. Reported as average, median, advance/decline split and
    晋级率 (how many are limit-up again).

Honest gaps (never invented)
----------------------------
* 集合竞价过程 / 封单量 / 精确涨停时间 need minute or order-book data: 未接入.
* 流通市值 (daily_basic) is not ingested, so 中军 / 补涨 can only be inferred
  from ladder structure; that limitation is stated in the payload.
* New listings and corporate-action days are excluded from limit judgements
  (their limit rules / base price are not reliable) and counted separately.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from backend.data.calendar import TradingCalendar
from backend.data.providers._a_share_utils import is_st_flag, limit_rate
from backend.data.providers.base import DataProvider
from backend.market.limit_ladder import heights_from_sets

# Days after listing during which A-share limit rules are not reliable for a
# price-limit judgement: 创业板/科创板 trade without a limit for their first 5
# trading days and 主板 has a 44% first-day cap.
_NEW_LISTING_TRADING_DAYS = 5

# A real limit-board move can round slightly past its nominal rate (a 0.58 close
# on a 0.73 base is -20.55% on a 20% board), so only larger excursions are
# treated as corporate actions (ex-dividend / bonus shares) rather than trades.
_CORPORATE_ACTION_SLACK = 0.6

# Main-board ST stocks are capped at 5%. The local ``stock_basic`` names are a
# point-in-time snapshot and go stale (e.g. 600734 is still named "*ST实达"
# while it clearly trades on a 10% cap), so the 5% cap is only applied when the
# stock's own recent moves are actually consistent with a 5% band.
_ST_RATE_MAX_MOVE = 5.6

_UNAVAILABLE = (
    "集合竞价过程、封单量、精确涨停时间：需要分钟级/盘口数据，当前未接入",
    "流通市值与量比（daily_basic）：未接入，中军/补涨只能按连板高度结构推断",
)


def _round(value: Optional[float], digits: int = 2) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    return round(float(value), digits)


def level_label(height: int) -> str:
    if height <= 0:
        return "未涨停"
    if height == 1:
        return "首板"
    if height == 2:
        return "二板"
    if height == 3:
        return "三板"
    return f"{height}板"


def bucket_label(height: int) -> str:
    """Summary bucket: 首板 / 二板 / 三板 / 四板及以上."""
    if height <= 1:
        return "首板"
    if height == 2:
        return "二板"
    if height == 3:
        return "三板"
    return "四板及以上"


@dataclass(frozen=True)
class LadderStock:
    """One limit-up stock on the requested day."""

    ts_code: str
    symbol: str = ""
    name: str = ""
    industry: str = ""
    close: Optional[float] = None
    pct_chg: Optional[float] = None
    height: int = 0
    up_limit: Optional[float] = None
    down_limit: Optional[float] = None
    amount: Optional[float] = None
    is_st: bool = False
    structural_role: str = "普通"

    def to_dict(self) -> dict[str, Any]:
        return {
            "ts_code": self.ts_code,
            "symbol": self.symbol,
            "name": self.name,
            "industry": self.industry,
            "close": _round(self.close),
            "pct_chg": _round(self.pct_chg),
            "height": self.height,
            "level": level_label(self.height),
            "up_limit": _round(self.up_limit),
            "down_limit": _round(self.down_limit),
            "amount": _round(self.amount, 0),
            "is_st": self.is_st,
            "structural_role": self.structural_role,
        }


@dataclass(frozen=True)
class LadderLevel:
    height: int
    count: int
    stocks: tuple[LadderStock, ...] = ()
    # Every code at this height. ``stocks`` is capped for payload size, but the
    # opportunity engine needs the full set to build its candidate pool.
    all_codes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "height": self.height,
            "label": level_label(self.height),
            "count": self.count,
            "stocks": [s.to_dict() for s in self.stocks],
        }


@dataclass(frozen=True)
class PromotionStat:
    """How yesterday's ``height``-board stocks did today."""

    height: int
    count: int
    promoted: int

    @property
    def rate(self) -> Optional[float]:
        return round(self.promoted / self.count, 4) if self.count else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "height": self.height,
            "label": level_label(self.height),
            "count": self.count,
            "promoted": self.promoted,
            "rate": self.rate,
        }


@dataclass(frozen=True)
class YesterdayPremium:
    """Yesterday's limit-up stocks, measured on the requested date."""

    date: str
    measured_on: str
    count: int = 0
    evaluated: int = 0
    suspended: int = 0
    avg_pct: Optional[float] = None
    median_pct: Optional[float] = None
    up_count: int = 0
    down_count: int = 0
    flat_count: int = 0
    up_ratio: Optional[float] = None
    promoted_count: int = 0
    promotion_rate: Optional[float] = None
    broken_count: int = 0
    limit_down_count: int = 0
    best: Optional[str] = None
    best_pct: Optional[float] = None
    worst: Optional[str] = None
    worst_pct: Optional[float] = None
    promotions: tuple[PromotionStat, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "measured_on": self.measured_on,
            "count": self.count,
            "evaluated": self.evaluated,
            "suspended": self.suspended,
            "avg_pct": _round(self.avg_pct),
            "median_pct": _round(self.median_pct),
            "up_count": self.up_count,
            "down_count": self.down_count,
            "flat_count": self.flat_count,
            "up_ratio": _round(self.up_ratio, 4),
            "promoted_count": self.promoted_count,
            "promotion_rate": _round(self.promotion_rate, 4),
            "broken_count": self.broken_count,
            "limit_down_count": self.limit_down_count,
            "best": self.best,
            "best_pct": _round(self.best_pct),
            "worst": self.worst,
            "worst_pct": _round(self.worst_pct),
            "promotions": [p.to_dict() for p in self.promotions],
        }


@dataclass(frozen=True)
class IndustryLadder:
    """Ladder structure inside one industry."""

    name: str
    limit_up_count: int
    max_height: int
    level_counts: dict[int, int] = field(default_factory=dict)
    leader_candidates: tuple[str, ...] = ()
    mid_candidates: tuple[str, ...] = ()
    catch_up_candidates: tuple[str, ...] = ()
    stocks: tuple[LadderStock, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "limit_up_count": self.limit_up_count,
            "max_height": self.max_height,
            "max_level": level_label(self.max_height),
            "level_counts": {level_label(h): c for h, c in sorted(self.level_counts.items())},
            "leader_candidates": list(self.leader_candidates),
            "mid_candidates": list(self.mid_candidates),
            "catch_up_candidates": list(self.catch_up_candidates),
            "stocks": [s.to_dict() for s in self.stocks],
        }


@dataclass(frozen=True)
class LadderSnapshot:
    date: str
    available: bool = True
    source: str = "local_history"
    universe_size: int = 0
    traded_count: int = 0
    limit_up_count: int = 0
    limit_down_count: int = 0
    broken_count: int = 0
    reseal_count: int = 0
    max_height: int = 0
    levels: tuple[LadderLevel, ...] = ()
    buckets: dict[str, int] = field(default_factory=dict)
    gaps: tuple[int, ...] = ()
    is_complete: bool = False
    highest: tuple[LadderStock, ...] = ()
    yesterday: Optional[YesterdayPremium] = None
    industries: tuple[IndustryLadder, ...] = ()
    dominant_industry: Optional[str] = None
    broken_stocks: tuple[LadderStock, ...] = ()
    excluded_new_listing: int = 0
    excluded_corporate_action: int = 0
    # Code-level detail behind those two counters, plus the day's board
    # failures: the hard-elimination layers need codes, not just totals.
    excluded_new_listing_codes: tuple[str, ...] = ()
    excluded_corporate_action_codes: tuple[str, ...] = ()
    broken_codes: tuple[str, ...] = ()
    reseal_codes: tuple[str, ...] = ()
    prev_broken_codes: tuple[str, ...] = ()
    # Per-day structure for the whole lookback window: the emotion engine needs
    # the *trend* (height rising/falling, ladder building/breaking), not just
    # today's snapshot. Cheap to produce -- it is the same pass that already
    # computes the ladder.
    recent: tuple["LadderDay", ...] = ()
    notes: tuple[str, ...] = ()
    unavailable: tuple[str, ...] = _UNAVAILABLE

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "available": self.available,
            "source": self.source,
            "universe_size": self.universe_size,
            "traded_count": self.traded_count,
            "limit_up_count": self.limit_up_count,
            "limit_down_count": self.limit_down_count,
            "broken_count": self.broken_count,
            "reseal_count": self.reseal_count,
            "max_height": self.max_height,
            "buckets": dict(self.buckets),
            "levels": [level.to_dict() for level in self.levels],
            "gaps": list(self.gaps),
            "gap_labels": [level_label(g) for g in self.gaps],
            "is_complete": self.is_complete,
            "highest": [s.to_dict() for s in self.highest],
            "yesterday": self.yesterday.to_dict() if self.yesterday else None,
            "industries": [i.to_dict() for i in self.industries],
            "dominant_industry": self.dominant_industry,
            "broken_stocks": [s.to_dict() for s in self.broken_stocks],
            "excluded_new_listing": self.excluded_new_listing,
            "excluded_corporate_action": self.excluded_corporate_action,
            "recent": [d.to_dict() for d in self.recent],
            "notes": list(self.notes),
            "unavailable": list(self.unavailable),
        }


@dataclass(frozen=True)
class LadderDay:
    """One day's structure summary inside the lookback window."""

    date: str
    limit_up_count: int = 0
    limit_down_count: int = 0
    broken_count: int = 0
    reseal_count: int = 0
    max_height: int = 0
    buckets: dict[str, int] = field(default_factory=dict)
    promotion_rate: Optional[float] = None
    avg_premium: Optional[float] = None
    up_ratio: Optional[float] = None
    high_board_avg_pct: Optional[float] = None
    is_complete: bool = False
    gap_count: int = 0
    traded_count: int = 0
    advance_count: int = 0
    decline_count: int = 0
    total_amount: Optional[float] = None

    @property
    def broken_ratio(self) -> Optional[float]:
        attempted = self.limit_up_count + self.broken_count
        return round(self.broken_count / attempted, 4) if attempted else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "limit_up_count": self.limit_up_count,
            "limit_down_count": self.limit_down_count,
            "broken_count": self.broken_count,
            "broken_ratio": self.broken_ratio,
            "reseal_count": self.reseal_count,
            "max_height": self.max_height,
            "buckets": dict(self.buckets),
            "promotion_rate": _round(self.promotion_rate, 4),
            "avg_premium": _round(self.avg_premium),
            "up_ratio": _round(self.up_ratio, 4),
            "high_board_avg_pct": _round(self.high_board_avg_pct),
            "is_complete": self.is_complete,
            "gap_count": self.gap_count,
            "traded_count": self.traded_count,
            "advance_count": self.advance_count,
            "decline_count": self.decline_count,
            "total_amount": _round(self.total_amount, 0),
        }


class LimitLadderEngine:
    """Build the limit-up ladder for one trading day from real daily bars."""

    def __init__(
        self,
        provider: DataProvider,
        *,
        lookback_days: int = 20,
        stocks_per_level: int = 25,
        top_industries: int = 8,
    ) -> None:
        if lookback_days < 2:
            raise ValueError("lookback_days must be at least 2")
        self.provider = provider
        self.lookback_days = lookback_days
        self.stocks_per_level = stocks_per_level
        self.top_industries = top_industries
        self.calendar = TradingCalendar(provider)
        self._by_day: dict[str, pd.DataFrame] = {}
        self._pct_maps: dict[str, dict[str, Any]] = {}

    # -- public --------------------------------------------------------
    def snapshot(self, date: str) -> LadderSnapshot:
        basic = self._basic()
        if basic.empty:
            return LadderSnapshot(date=date, available=False, notes=("缺少股票基础信息",))
        codes = basic["ts_code"].tolist()

        all_days = self.calendar.trading_days()
        window = [d for d in all_days if d <= date][-self.lookback_days :]
        if not window or window[-1] != date:
            return LadderSnapshot(
                date=date,
                available=False,
                notes=(f"本地历史不含交易日 {date}，无法计算梯队",),
            )

        daily = self.provider.daily(ts_codes=codes, start_date=window[0], end_date=date)
        if daily.empty:
            return LadderSnapshot(
                date=date, available=False, notes=(f"{date} 无本地日线数据",)
            )
        daily = daily.copy()
        daily["trade_date"] = daily["trade_date"].astype(str)
        present_days = [d for d in window if d in set(daily["trade_date"])]
        if date not in present_days:
            return LadderSnapshot(
                date=date, available=False, notes=(f"{date} 无本地日线数据",)
            )

        info = self._stock_info(basic)
        sets = self._daily_sets(daily, present_days, codes, info, all_days)
        heights_by_day = heights_from_sets(sets["limit_up"], present_days)
        heights = heights_by_day.get(date, {})

        levels, gaps, highest = self._levels(date, heights, sets, info)
        yesterday = self._yesterday_premium(present_days, sets, heights_by_day)
        recent = self._recent_days(present_days, sets, heights_by_day)
        industries = self._industries(date, heights, sets, info)
        dominant = self._dominant_industry(industries, highest)

        notes = []
        new_listing_today = sets["new_listing"].get(date, set())
        corporate_today = sets["corporate_action"].get(date, set())
        stale_st = sets.get("stale_st_names") or set()
        if stale_st:
            notes.append(
                f"{len(stale_st)} 只股票名称带 ST 但近期波动超出 5% 区间，"
                "已按 10% 涨跌幅限制处理（本地名称快照可能已过期）"
            )
        if new_listing_today:
            notes.append(
                f"{date} 有 {len(new_listing_today)} 只次新股（上市不足 "
                f"{_NEW_LISTING_TRADING_DAYS} 个交易日），涨跌幅限制不适用，已排除出涨停判定"
            )
        if corporate_today:
            notes.append(
                f"{date} 有 {len(corporate_today)} 只疑似除权/送转（涨跌幅超出该板块限制），"
                "已排除出涨停判定"
            )
        notes.append(
            "回封为日线口径（当日曾跌破涨停价、收盘收于涨停价），无法区分盘中反复开板"
        )
        notes.append("连续涨停天数按本地交易日序列计算，停牌日会中断连板")

        return LadderSnapshot(
            date=date,
            available=True,
            universe_size=len(codes),
            traded_count=int(daily[daily["trade_date"] == date]["ts_code"].nunique()),
            limit_up_count=len(heights),
            limit_down_count=len(sets["limit_down"].get(date, set())),
            broken_count=len(sets["broken"].get(date, set())),
            reseal_count=len(sets["reseal"].get(date, set())),
            max_height=max(heights.values()) if heights else 0,
            levels=levels,
            buckets={
                bucket: sum(level.count for level in levels if bucket_label(level.height) == bucket)
                for bucket in ("首板", "二板", "三板", "四板及以上")
            },
            gaps=gaps,
            is_complete=not gaps and bool(heights),
            highest=highest,
            yesterday=yesterday,
            industries=industries,
            dominant_industry=dominant,
            broken_stocks=self._broken_stocks(date, sets, info),
            excluded_new_listing=len(new_listing_today),
            excluded_corporate_action=len(corporate_today),
            excluded_new_listing_codes=tuple(sorted(new_listing_today)),
            excluded_corporate_action_codes=tuple(sorted(corporate_today)),
            broken_codes=tuple(sorted(sets["broken"].get(date, set()))),
            reseal_codes=tuple(sorted(sets["reseal"].get(date, set()))),
            prev_broken_codes=(
                tuple(sorted(sets["broken"].get(present_days[-2], set())))
                if len(present_days) >= 2
                else ()
            ),
            recent=recent,
            notes=tuple(notes),
        )

    # -- internals -----------------------------------------------------
    def _basic(self) -> pd.DataFrame:
        try:
            basic = self.provider.stock_basic(list_status="L")
        except Exception:  # noqa: BLE001
            basic = self.provider.stock_basic()
        if basic is None or basic.empty or "ts_code" not in basic.columns:
            return pd.DataFrame(columns=["ts_code"])
        return basic

    def _stock_info(self, basic: pd.DataFrame) -> dict[str, dict[str, Any]]:
        info: dict[str, dict[str, Any]] = {}
        has_st = "is_st" in basic.columns
        has_industry = "industry" in basic.columns
        has_list_date = "list_date" in basic.columns
        for _, row in basic.iterrows():
            code = str(row["ts_code"])
            name = str(row.get("name") or "")
            explicit = str(row.get("is_st") or "").strip().upper() if has_st else ""
            info[code] = {
                "symbol": str(row.get("symbol") or ""),
                "name": name,
                "industry": str(row.get("industry") or "") if has_industry else "",
                "list_date": str(row.get("list_date") or "") if has_list_date else "",
                # BaoStock leaves is_st empty, so ST status is also read from the
                # name (ST/*ST prefixes) instead of being assumed False.
                "is_st": is_st_flag(explicit) or "ST" in name.upper(),
            }
        return info

    def _daily_sets(
        self,
        daily: pd.DataFrame,
        days: list[str],
        codes: list[str],
        info: dict[str, dict[str, Any]],
        all_days: list[str],
    ) -> dict[str, Any]:
        """Per-day limit-up / limit-down / broken / reseal code sets."""
        limit_up: dict[str, set[str]] = {}
        limit_down: dict[str, set[str]] = {}
        broken: dict[str, set[str]] = {}
        reseal: dict[str, set[str]] = {}
        new_listing: dict[str, set[str]] = {}
        corporate_action: dict[str, set[str]] = {}

        day_index = {day: index for index, day in enumerate(all_days)}
        list_index = {
            code: (
                (
                    # Listed before the store window starts: definitely not new.
                    -10**6
                    if bisect_left(all_days, meta["list_date"]) == 0
                    and str(meta["list_date"]) <= all_days[0]
                    else bisect_left(all_days, meta["list_date"])
                )
                if len(str(meta.get("list_date") or "")) == 8
                else -10**6
            )
            for code, meta in info.items()
        }
        rate_pct = {
            code: limit_rate(code, False) * 100 for code in info
        }
        # Largest absolute move per code inside the window: used to decide
        # whether a name-based ST flag is still true (see _effective_st_flags).
        window_abs = (
            daily.assign(_abs=pd.to_numeric(daily["pct_chg"], errors="coerce").abs())
            .groupby("ts_code")["_abs"]
            .max()
            .to_dict()
        )
        effective_st, stale_st_names = self._effective_st_flags(
            info, window_abs, rate_pct
        )
        rate_pct = {
            code: limit_rate(code, effective_st.get(code, False)) * 100 for code in info
        }

        by_day = {str(d): g for d, g in daily.groupby("trade_date")}
        self._by_day = by_day
        self._pct_maps = {}
        for day in days:
            bars = by_day.get(day)
            if bars is None or bars.empty:
                continue
            limits = self._limits(codes, day)
            merged = bars.merge(
                limits[["ts_code", "trade_date", "up_limit", "down_limit"]]
                if not limits.empty
                else pd.DataFrame(
                    columns=["ts_code", "trade_date", "up_limit", "down_limit"]
                ),
                on=["ts_code", "trade_date"],
                how="left",
            )
            close = pd.to_numeric(merged["close"], errors="coerce")
            high = pd.to_numeric(merged["high"], errors="coerce")
            low = pd.to_numeric(merged["low"], errors="coerce")
            pct = pd.to_numeric(
                merged["pct_chg"]
                if "pct_chg" in merged.columns
                else pd.Series(float("nan"), index=merged.index),
                errors="coerce",
            )
            up = pd.to_numeric(merged["up_limit"], errors="coerce")
            down = pd.to_numeric(merged["down_limit"], errors="coerce")

            codes_today = merged["ts_code"].astype(str)
            # Days since listing, computed from the store's own trading calendar
            # so a stock listed inside the window is recognised immediately.
            since_listing = codes_today.map(
                lambda code: day_index[day] - list_index.get(code, -10**6) + 1
            )
            is_new = since_listing.between(1, _NEW_LISTING_TRADING_DAYS)
            moved_too_far = pct.abs() > (codes_today.map(rate_pct) + _CORPORATE_ACTION_SLACK)
            is_corporate_action = moved_too_far & ~is_new
            excluded = is_new | is_corporate_action
            new_listing[day] = set(codes_today[is_new])
            corporate_action[day] = set(codes_today[is_corporate_action])

            hit_up = close.notna() & up.notna() & (close >= up) & ~excluded
            hit_down = close.notna() & down.notna() & (close <= down) & ~excluded
            hit_broken = (
                high.notna() & up.notna() & (high >= up) & (close < up) & ~excluded
            )
            hit_reseal = (
                high.notna()
                & low.notna()
                & up.notna()
                & (high >= up)
                & (low < up)
                & (close >= up)
                & ~excluded
            )
            limit_up[day] = set(codes_today[hit_up])
            limit_down[day] = set(codes_today[hit_down])
            broken[day] = set(codes_today[hit_broken])
            reseal[day] = set(codes_today[hit_reseal])

        return {
            "limit_up": limit_up,
            "limit_down": limit_down,
            "broken": broken,
            "reseal": reseal,
            "new_listing": new_listing,
            "corporate_action": corporate_action,
            "effective_st": effective_st,
            "stale_st_names": stale_st_names,
            "by_day": by_day,
            "limits_day": {day: self._limits(codes, day) for day in days},
        }

    def _pct_map(self, day: str) -> dict[str, Any]:
        """date -> {code: pct_chg}, built once per day instead of per query."""
        cached = self._pct_maps.get(day)
        if cached is not None:
            return cached
        bars = self._by_day.get(day)
        if bars is None or bars.empty:
            return {}
        mapping = dict(
            zip(bars["ts_code"].astype(str), pd.to_numeric(bars["pct_chg"], errors="coerce"))
        )
        self._pct_maps[day] = mapping
        return mapping

    def _limits(self, codes: list[str], day: str) -> pd.DataFrame:
        try:
            frame = self.provider.stk_limit(ts_codes=codes, trade_date=day)
        except Exception:  # noqa: BLE001
            return pd.DataFrame(columns=["ts_code", "trade_date", "up_limit", "down_limit"])
        return frame if frame is not None else pd.DataFrame()

    @staticmethod
    def _effective_st_flags(
        info: dict[str, dict[str, Any]],
        window_abs: dict[str, float],
        board_rate_pct: dict[str, float],
    ) -> tuple[dict[str, bool], set[str]]:
        """Decide the 5%-cap question from the data, not from a stale name.

        Only the main board is ambiguous (创业板/科创板 stay at 20% either way).
        A stock named "ST" whose own recent moves exceed a 5% band is traded on
        the normal 10% cap, so the flag is corrected and reported.
        """
        effective: dict[str, bool] = {}
        stale: set[str] = set()
        for code, meta in info.items():
            named_st = bool(meta.get("is_st"))
            if not named_st:
                effective[code] = False
                continue
            if board_rate_pct.get(code, 10.0) > 10.0:
                # 创业板/科创板 ST keeps the 20% band, so nothing to decide.
                effective[code] = True
                continue
            observed = window_abs.get(code)
            if observed is not None and not pd.isna(observed) and float(observed) > _ST_RATE_MAX_MOVE:
                effective[code] = False
                stale.add(code)
            else:
                effective[code] = True
        return effective, stale

    def _stock(
        self,
        code: str,
        day: str,
        sets: dict[str, Any],
        info: dict[str, dict[str, Any]],
        *,
        height: int,
        role: str = "普通",
    ) -> Optional[LadderStock]:
        bars = sets["by_day"].get(day)
        if bars is None:
            return None
        row = bars[bars["ts_code"].astype(str) == code]
        if row.empty:
            return None
        row = row.iloc[0]
        limits = sets["limits_day"].get(day)
        up = down = None
        if limits is not None and not limits.empty:
            match = limits[limits["ts_code"].astype(str) == code]
            if not match.empty:
                up = match.iloc[0].get("up_limit")
                down = match.iloc[0].get("down_limit")
        meta = info.get(code, {})
        return LadderStock(
            ts_code=code,
            symbol=str(meta.get("symbol") or ""),
            name=str(meta.get("name") or ""),
            industry=str(meta.get("industry") or ""),
            close=row.get("close"),
            pct_chg=row.get("pct_chg"),
            height=height,
            up_limit=up,
            down_limit=down,
            amount=row.get("amount"),
            is_st=bool(sets.get("effective_st", {}).get(code, meta.get("is_st"))),
            structural_role=role,
        )

    def _levels(
        self,
        date: str,
        heights: dict[str, int],
        sets: dict[str, Any],
        info: dict[str, dict[str, Any]],
    ) -> tuple[tuple[LadderLevel, ...], tuple[int, ...], tuple[LadderStock, ...]]:
        by_height: dict[int, list[str]] = {}
        for code, height in heights.items():
            by_height.setdefault(height, []).append(code)
        max_height = max(by_height) if by_height else 0
        gaps = tuple(h for h in range(1, max_height) if h not in by_height)

        levels: list[LadderLevel] = []
        for height in sorted(by_height, reverse=True):
            codes = sorted(by_height[height])
            stocks = []
            for code in codes[: self.stocks_per_level]:
                stock = self._stock(code, date, sets, info, height=height)
                if stock is not None:
                    stocks.append(stock)
            levels.append(
            LadderLevel(
                height=height,
                count=len(codes),
                stocks=tuple(stocks),
                all_codes=tuple(codes),
            )
            )

        highest_codes = by_height.get(max_height, []) if max_height else []
        highest = []
        for code in highest_codes:
            stock = self._stock(
                code, date, sets, info, height=max_height, role="最高板"
            )
            if stock is not None:
                highest.append(stock)
        return tuple(levels), gaps, tuple(highest)

    def _yesterday_premium(
        self,
        days: list[str],
        sets: dict[str, Any],
        heights_by_day: dict[str, dict[str, int]],
        *,
        previous_height_min: int = 0,
    ) -> Optional[YesterdayPremium]:
        if len(days) < 2:
            return None
        return self._premium_between(
            days[-1], days[-2], sets, heights_by_day, previous_height_min
        )

    def _premium_between(
        self,
        today: str,
        yesterday: str,
        sets: dict[str, Any],
        heights_by_day: dict[str, dict[str, int]],
        previous_height_min: int = 0,
    ) -> Optional[YesterdayPremium]:
        y_limits = sets["limit_up"].get(yesterday, set())
        if not y_limits:
            return YesterdayPremium(date=yesterday, measured_on=today)

        pct_by_code = self._pct_map(today)
        if not pct_by_code:
            return None
        today_limits = sets["limit_up"].get(today, set())
        today_broken = sets["broken"].get(today, set())
        today_down = sets["limit_down"].get(today, set())
        y_heights = heights_by_day.get(yesterday, {})

        values: list[float] = []
        up = down = flat = suspended = 0
        promoted = broken = limit_down = 0
        best_code = worst_code = None
        best_pct: Optional[float] = None
        worst_pct: Optional[float] = None
        promotion_counts: dict[int, list[int]] = {}

        for code in sorted(y_limits):
            height = int(y_heights.get(code, 1))
            if previous_height_min and height < previous_height_min:
                continue
            bucket = promotion_counts.setdefault(height, [0, 0])
            bucket[0] += 1
            if code in today_limits:
                bucket[1] += 1
                promoted += 1
            if code in today_broken:
                broken += 1
            if code in today_down:
                limit_down += 1

            pct = pct_by_code.get(code)
            if pct is None or pd.isna(pct):
                suspended += 1
                continue
            pct = float(pct)
            values.append(pct)
            if pct > 0:
                up += 1
            elif pct < 0:
                down += 1
            else:
                flat += 1
            if best_pct is None or pct > best_pct:
                best_pct, best_code = pct, code
            if worst_pct is None or pct < worst_pct:
                worst_pct, worst_code = pct, code

        if not values:
            return YesterdayPremium(
                date=yesterday,
                measured_on=today,
                count=len(y_limits),
                suspended=suspended,
            )

        series = pd.Series(values)
        promotions = tuple(
            PromotionStat(height=height, count=total, promoted=ok)
            for height, (total, ok) in sorted(promotion_counts.items())
        )
        return YesterdayPremium(
            date=yesterday,
            measured_on=today,
            count=len(y_limits),
            evaluated=len(values),
            suspended=suspended,
            avg_pct=float(series.mean()),
            median_pct=float(series.median()),
            up_count=up,
            down_count=down,
            flat_count=flat,
            up_ratio=up / len(values),
            promoted_count=promoted,
            promotion_rate=promoted / len(y_limits),
            broken_count=broken,
            limit_down_count=limit_down,
            best=best_code,
            best_pct=best_pct,
            worst=worst_code,
            worst_pct=worst_pct,
            promotions=promotions,
        )

    def _recent_days(
        self,
        days: list[str],
        sets: dict[str, Any],
        heights_by_day: dict[str, dict[str, int]],
    ) -> tuple[LadderDay, ...]:
        """Per-day structure summaries for the whole window (trend context)."""
        summaries: list[LadderDay] = []
        for index, day in enumerate(days):
            heights = heights_by_day.get(day, {})
            buckets = {
                bucket: sum(1 for h in heights.values() if bucket_label(h) == bucket)
                for bucket in ("首板", "二板", "三板", "四板及以上")
            }
            max_height = max(heights.values()) if heights else 0
            gaps = [h for h in range(1, max_height) if h not in set(heights.values())]

            premium = None
            high_board = None
            if index >= 1:
                premium = self._premium_between(
                    day, days[index - 1], sets, heights_by_day
                )
                high_board = self._premium_between(
                    day, days[index - 1], sets, heights_by_day, previous_height_min=3
                )

            bars = sets["by_day"].get(day)
            advance = decline = traded = 0
            total_amount = None
            if bars is not None and not bars.empty:
                pct = pd.to_numeric(bars["pct_chg"], errors="coerce")
                traded = int(pct.notna().sum())
                advance = int((pct > 0).sum())
                decline = int((pct < 0).sum())
                if "amount" in bars.columns:
                    amounts = pd.to_numeric(bars["amount"], errors="coerce")
                    if amounts.notna().any():
                        total_amount = float(amounts.sum(skipna=True))

            summaries.append(
                LadderDay(
                    date=day,
                    limit_up_count=len(heights),
                    limit_down_count=len(sets["limit_down"].get(day, set())),
                    broken_count=len(sets["broken"].get(day, set())),
                    reseal_count=len(sets["reseal"].get(day, set())),
                    max_height=max_height,
                    buckets=buckets,
                    promotion_rate=premium.promotion_rate if premium else None,
                    avg_premium=premium.avg_pct if premium else None,
                    up_ratio=premium.up_ratio if premium else None,
                    high_board_avg_pct=(
                        high_board.avg_pct
                        if high_board and high_board.evaluated
                        else None
                    ),
                    is_complete=not gaps and bool(heights),
                    gap_count=len(gaps),
                    traded_count=traded,
                    advance_count=advance,
                    decline_count=decline,
                    total_amount=total_amount,
                )
            )
        return tuple(summaries)

    def _industries(
        self,
        date: str,
        heights: dict[str, int],
        sets: dict[str, Any],
        info: dict[str, dict[str, Any]],
    ) -> tuple[IndustryLadder, ...]:
        by_industry: dict[str, dict[str, int]] = {}
        for code, height in heights.items():
            industry = str(info.get(code, {}).get("industry") or "") or "未分类"
            by_industry.setdefault(industry, {})[code] = height

        results: list[IndustryLadder] = []
        for industry, members in by_industry.items():
            max_height = max(members.values())
            counts: dict[int, int] = {}
            for height in members.values():
                counts[height] = counts.get(height, 0) + 1
            leaders = tuple(sorted(c for c, h in members.items() if h == max_height))
            mids = tuple(
                sorted(c for c, h in members.items() if 2 <= h < max_height)
            )
            catch_up = tuple(
                sorted(c for c, h in members.items() if h == 1)
            ) if max_height >= 3 else ()
            stocks = []
            for code in sorted(members, key=lambda c: (-members[c], c))[
                : self.stocks_per_level
            ]:
                height = members[code]
                if height == max_height:
                    role = "龙头候选"
                elif height >= 2:
                    role = "中军候选"
                elif max_height >= 3:
                    role = "补涨候选"
                else:
                    role = "普通"
                stock = self._stock(code, date, sets, info, height=height, role=role)
                if stock is not None:
                    stocks.append(stock)
            results.append(
                IndustryLadder(
                    name=industry,
                    limit_up_count=len(members),
                    max_height=max_height,
                    level_counts=counts,
                    leader_candidates=leaders,
                    mid_candidates=mids,
                    catch_up_candidates=catch_up,
                    stocks=tuple(stocks),
                )
            )
        results.sort(key=lambda i: (-i.max_height, -i.limit_up_count, i.name))
        return tuple(results[: self.top_industries])

    def _dominant_industry(
        self,
        industries: tuple[IndustryLadder, ...],
        highest: tuple[LadderStock, ...],
    ) -> Optional[str]:
        if not industries:
            return None
        top_height = industries[0].max_height
        if highest:
            top_height = max(top_height, max(s.height for s in highest))
        same_height = [i for i in industries if i.max_height == top_height]
        if not same_height:
            return industries[0].name
        return max(same_height, key=lambda i: (i.limit_up_count, i.name)).name

    def _broken_stocks(
        self,
        date: str,
        sets: dict[str, Any],
        info: dict[str, dict[str, Any]],
    ) -> tuple[LadderStock, ...]:
        stocks = []
        for code in sorted(sets["broken"].get(date, set()))[: self.stocks_per_level]:
            stock = self._stock(code, date, sets, info, height=0, role="炸板")
            if stock is not None:
                stocks.append(stock)
        return tuple(stocks)
