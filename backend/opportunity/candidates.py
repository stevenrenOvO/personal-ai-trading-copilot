"""Step-4 stage 1: candidate pool + four-layer hard elimination.

The pool is built from **hot sectors x the limit-up ladder x per-stock lane**
(Step 2), not from a technical breakout scan. Hard elimination runs *before*
any ranking and cannot be compensated by a score: a name that fails a rule is
out of the pool, with the reason recorded so the outcome can be audited.

Four layers (per the accepted Step-4 design):

1. 标的层  ST/*ST, 次新股, 除权/送转日, 当日停牌无成交, 一字跌停开盘
2. 板块层  与热点板块脱节, 孤立板, 板块退潮（炸板率高且无回封）
3. 梯队层  退潮/分化/冰点的高位板, 断层上方孤立高度, 同板块已被更高板压制
4. 结构层  跌破上一涨停日最低价, 放量滞涨, 昨日炸板后低开低走, 放天量未封板

Nothing here decides a buy: the output is a candidate list plus every
elimination that happened and every warning the survivors carry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from backend.board.ladder import LadderSnapshot, LimitLadderEngine, level_label
from backend.data.providers.base import DataProvider
from backend.market.engine import MarketEngine, MarketState
from backend.sector.engine import SectorEngine, SectorScore

# Layer-2 thresholds
HOT_SECTOR_TOP_N = 10
ISOLATED_BOARD_MAX = 1
SECTOR_DECAY_BROKEN_RATIO = 0.50

# Layer-3 thresholds
HIGH_BOARD_MIN = 5
ISOLATED_GAP_MIN = 2

# Layer-4 thresholds
STALL_VOLUME_RATIO = 1.5
STALL_GAIN_MAX = 1.0
HUGE_VOLUME_RATIO = 5.0
WEAK_TURN_LOSS = -3.0
VOLUME_LOOKBACK = 5

LAYERS = ("标的层", "板块层", "梯队层", "结构层")


@dataclass(frozen=True)
class Elimination:
    ts_code: str
    name: str
    layer: str
    rule: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "ts_code": self.ts_code,
            "name": self.name,
            "layer": self.layer,
            "rule": self.rule,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class Candidate:
    ts_code: str
    symbol: str = ""
    name: str = ""
    industry: str = ""
    height: int = 0
    structural_role: str = "普通"
    setup_hint: str = ""
    close: Optional[float] = None
    pct_chg: Optional[float] = None
    up_limit: Optional[float] = None
    down_limit: Optional[float] = None
    board_form: str = ""
    volume_ratio: Optional[float] = None
    is_st: bool = False
    amount: Optional[float] = None
    sector_rank: Optional[int] = None
    sector_limit_up_count: int = 0
    sector_relative_strength: Optional[float] = None
    sector_broken_ratio: Optional[float] = None
    reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    requires_minute_data: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "ts_code": self.ts_code,
            "symbol": self.symbol,
            "name": self.name,
            "industry": self.industry,
            "height": self.height,
            "level": level_label(self.height),
            "role": self.structural_role,
            "setup_hint": self.setup_hint,
            "close": self.close,
            "pct_chg": self.pct_chg,
            "up_limit": self.up_limit,
            "down_limit": self.down_limit,
            "board_form": self.board_form,
            "volume_ratio": self.volume_ratio,
            "is_st": self.is_st,
            "amount": self.amount,
            "sector_rank": self.sector_rank,
            "sector_limit_up_count": self.sector_limit_up_count,
            "sector_relative_strength": self.sector_relative_strength,
            "sector_broken_ratio": self.sector_broken_ratio,
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
            "requires_minute_data": self.requires_minute_data,
        }


@dataclass(frozen=True)
class CandidatePool:
    date: str
    available: bool = True
    stage: str = ""
    stage_rule: str = ""
    universe_size: int = 0
    limit_up_total: int = 0
    pool_size: int = 0
    layer_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    candidates: tuple[Candidate, ...] = ()
    eliminations: tuple[Elimination, ...] = ()
    notes: tuple[str, ...] = ()
    unavailable: tuple[str, ...] = ()

    @property
    def has_high_quality(self) -> bool:
        """False when every survivor still needs data we do not have."""
        return any(not c.requires_minute_data for c in self.candidates)

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "available": self.available,
            "stage": self.stage,
            "stage_rule": self.stage_rule,
            "universe_size": self.universe_size,
            "limit_up_total": self.limit_up_total,
            "pool_size": self.pool_size,
            "layer_counts": self.layer_counts,
            "has_high_quality": self.has_high_quality,
            "candidates": [c.to_dict() for c in self.candidates],
            "eliminations": [e.to_dict() for e in self.eliminations],
            "notes": list(self.notes),
            "unavailable": list(self.unavailable),
        }


def _f(value: Any) -> Optional[float]:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


class CandidatePoolEngine:
    """Build the short-term candidate pool for one confirmed trading day."""

    def __init__(
        self,
        provider: DataProvider,
        *,
        ladder_engine: Optional[LimitLadderEngine] = None,
        sector_engine: Optional[SectorEngine] = None,
        market_engine: Optional[MarketEngine] = None,
    ) -> None:
        self.provider = provider
        self.ladder_engine = ladder_engine or LimitLadderEngine(provider)
        self.sector_engine = sector_engine or SectorEngine(provider)
        self.market_engine = market_engine or MarketEngine(provider)

    # -- public --------------------------------------------------------
    def build(
        self,
        date: str,
        *,
        stage: Optional[str] = None,
        stage_rule: str = "",
        ladder: Optional[LadderSnapshot] = None,
        sectors: Optional[list[SectorScore]] = None,
        market: Optional[MarketState] = None,
    ) -> CandidatePool:
        ladder = ladder or self.ladder_engine.snapshot(date)
        if not ladder.available:
            return CandidatePool(
                date=date,
                available=False,
                notes=tuple(ladder.notes) or (f"{date} 无本地结构数据",),
                unavailable=("候选池无法构建：当日结构未确认",),
            )
        sectors = sectors if sectors is not None else self.sector_engine.ranked(date)
        market = market or self.market_engine.calculate(date)
        if stage is None:
            from backend.emotion.engine import EmotionEngine

            stage = EmotionEngine(self.provider).calculate(
                date, market=market, ladder=ladder
            ).emotion_cycle

        info = self._stock_info()
        bars, limits = self._market_data(date, ladder)
        sector_index = {s.name: s for s in sectors}
        sector_stats = self._sector_stats(ladder, info)

        raw: dict[str, dict[str, Any]] = {}
        for level in ladder.levels:
            for code in level.all_codes:
                raw[code] = {
                    "height": level.height,
                    "setup_hint": "首板" if level.height == 1 else "连板接力",
                }
        for code in self._weak_to_strong_candidates(date, ladder, bars):
            raw.setdefault(code, {"height": 0, "setup_hint": "弱转强(待确认)"})
        # Names the ladder already excluded (次新/除权) are *not* limit-up in its
        # eyes, but they still belong to the universe: putting them in the raw
        # pool lets the 标的层 record an explicit elimination instead of having
        # them vanish silently.
        for code in ladder.excluded_new_listing_codes:
            raw.setdefault(code, {"height": 0, "setup_hint": "次新"})
        for code in ladder.excluded_corporate_action_codes:
            raw.setdefault(code, {"height": 0, "setup_hint": "除权/送转"})

        counts = {layer: {"considered": 0, "eliminated": 0, "passed": 0} for layer in LAYERS}
        eliminations: list[Elimination] = []
        survivors: list[str] = sorted(raw)

        for layer in LAYERS:
            kept: list[str] = []
            for code in survivors:
                counts[layer]["considered"] += 1
                reason = self._check_layer(
                    layer,
                    code,
                    raw[code],
                    info,
                    bars,
                    limits,
                    sector_index,
                    sector_stats,
                    ladder,
                    stage,
                    date,
                )
                if reason is None:
                    kept.append(code)
                    continue
                rule, detail = reason
                counts[layer]["eliminated"] += 1
                eliminations.append(
                    Elimination(
                        ts_code=code,
                        name=info.get(code, {}).get("name", ""),
                        layer=layer,
                        rule=rule,
                        detail=detail,
                    )
                )
            counts[layer]["passed"] = len(kept)
            survivors = kept

        candidates = [
            self._to_candidate(
                code, raw[code], info, bars, limits, sector_index, sector_stats, date
            )
            for code in survivors
        ]
        # Review order only (ranking is stage 3): higher board, hotter sector,
        # stronger move, then code for stability.
        candidates.sort(
            key=lambda c: (
                -c.height,
                c.sector_rank if c.sector_rank is not None else 999,
                -(c.pct_chg or 0.0),
                c.ts_code,
            )
        )

        notes = list(ladder.notes)
        if candidates and not any(not c.requires_minute_data for c in candidates):
            notes.append(
                "本次候选全部需要分钟数据确认（弱转强/半路），当前未接入 → 视为今日无高质量机会"
            )
        unavailable = ["封单量/涨停时间/竞价过程：未接入，未参与筛选"]
        if not bars.attrs.get("amount_verified", True):
            unavailable.append("成交额（回补日数据源未提供）：未验证，未参与淘汰")

        return CandidatePool(
            date=date,
            available=True,
            stage=stage,
            stage_rule=stage_rule,
            universe_size=ladder.universe_size,
            limit_up_total=ladder.limit_up_count,
            pool_size=len(survivors),
            layer_counts=counts,
            candidates=tuple(candidates),
            eliminations=tuple(eliminations),
            notes=tuple(notes),
            unavailable=tuple(unavailable),
        )

    # -- data ----------------------------------------------------------
    def _stock_info(self) -> dict[str, dict[str, Any]]:
        try:
            basic = self.provider.stock_basic(list_status="L")
        except Exception:  # noqa: BLE001
            basic = self.provider.stock_basic()
        info: dict[str, dict[str, Any]] = {}
        if basic is None or basic.empty:
            return info
        for _, row in basic.iterrows():
            code = str(row["ts_code"])
            name = str(row.get("name") or "")
            info[code] = {
                "symbol": str(row.get("symbol") or ""),
                "name": name,
                "industry": str(row.get("industry") or ""),
                "list_date": str(row.get("list_date") or ""),
                "is_st": "ST" in name.upper(),
            }
        return info

    def _market_data(
        self, date: str, ladder: LadderSnapshot
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        codes = sorted(
            {code for level in ladder.levels for code in level.all_codes}
            | set(ladder.prev_broken_codes)
        )
        if not codes:
            empty = pd.DataFrame()
            empty.attrs["amount_verified"] = True
            return empty, pd.DataFrame()
        days = [d.date for d in ladder.recent][-(VOLUME_LOOKBACK + 1) :] or [date]
        bars = self.provider.daily(ts_codes=codes, start_date=days[0], end_date=date)
        limits = self.provider.stk_limit(ts_codes=codes, trade_date=date)
        if bars is None:
            bars = pd.DataFrame()
        amount_verified = True
        if not bars.empty and "amount" in bars.columns:
            today = bars[bars["trade_date"].astype(str) == date]
            amount_verified = bool(
                pd.to_numeric(today["amount"], errors="coerce").notna().any()
            )
        bars.attrs["amount_verified"] = amount_verified
        return bars, limits if limits is not None else pd.DataFrame()

    def _weak_to_strong_candidates(
        self, date: str, ladder: LadderSnapshot, bars: pd.DataFrame
    ) -> set[str]:
        """Yesterday broke the board, today strong (>= +5%)."""
        if bars.empty or not ladder.prev_broken_codes:
            return set()
        today = bars[bars["trade_date"].astype(str) == date]
        if today.empty:
            return set()
        pct = pd.to_numeric(today["pct_chg"], errors="coerce")
        strong = set(today.loc[pct >= 5.0, "ts_code"].astype(str))
        return strong & set(ladder.prev_broken_codes)

    def _sector_stats(
        self, ladder: LadderSnapshot, info: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        """Per-industry limit-up / broken / reseal counts for the day."""
        stats: dict[str, dict[str, Any]] = {}

        def bucket(code: str) -> dict[str, Any]:
            industry = info.get(code, {}).get("industry") or "未分类"
            return stats.setdefault(
                industry, {"limit_up": 0, "broken": 0, "reseal": 0, "max_height": 0}
            )

        for level in ladder.levels:
            for code in level.all_codes:
                entry = bucket(code)
                entry["limit_up"] += 1
                entry["max_height"] = max(entry["max_height"], level.height)
        for code in ladder.broken_codes:
            bucket(code)["broken"] += 1
        for code in ladder.reseal_codes:
            bucket(code)["reseal"] += 1
        return stats

    # -- layers --------------------------------------------------------
    def _check_layer(
        self,
        layer: str,
        code: str,
        meta: dict[str, Any],
        info: dict[str, dict[str, Any]],
        bars: pd.DataFrame,
        limits: pd.DataFrame,
        sector_index: dict[str, SectorScore],
        sector_stats: dict[str, dict[str, Any]],
        ladder: LadderSnapshot,
        stage: str,
        date: str,
    ) -> Optional[tuple[str, str]]:
        row = self._row(bars, code, date)
        if layer == "标的层":
            return self._instrument_layer(code, row, info, limits, ladder)
        if layer == "板块层":
            return self._sector_layer(code, info, sector_index, sector_stats)
        if layer == "梯队层":
            return self._ladder_layer(code, meta, info, ladder, stage, sector_stats)
        return self._structure_layer(code, meta, info, bars, row, limits, ladder, date)

    @staticmethod
    def _row(bars: pd.DataFrame, code: str, date: str) -> Optional[pd.Series]:
        if bars is None or bars.empty:
            return None
        match = bars[
            (bars["ts_code"].astype(str) == code)
            & (bars["trade_date"].astype(str) == date)
        ]
        return None if match.empty else match.iloc[0]

    @staticmethod
    def _limit_value(limits: pd.DataFrame, code: str, column: str) -> Optional[float]:
        if limits is None or limits.empty or column not in limits.columns:
            return None
        match = limits[limits["ts_code"].astype(str) == code]
        return None if match.empty else _f(match.iloc[0][column])

    def _instrument_layer(self, code, row, info, limits, ladder) -> Optional[tuple[str, str]]:
        entry = info.get(code, {})
        name = entry.get("name", "")
        if entry.get("is_st"):
            return ("ST/*ST 标的", f"{name} 属风险警示股（5% 限制），不纳入超短线候选")
        if code in set(ladder.excluded_new_listing_codes):
            return (
                "次新股",
                f"{name} 上市不足 5 个交易日，涨跌幅限制不适用，结构不可比",
            )
        if code in set(ladder.excluded_corporate_action_codes):
            return (
                "除权/送转日",
                f"{name} 当日涨跌幅超出该板块限制，疑似除权/送转，基准价不可靠",
            )
        if row is None:
            return ("当日停牌/无成交", f"{name} 当日无成交数据")
        down = self._limit_value(limits, code, "down_limit")
        open_price = _f(row.get("open"))
        if down is not None and open_price is not None and open_price <= down:
            return ("一字跌停开盘", f"{name} 开盘 {open_price} ≤ 跌停价 {down}")
        return None

    def _sector_layer(self, code, info, sector_index, sector_stats) -> Optional[tuple[str, str]]:
        industry = info.get(code, {}).get("industry") or "未分类"
        name = info.get(code, {}).get("name", "")
        sector = sector_index.get(industry)
        stats = sector_stats.get(industry, {"limit_up": 0, "broken": 0, "reseal": 0})
        rank = sector.rank if sector else None
        relative = sector.relative_strength if sector else None
        # 热点口径与设计文档一致：板块涨停家数 ≥2 或 相对强度 > 0（额外把排名前列
        # 也算作热点）。之前实现只用了「排名 ≤10 或 相对强度>0」，会把当日最高板
        # 所在板块（涨停 2 家但相对强度为负）误判为"脱节"。
        hot = (
            stats["limit_up"] >= 2
            or (relative is not None and relative > 0)
            or (rank is not None and rank <= HOT_SECTOR_TOP_N)
        )
        if not hot:
            return (
                "与热点板块脱节",
                f"{name} 所属『{industry}』排名 {rank or '未入榜'}、相对强度 "
                f"{'—' if relative is None else f'{relative:+.2f}'}、涨停 {stats['limit_up']} 家，"
                "不在当日热点主线",
            )
        if stats["limit_up"] <= ISOLATED_BOARD_MAX and (relative is None or relative <= 0):
            return (
                "孤立板",
                f"{name} 所属『{industry}』当日仅 {stats['limit_up']} 家涨停且板块无相对强度",
            )
        attempt = stats["limit_up"] + stats["broken"]
        if attempt:
            broken_ratio = stats["broken"] / attempt
            if broken_ratio >= SECTOR_DECAY_BROKEN_RATIO and stats["reseal"] == 0:
                return (
                    "板块退潮",
                    f"『{industry}』炸板率 {broken_ratio:.0%}"
                    f"（炸板 {stats['broken']} / 涨停 {stats['limit_up']}）且无回封",
                )
        return None

    def _ladder_layer(self, code, meta, info, ladder, stage, sector_stats) -> Optional[tuple[str, str]]:
        height = int(meta.get("height") or 0)
        name = info.get(code, {}).get("name", "")
        industry = info.get(code, {}).get("industry") or "未分类"
        if height >= HIGH_BOARD_MIN and stage in ("分化", "退潮", "冰点"):
            return (
                "高位板在退潮/分化阶段",
                f"{name} 已 {height} 板，当前阶段『{stage}』，高位股不纳入候选",
            )
        gaps = tuple(ladder.gaps or ())
        if len(gaps) >= ISOLATED_GAP_MIN and height:
            same_height = next((l for l in ladder.levels if l.height == height), None)
            if (
                same_height is not None
                and same_height.count <= 1
                and height > max(gaps)
            ):
                return (
                    "断层上方孤立高度",
                    f"{name} 在 {height} 板且该高度仅此 1 只，下方断层 {list(gaps)}，梯队无承接",
                )
        stats = sector_stats.get(industry)
        # 只在「中间高度」上生效：2 板及以上若同板块已有更高板，既非龙头也非
        # 新启动，属被压制；首板是补涨/新启动位置，设计里明确要保留，不能在此淘汰。
        if stats and height >= 2 and stats.get("max_height", 0) > height:
            return (
                "同板块已被更高板压制",
                f"{name} 为 {height} 板，同板块『{industry}』已有 {stats['max_height']} 板，"
                "高度不占优",
            )
        return None

    def _structure_layer(self, code, meta, info, bars, row, limits, ladder, date):
        name = info.get(code, {}).get("name", "")
        height = int(meta.get("height") or 0)
        if row is None:
            return ("当日无成交", f"{name} 当日无成交数据")
        close = _f(row.get("close"))
        pct = _f(row.get("pct_chg"))
        if code in set(ladder.prev_broken_codes) and pct is not None and pct <= WEAK_TURN_LOSS:
            return (
                "昨日炸板后今日低开低走",
                f"{name} 昨日炸板、今日收盘 {pct:+.2f}%，结构未修复",
            )
        volume_ratio, _ = self._volume_facts(bars, code, date)
        if volume_ratio is not None and volume_ratio >= HUGE_VOLUME_RATIO:
            up = self._limit_value(limits, code, "up_limit")
            if up is not None and close is not None and close < up:
                return (
                    "放天量未封板",
                    f"{name} 量能 {volume_ratio:.1f}x 但收盘未封板，分歧过大",
                )
        if (
            volume_ratio is not None
            and volume_ratio >= STALL_VOLUME_RATIO
            and pct is not None
            and pct <= STALL_GAIN_MAX
            and height > 0
        ):
            return (
                "放量滞涨",
                f"{name} 量能 {volume_ratio:.1f}x 而涨幅仅 {pct:+.2f}%，量价不匹配",
            )
        # Reachable for ladder names (2板+) and for weak-to-strong names, whose
        # close today can genuinely sit below the previous limit-up day's low.
        # On a pure limit-up pool this rule is usually inert because a stock
        # closing at its limit is above that low by construction.
        revisitable = height >= 2 or str(meta.get("setup_hint", "")).startswith("弱转强")
        if revisitable:
            prev_low = self._previous_limit_up_low(bars, code, date)
            if prev_low is not None and close is not None and close < prev_low:
                return (
                    "跌破上一涨停日最低价",
                    f"{name} 收盘 {close} 低于上一涨停日最低价 {prev_low}",
                )
        return None

    # -- helpers -------------------------------------------------------
    @staticmethod
    def _volume_facts(
        bars: pd.DataFrame, code: str, date: str
    ) -> tuple[Optional[float], Optional[float]]:
        """(今日量 / 前 N 日均量, 前一交易日涨幅)"""
        if bars is None or bars.empty:
            return None, None
        series = bars[bars["ts_code"].astype(str) == code].sort_values("trade_date")
        dates = series["trade_date"].astype(str).tolist()
        if date not in dates:
            return None, None
        index = dates.index(date)
        if index < 1:
            return None, None
        vols = pd.to_numeric(series["vol"], errors="coerce")
        today_vol = vols.iloc[index]
        window = vols.iloc[max(0, index - VOLUME_LOOKBACK) : index]
        baseline = window.mean() if not window.empty else None
        ratio = None
        if baseline and not pd.isna(baseline) and float(baseline) > 0 and not pd.isna(today_vol):
            ratio = float(today_vol) / float(baseline)
        pcts = pd.to_numeric(series["pct_chg"], errors="coerce")
        prev_pct = (
            None if pd.isna(pcts.iloc[index - 1]) else float(pcts.iloc[index - 1])
        )
        return ratio, prev_pct

    def _previous_limit_up_low(
        self, bars: pd.DataFrame, code: str, date: str
    ) -> Optional[float]:
        """Low of the most recent earlier day on which the code closed limit-up."""
        if bars is None or bars.empty:
            return None
        series = bars[bars["ts_code"].astype(str) == code].sort_values("trade_date")
        dates = series["trade_date"].astype(str).tolist()
        if date not in dates:
            return None
        for index in range(dates.index(date) - 1, -1, -1):
            row = series.iloc[index]
            limits = self.provider.stk_limit(ts_codes=[code], trade_date=dates[index])
            up = self._limit_value(limits, code, "up_limit")
            close = _f(row.get("close"))
            if up is not None and close is not None and close >= up:
                return _f(row.get("low"))
        return None

    def _to_candidate(
        self, code, meta, info, bars, limits, sector_index, sector_stats, date
    ) -> Candidate:
        row = self._row(bars, code, date)
        entry = info.get(code, {})
        industry = entry.get("industry") or "未分类"
        sector = sector_index.get(industry)
        stats = sector_stats.get(
            industry, {"limit_up": 0, "broken": 0, "reseal": 0, "max_height": 0}
        )
        attempt = stats["limit_up"] + stats["broken"]
        height = int(meta.get("height") or 0)
        up = self._limit_value(limits, code, "up_limit")
        down = self._limit_value(limits, code, "down_limit")
        close = _f(row.get("close")) if row is not None else None
        open_price = _f(row.get("open")) if row is not None else None
        low = _f(row.get("low")) if row is not None else None
        volume_ratio, _ = self._volume_facts(bars, code, date)

        warnings: list[str] = []
        if row is not None and _f(row.get("amount")) is None:
            warnings.append("成交额未接入（该交易日数据源未提供），流动性未验证")

        return Candidate(
            ts_code=code,
            symbol=entry.get("symbol", ""),
            name=entry.get("name", ""),
            industry=industry,
            height=height,
            structural_role=self._role_for(height, industry, stats),
            setup_hint=meta.get("setup_hint", ""),
            close=close,
            pct_chg=_f(row.get("pct_chg")) if row is not None else None,
            up_limit=up,
            down_limit=down,
            board_form=self._board_form(open_price, low, close, up),
            volume_ratio=None if volume_ratio is None else round(volume_ratio, 2),
            is_st=bool(entry.get("is_st")),
            amount=_f(row.get("amount")) if row is not None else None,
            sector_rank=sector.rank if sector else None,
            sector_limit_up_count=stats["limit_up"],
            sector_relative_strength=(
                round(sector.relative_strength, 4) if sector else None
            ),
            sector_broken_ratio=(round(stats["broken"] / attempt, 4) if attempt else None),
            reasons=tuple(self._reasons(height, industry, stats, sector)),
            warnings=tuple(warnings),
            requires_minute_data=meta.get("setup_hint", "").startswith("弱转强"),
        )

    @staticmethod
    def _board_form(open_price, low, close, up) -> str:
        if up is None or close is None:
            return "未接入"
        if close < up:
            return "未封板"
        if open_price is not None and low is not None and open_price >= up and low >= up:
            return "一字板/未开板"
        if open_price is not None and open_price >= up:
            return "开盘封板"
        if low is not None and low < up:
            return "盘中触板回封"
        return "封板"

    @staticmethod
    def _role_for(height: int, industry: str, stats: dict[str, Any]) -> str:
        if height and height == stats.get("max_height", 0) and stats.get("limit_up", 0) >= 2:
            return "龙头候选"
        if height >= 2:
            return "中军候选"
        if stats.get("max_height", 0) >= 3:
            return "补涨候选"
        return "普通"

    @staticmethod
    def _reasons(height: int, industry: str, stats: dict[str, Any], sector) -> list[str]:
        reasons: list[str] = []
        if stats.get("limit_up", 0) >= 2:
            reasons.append(f"板块『{industry}』当日涨停 {stats['limit_up']} 家，具备板块效应")
        if sector is not None:
            reasons.append(
                f"板块强度排名第 {sector.rank}，相对强度 {sector.relative_strength:+.2f}"
            )
        if height >= 2:
            reasons.append(f"连板高度 {height} 板，处于梯队内")
        elif height == 1:
            reasons.append("首板结构，处于板块梯队起点")
        else:
            reasons.append("昨日炸板、今日强势（弱转强待确认）")
        return reasons
