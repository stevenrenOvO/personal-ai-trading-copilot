"""Sector-level strength model.

Builds a ranked sector view from normalized daily bars. The score is a
weighted blend of relative strength, breadth, limit-ups, persistence, leader
performance and internal diffusion, all computed from data so the ranking can
be explained.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

from backend.data.numeric import safe_mean, safe_sum
from backend.data.providers.base import DataProvider


@dataclass(frozen=True)
class SectorState:
    name: str
    date: str
    avg_change: float
    avg_pct_chg: float
    total_volume: float
    total_amount: Optional[float]
    advance_count: int
    decline_count: int
    stock_codes: tuple[str, ...]
    limit_up_count: int = 0
    leader: Optional[str] = None
    leader_name: Optional[str] = None
    leader_pct_chg: Optional[float] = None
    relative_strength: float = 0.0
    persistence: float = 0.0
    diffusion: float = 0.0
    weakening: float = 0.0
    score: float = 0.0
    risk: str = "medium"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "date": self.date,
            "avg_change": self.avg_change,
            "avg_pct_chg": self.avg_pct_chg,
            "total_volume": self.total_volume,
            "total_amount": self.total_amount,
            "advance_count": self.advance_count,
            "decline_count": self.decline_count,
            "stock_codes": list(self.stock_codes),
            "limit_up_count": self.limit_up_count,
            "leader": self.leader,
            "leader_name": self.leader_name,
            "leader_pct_chg": self.leader_pct_chg,
            "relative_strength": self.relative_strength,
            "persistence": self.persistence,
            "diffusion": self.diffusion,
            "weakening": self.weakening,
            "score": self.score,
            "risk": self.risk,
        }


@dataclass(frozen=True)
class SectorScore:
    name: str
    date: str
    rank: int
    score: float
    relative_strength: float
    breadth: float
    limit_up_count: int
    leader: Optional[str]
    leader_name: Optional[str]
    leader_pct_chg: Optional[float]
    persistence: float
    diffusion: float
    weakening: float
    risk: str
    avg_pct_chg: float
    stock_codes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "date": self.date,
            "rank": self.rank,
            "score": self.score,
            "relative_strength": self.relative_strength,
            "breadth": self.breadth,
            "limit_up_count": self.limit_up_count,
            "leader": self.leader,
            "leader_name": self.leader_name,
            "leader_pct_chg": self.leader_pct_chg,
            "persistence": self.persistence,
            "diffusion": self.diffusion,
            "weakening": self.weakening,
            "risk": self.risk,
            "avg_pct_chg": self.avg_pct_chg,
            "stock_codes": list(self.stock_codes),
        }


def _risk_for_score(score: float) -> str:
    if score >= 75:
        return "low"
    if score >= 55:
        return "medium"
    return "high"


class SectorEngine:
    """Aggregate daily bars into a ranked sector view."""

    def __init__(
        self,
        provider: DataProvider,
        sector_map: Optional[dict[str, list[str]]] = None,
    ) -> None:
        self.provider = provider
        self._sector_map = dict(sector_map) if sector_map else None
        self._name_map: dict[str, str] | None = None

    def _names(self) -> dict[str, str]:
        if self._name_map is not None:
            return self._name_map
        try:
            basic = self.provider.stock_basic(list_status="L")
        except Exception:
            basic = self.provider.stock_basic()
        mapping: dict[str, str] = {}
        if not basic.empty and "ts_code" in basic.columns and "name" in basic.columns:
            for _, row in basic.iterrows():
                mapping[str(row["ts_code"])] = str(row.get("name", "") or "")
        self._name_map = mapping
        return mapping

    def _load_sector_map(self) -> dict[str, list[str]]:
        if self._sector_map is not None:
            return self._sector_map

        basic = self.provider.stock_basic(list_status="L")
        if basic.empty:
            self._sector_map = {}
            return self._sector_map

        grouped = basic.groupby("industry", sort=False)["ts_code"].apply(list).to_dict()
        self._sector_map = {
            str(sector): [str(code) for code in codes]
            for sector, codes in grouped.items()
            if sector
        }
        return self._sector_map

    def calculate(
        self,
        date: str,
        sector_name: Optional[str] = None,
        previous_date: Optional[str] = None,
    ) -> list[SectorState]:
        """Return legacy sector states (sorted by average change)."""
        scored = self._score_sectors(date, sector_name, previous_date)
        return sorted(
            [s["state"] for s in scored],
            key=lambda state: state.avg_pct_chg,
            reverse=True,
        )

    def ranked(
        self,
        date: str,
        sector_name: Optional[str] = None,
        previous_date: Optional[str] = None,
        top_n: Optional[int] = None,
    ) -> list[SectorScore]:
        """Return ranked sector scores, strongest first."""
        scored = self._score_sectors(date, sector_name, previous_date)
        scores = [s["score"] for s in scored]
        scores.sort(key=lambda item: item.score, reverse=True)
        if top_n is not None:
            scores = scores[:top_n]
        for rank, item in enumerate(scores, start=1):
            object.__setattr__(item, "rank", rank)
        return scores

    def _score_sectors(
        self,
        date: str,
        sector_name: Optional[str],
        previous_date: Optional[str],
    ) -> list[dict[str, Any]]:
        sector_map = self._load_sector_map()
        if not sector_map:
            return []

        all_codes = sorted({code for codes in sector_map.values() for code in codes})
        today = self.provider.daily(ts_codes=all_codes, start_date=date, end_date=date)
        if today.empty:
            return []

        if previous_date is None:
            previous_date = self._guess_previous_date(all_codes, date)
        yesterday = (
            self.provider.daily(
                ts_codes=all_codes, start_date=previous_date, end_date=previous_date
            )
            if previous_date
            else pd.DataFrame()
        )

        limit_df = self.provider.stk_limit(ts_codes=all_codes, trade_date=date)
        market_avg = safe_mean(today["pct_chg"]) or 0.0
        names = self._names()

        results: list[dict[str, Any]] = []
        for sector, codes in sector_map.items():
            if sector_name and sector != sector_name:
                continue

            sector_data = today[today["ts_code"].isin(codes)]
            if sector_data.empty:
                continue

            avg_pct = safe_mean(sector_data["pct_chg"]) or 0.0
            avg_change = safe_mean(sector_data["change"]) or 0.0
            total_volume = safe_sum(sector_data["vol"]) or 0.0
            total_amount = safe_sum(sector_data["amount"])
            advance = int((sector_data["change"] > 0).sum())
            decline = int((sector_data["change"] < 0).sum())
            total = len(sector_data)

            limit_up_count = self._count_limit_ups(sector_data, limit_df)
            leader_row = sector_data.loc[sector_data["pct_chg"].idxmax()]
            leader = str(leader_row["ts_code"])
            leader_name = names.get(leader, "")
            leader_pct = float(leader_row["pct_chg"])

            relative_strength = avg_pct - market_avg
            breadth = advance / total if total else 0.0
            diffusion = advance / total if total else 0.0
            weakening = decline / total if total else 0.0
            persistence = self._persistence(
                codes, yesterday, avg_pct, previous_date
            )

            score = self._sector_score(
                relative_strength=relative_strength,
                breadth=breadth,
                limit_up_count=limit_up_count,
                persistence=persistence,
                leader_pct=leader_pct,
                total=total,
            )
            risk = _risk_for_score(score)

            state = SectorState(
                name=sector,
                date=date,
                avg_change=avg_change,
                avg_pct_chg=avg_pct,
                total_volume=total_volume,
                total_amount=total_amount,
                advance_count=advance,
                decline_count=decline,
                stock_codes=tuple(sorted(set(codes))),
                limit_up_count=limit_up_count,
                leader=leader,
                leader_name=leader_name,
                leader_pct_chg=leader_pct,
                relative_strength=round(relative_strength, 4),
                persistence=round(persistence, 4),
                diffusion=round(diffusion, 4),
                weakening=round(weakening, 4),
                score=round(score, 2),
                risk=risk,
            )
            results.append({"state": state, "score": _to_sector_score(state, 0)})
        return results

    def _guess_previous_date(self, all_codes: list[str], date: str) -> Optional[str]:
        daily = self.provider.daily(ts_codes=all_codes)
        if daily.empty:
            return None
        dates = sorted(str(d) for d in daily["trade_date"].unique() if str(d) < date)
        return dates[-1] if dates else None

    def _count_limit_ups(self, sector_data: pd.DataFrame, limit_df: pd.DataFrame) -> int:
        if limit_df.empty or "up_limit" not in limit_df.columns:
            return 0
        merged = sector_data.merge(
            limit_df[["ts_code", "trade_date", "up_limit"]],
            on=["ts_code", "trade_date"],
            how="left",
        )
        return int((merged["close"] >= merged["up_limit"]).sum())

    def _persistence(
        self,
        codes: list[str],
        yesterday: pd.DataFrame,
        today_avg: float,
        previous_date: Optional[str],
    ) -> float:
        if yesterday.empty:
            return 0.0
        y_sector = yesterday[yesterday["ts_code"].isin(codes)]
        if y_sector.empty:
            return 0.0
        prev_avg = float(y_sector["pct_chg"].mean(skipna=True))
        delta = today_avg - prev_avg
        # Bounded logistic-ish transform to a 0..1 persistence score.
        return max(0.0, min(1.0, 0.5 + delta / 10.0))

    def _sector_score(
        self,
        *,
        relative_strength: float,
        breadth: float,
        limit_up_count: int,
        persistence: float,
        leader_pct: float,
        total: int,
    ) -> float:
        score = 50.0
        score += relative_strength * 2.0
        score += (breadth - 0.5) * 40.0
        score += min(12.0, limit_up_count * 2.0)
        score += (persistence - 0.5) * 20.0
        score += max(-10.0, min(10.0, leader_pct * 0.5))
        return max(0.0, min(100.0, score))


def _to_sector_score(state: SectorState, rank: int) -> SectorScore:
    return SectorScore(
        name=state.name,
        date=state.date,
        rank=rank,
        score=state.score,
        relative_strength=state.relative_strength,
        breadth=state.diffusion,
        limit_up_count=state.limit_up_count,
        leader=state.leader,
        leader_name=state.leader_name,
        leader_pct_chg=state.leader_pct_chg,
        persistence=state.persistence,
        diffusion=state.diffusion,
        weakening=state.weakening,
        risk=state.risk,
        avg_pct_chg=state.avg_pct_chg,
        stock_codes=state.stock_codes,
    )
