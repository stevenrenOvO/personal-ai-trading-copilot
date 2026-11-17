"""Personal risk profile for the user's own (small) trading account.

This is separate from the 1,000,000 research paper account. It records the
user's own rules (stated 2026-11) so sizing and pre-trade checks never depend
on hard-coded numbers:

* risk per trade 1% of capital by default, never more than 2%
* single stock (or sector) at most 20% of capital
* first entry 10%-15% of capital
* reference stop distance 5%-8%
* total exposure capped by market regime (range 30%-50%, weak/unclear <=20%)

Stored as JSON under ``<data root>/personal/risk_profile.json``. A missing file
means "defaults", reported as such instead of pretending the user saved them.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from backend.data.atomic import atomic_write, file_lock

# Market-state labels produced by backend.market.engine.classify_market.
DEFAULT_REGIME_BY_STATE: dict[str, str] = {
    "亢奋": "strong",
    "强势": "strong",
    "偏强": "range",
    "中性": "range",
    "偏弱": "weak",
    "弱势": "weak",
    "冰点": "weak",
}

REGIME_LABELS = {"strong": "趋势向上", "range": "震荡", "weak": "弱势/趋势不明"}


@dataclass
class RiskProfile:
    capital: float = 12_000.0
    risk_per_trade: float = 0.01
    max_risk_per_trade: float = 0.02
    max_single_weight: float = 0.20
    first_entry_min: float = 0.10
    first_entry_max: float = 0.15
    max_stop_pct: float = 0.08
    # Orders are placed for the next session without watching the market:
    # a buy limit sits at most this far above the last close; a higher open
    # means the buy is skipped.
    chase_limit: float = 0.02
    # The user's rules state 30%-50% for a ranging market and <=20% when weak
    # or unclear. No rule was stated for a clear uptrend, so it defaults to the
    # range ceiling until the user sets it explicitly.
    exposure_caps: dict[str, float] = field(
        default_factory=lambda: {"strong": 0.50, "range": 0.50, "weak": 0.20}
    )
    lot_size: int = 100
    commission_rate: float = 0.00025
    min_commission: float = 5.0
    stamp_tax_rate: float = 0.0005
    updated_at: Optional[str] = None

    def validate(self) -> "RiskProfile":
        def ratio(name: str, value: Any) -> None:
            if not isinstance(value, (int, float)) or isinstance(value, bool) \
                    or not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} 须为 0 到 1 之间的比例（如 0.01 表示 1%）")

        if not isinstance(self.capital, (int, float)) or isinstance(self.capital, bool) \
                or not math.isfinite(self.capital) or self.capital <= 0:
            raise ValueError("capital 须为正数（账户资金，元）")
        for name in ("risk_per_trade", "max_risk_per_trade", "max_single_weight",
                     "first_entry_min", "first_entry_max", "max_stop_pct", "chase_limit"):
            ratio(name, getattr(self, name))
        if self.risk_per_trade > self.max_risk_per_trade:
            raise ValueError("risk_per_trade 不能大于 max_risk_per_trade")
        if self.first_entry_min > self.first_entry_max:
            raise ValueError("first_entry_min 不能大于 first_entry_max")
        if self.first_entry_max > self.max_single_weight:
            raise ValueError("first_entry_max 不能大于 max_single_weight")
        if not isinstance(self.exposure_caps, dict) or set(self.exposure_caps) != set(REGIME_LABELS):
            raise ValueError("exposure_caps 须包含 strong / range / weak 三项")
        for key, value in self.exposure_caps.items():
            ratio(f"exposure_caps.{key}", value)
        if not isinstance(self.lot_size, int) or isinstance(self.lot_size, bool) or self.lot_size <= 0:
            raise ValueError("lot_size 须为正整数")
        for name in ("commission_rate", "min_commission", "stamp_tax_rate"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) \
                    or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} 须为非负数")
        return self

    def regime_for_state(self, market_state: Optional[str]) -> Optional[str]:
        if market_state in REGIME_LABELS:
            return market_state
        return DEFAULT_REGIME_BY_STATE.get(market_state or "")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["exposure_caps"] = dict(self.exposure_caps)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RiskProfile":
        if not isinstance(data, dict):
            raise ValueError("风控参数须为 JSON 对象")
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"未知的风控参数: {', '.join(unknown)}")
        merged = cls().to_dict()
        for key, value in data.items():
            if key == "exposure_caps" and isinstance(value, dict):
                caps = dict(merged["exposure_caps"])
                caps.update(value)
                merged[key] = caps
            else:
                merged[key] = value
        if isinstance(merged.get("lot_size"), float) and merged["lot_size"].is_integer():
            merged["lot_size"] = int(merged["lot_size"])
        return cls(**merged).validate()


class RiskProfileStore:
    def __init__(self, root: Path) -> None:
        self.path = Path(root) / "personal" / "risk_profile.json"

    def load(self) -> tuple[RiskProfile, str]:
        """Return the profile and its source: ``saved`` or ``default``."""
        if not self.path.exists():
            return RiskProfile(), "default"
        data = json.loads(self.path.read_text(encoding="utf-8"))
        return RiskProfile.from_dict(data), "saved"

    def update(self, changes: dict[str, Any]) -> RiskProfile:
        if not isinstance(changes, dict) or not changes:
            raise ValueError("请提供要修改的风控参数")
        changes = {k: v for k, v in changes.items() if k != "updated_at"}
        with file_lock(self.path):
            current, _ = self.load()
            data = current.to_dict()
            for key, value in changes.items():
                if key == "exposure_caps" and isinstance(value, dict):
                    data["exposure_caps"] = {**data["exposure_caps"], **value}
                else:
                    data[key] = value
            data["updated_at"] = datetime.now().isoformat(timespec="seconds")
            profile = RiskProfile.from_dict(data)
            atomic_write(self.path, json.dumps(profile.to_dict(), ensure_ascii=False,
                                               allow_nan=False, indent=2) + "\n")
        return profile
