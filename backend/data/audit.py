"""Data-consistency audit for the local full-market history (V1.0 收尾).

Answers "why does our number differ from 同花顺" by checking what can be
checked internally, and by classifying the known external differences instead
of silently bending rules to match another vendor.

Classification used in the report:

* 数据源差异     the vendor simply publishes a different value/field
* 统计口径差异   same data, different denominator/definition
* 复权差异       adjusted vs unadjusted prices
* 时间差异       timestamp / cut-off differences (incl. delayed snapshots)
* 系统 Bug        our own defect -- the only category we fix
* 无法确认       not enough evidence yet

Run: python -m backend.data.audit [--date YYYYMMDD]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from backend.data.history import MarketHistory

KNOWN_DIFFERENCES = (
    ("成交额（回补日缺失）", "数据源差异", "腾讯日线 k-line 不含成交额，未估算"),
    ("涨跌停价", "统计口径差异", "本地按 pre_close × 板块规则推导，非交易所字段"),
    ("ST 名称与 5% 限制", "数据源差异", "stock_basic 名称快照会过期，已用近期波动校正"),
    ("除权/送转日涨跌幅", "复权差异", "不复权价，除权日基准不同；极端行已置空"),
    ("停牌与无成交", "统计口径差异", "停牌日无 bar（部分数据源给平盘假 bar）"),
    ("概念题材板块", "统计口径差异", "本地只有行业分类，题材横跨行业会漏"),
    ("涨停家数", "统计口径差异", "本地排除次新/除权并推导限价，与软件略有差异"),
    ("连板高度", "统计口径差异", "停牌中断连板；软件多按自然交易日"),
    ("盘中时间戳", "时间差异", "只有延迟快照，无逐笔/竞价时间"),
    ("市值/换手/量比", "数据源差异", "daily_basic 未接入"),
)


@dataclass
class AuditReport:
    date: Optional[str] = None
    codes: int = 0
    rows: int = 0
    checks: dict[str, Any] = field(default_factory=dict)
    bugs: tuple[str, ...] = ()
    known: tuple[tuple[str, str, str], ...] = KNOWN_DIFFERENCES

    @property
    def ok(self) -> bool:
        return not self.bugs

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "codes": self.codes,
            "rows": self.rows,
            "checks": self.checks,
            "bugs": list(self.bugs),
            "known_differences": [
                {"item": a, "class": b, "note": c} for a, b, c in self.known
            ],
            "ok": self.ok,
        }


def audit(date: Optional[str] = None, *, db_path: Optional[Any] = None) -> AuditReport:
    history = MarketHistory(db_path)
    coverage = history.coverage()
    frame = history.load(start_date=date, end_date=date) if date else history.load()
    if frame.empty:
        return AuditReport(date=date, codes=0, rows=0, checks={"empty": True})

    close = pd.to_numeric(frame["close"], errors="coerce")
    high = pd.to_numeric(frame["high"], errors="coerce")
    low = pd.to_numeric(frame["low"], errors="coerce")
    open_ = pd.to_numeric(frame["open"], errors="coerce")
    pre = pd.to_numeric(frame["pre_close"], errors="coerce")
    pct = pd.to_numeric(frame["pct_chg"], errors="coerce")
    vol = pd.to_numeric(frame["vol"], errors="coerce")

    checks: dict[str, Any] = {
        "ohlc_high_below_low": int((high < low).sum()),
        "ohlc_high_below_close_or_open": int(
            (high < frame[["open", "close"]].apply(pd.to_numeric).max(axis=1)).sum()
        ),
        "ohlc_low_above_close_or_open": int(
            (low > frame[["open", "close"]].apply(pd.to_numeric).min(axis=1)).sum()
        ),
        "non_positive_close": int((close <= 0).sum()),
        "negative_volume": int((vol < 0).sum()),
        "missing_pre_close": int(pre.isna().sum()),
        "duplicate_keys": int(
            frame.duplicated(subset=["ts_code", "trade_date"]).sum()
        ),
    }
    # pct_chg must match (close/pre_close - 1); a corporate-action row is the
    # only legitimate mismatch, so it is counted separately, not as a bug.
    derived = (close / pre - 1.0) * 100.0
    diff = (derived - pct).abs()
    checks["pct_chg_mismatch_gt_0.5pp"] = int((diff > 0.5).sum())
    checks["pct_chg_blank_on_corporate_action"] = int(pct.isna().sum())
    checks["amount_missing"] = int(pd.to_numeric(frame["amount"], errors="coerce").isna().sum())
    checks["amount_missing_share"] = round(
        float(pd.to_numeric(frame["amount"], errors="coerce").isna().mean()), 4
    )

    bugs: list[str] = []
    for key in (
        "ohlc_high_below_low",
        "ohlc_high_below_close_or_open",
        "ohlc_low_above_close_or_open",
        "non_positive_close",
        "negative_volume",
        "duplicate_keys",
    ):
        if checks[key]:
            bugs.append(f"{key}={checks[key]}")

    return AuditReport(
        date=date,
        codes=int(frame["ts_code"].nunique()),
        rows=int(len(frame)),
        checks={**checks, "coverage": coverage},
        bugs=tuple(bugs),
    )


def main() -> None:
    import argparse
    import io
    import json
    import sys

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Local data consistency audit")
    parser.add_argument("--date", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = audit(args.date)
    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        return
    print(f"数据一致性审计｜{report.date or '全窗口'}｜代码 {report.codes}｜行数 {report.rows}")
    for key, value in report.checks.items():
        if key == "coverage":
            continue
        print(f"  - {key}: {value}")
    print("系统 Bug:", list(report.bugs) or "无")
    print("已知差异分类：")
    for item, klass, note in report.known:
        print(f"  - [{klass}] {item}: {note}")


if __name__ == "__main__":
    main()
