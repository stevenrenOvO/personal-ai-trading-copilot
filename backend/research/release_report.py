"""Freeze a local history snapshot and rerun the V1.1 research benchmark.

No network requests, real account access, or parameter search. The snapshot
and its manifest are kept under data/tmp; results are JSON and Markdown.
"""
from __future__ import annotations

import argparse
from contextlib import closing
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import sqlite3

import pandas as pd

from backend.backtest.ashare_engine import AShareBacktestEngine
from backend.backtest.metrics import calculate_metrics
from backend.data.providers.history import HistoryProvider
from backend.strategies.ma_cross import MaCrossStrategy
from backend.strategies.strong_sector_breakout import StrongSectorBreakoutStrategy

PROJECT = Path(__file__).resolve().parents[2]
ENGINE_CONFIG = {
    "initial_cash": 1_000_000.0, "commission_rate": 0.0003,
    "stamp_duty_rate": 0.0005, "slippage": 0.001, "lot_size": 100,
    "t_plus_one": True, "max_position_weight": 0.2,
    "stop_loss_pct": 0.08, "take_profit_pct": 0.20,
}
STRATEGIES = {
    "strong_sector_breakout": {"lookback": 20, "min_volume_ratio": 1.2,
        "max_sector_rank": None, "max_signals": 5},
    "ma_cross": {"fast_window": 5, "slow_window": 20},
}
WALK_FORWARD_CONFIG = {"train_days": 40, "test_days": 15, "step_days": 15}
LIMITATIONS = [
    "股票池来自当前 stock_basic 快照，缺少逐日上市、退市、ST 状态，存在幸存者偏差。",
    "涨跌停价格由历史 pre_close 和当前名称/截至当日波动启发式推导，并非供应商真实历史限制价。",
    "缺少复权因子、公司行动和股息现金流；不能把除权造成的价格变化全部解释成策略收益。",
    "缺少历史停牌表、盘口与流动性成交模型；无 bar 或无成交量时不成交，其余按日线开盘模拟。",
    "强势突破未接入逐日历史行业排名，本次仅评估价格/成交量突破规则，不能当作完整板块策略表现。",
    "未做参数搜索、训练或正式样本外盈利验证；这些结果是执行引擎回归基线，不是策略盈利证明。",
    "费用与滑点是冻结的研究假设，不代表任何真实券商收费；末日持仓按收盘估值，不强制平仓。",
    "数据是原本地缓存的冻结副本；没有重新联网核验供应商原始日线。",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _backup(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True)) as src:
        with closing(sqlite3.connect(target)) as dst:
            src.backup(dst)


def freeze_snapshot(data_root: Path, snapshot: Path) -> dict:
    """Create once; never silently replace a frozen experiment."""
    snapshot.mkdir(parents=True, exist_ok=False)
    _backup(data_root / "history" / "market_history.db", snapshot / "history" / "market_history.db")
    shutil.copyfile(data_root / "stock_basic.csv", snapshot / "stock_basic.csv")
    if (data_root / "cache" / "session.db").exists():
        _backup(data_root / "cache" / "session.db", snapshot / "session.db")
    if (data_root / "trade_cal.csv").exists():
        shutil.copyfile(data_root / "trade_cal.csv", snapshot / "trade_cal.csv")
    files = {str(p.relative_to(snapshot)).replace("\\", "/"): sha256(p)
             for p in sorted(snapshot.rglob("*")) if p.is_file()}
    manifest = {"format": 1, "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
                "source": str(data_root.resolve()), "files": files}
    _dump(snapshot / "manifest.json", manifest)
    return manifest


def verify_snapshot(snapshot: Path) -> dict:
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        path = (snapshot / name).resolve()
        if not path.is_relative_to(snapshot.resolve()) or not path.is_file() or sha256(path) != expected:
            raise ValueError(f"冻结样本校验失败: {name}")
    return manifest


def _calendar(snapshot: Path) -> pd.DataFrame:
    session = snapshot / "session.db"
    if session.exists():
        with closing(sqlite3.connect(session.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
            return pd.read_sql_query("SELECT cal_date,is_open FROM calendar ORDER BY cal_date", conn)
    path = snapshot / "trade_cal.csv"
    return pd.read_csv(path, dtype={"cal_date": str}) if path.exists() else pd.DataFrame(columns=["cal_date", "is_open"])


def data_quality(snapshot: Path) -> tuple[dict, tuple[str, str]]:
    """Choose the longest calendar-verified, sufficiently covered interval."""
    provider = HistoryProvider(base_dir=snapshot)
    bars = provider.history.load()
    basic = provider.stock_basic()
    listed = set(basic.ts_code.astype(str))
    stocks = bars[bars.ts_code.isin(listed)].copy()
    counts = stocks.groupby("trade_date").size()
    calendar = _calendar(snapshot)
    if bars.empty or calendar.empty:
        raise ValueError("研究需要本地日线与独立交易日历，不能从缺失行情推断交易日")
    first, last = str(bars.trade_date.min()), str(bars.trade_date.max())
    open_dates = sorted(str(d) for d in calendar.loc[calendar.is_open.astype(int) == 1, "cal_date"] if first <= str(d) < last)
    threshold = math.ceil(len(listed) * 0.9)
    runs, current = [], []
    for day in open_dates:
        if int(counts.get(day, 0)) >= threshold:
            current.append(day)
        else:
            if current:
                runs.append(current)
            current = []
    if current:
        runs.append(current)
    chosen = max(runs, key=lambda dates: (len(dates), dates[-1]), default=[])
    if len(chosen) < 22:
        raise ValueError("没有足够的连续完整历史（至少22个已验证交易日）进行20日指标研究")
    prices = stocks[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
    invalid = ~prices.apply(lambda column: column.map(lambda n: math.isfinite(n) and n > 0)).all(axis=1)
    inconsistent = (prices.high < prices[["open", "close"]].max(axis=1)) | (prices.low > prices[["open", "close"]].min(axis=1)) | (prices.high < prices.low)
    report = {
        "rows": len(bars), "stock_rows": len(stocks), "codes": int(bars.ts_code.nunique()),
        "stock_basic_codes": len(listed), "non_stock_codes": sorted(set(bars.ts_code) - listed),
        "dates": int(bars.trade_date.nunique()), "start": first, "end": last,
        "invalid_price_rows": int(invalid.sum()), "inconsistent_ohlc_rows": int(inconsistent.sum()),
        "nonpositive_or_missing_volume_rows": int((stocks.vol.isna() | (stocks.vol <= 0)).sum()),
        "missing_amount_rows": int(stocks.amount.isna().sum()),
        "calendar_start": str(calendar.cal_date.min()), "calendar_end": str(calendar.cal_date.max()),
        "missing_or_low_coverage_open_dates": [d for d in open_dates if int(counts.get(d, 0)) < threshold],
        "coverage_threshold": threshold, "selected_dates": chosen,
        "excluded_latest_date": last,
        "date_policy": "最长独立日历验证且股票覆盖>=90%的连续区间；排除数据库最后一天（未证明已收盘）",
        "per_date_stock_rows": {str(d): int(n) for d, n in counts.items()},
    }
    if invalid.any() or inconsistent.any():
        raise ValueError("冻结日线存在无效价格或OHLC矛盾，请先修复数据再运行研究")
    return report, (chosen[0], chosen[-1])


def _code_hashes() -> dict:
    folders = ["backend/backtest", "backend/strategies", "backend/paper", "backend/research", "backend/journal"]
    paths = {p for folder in folders for p in (PROJECT / folder).glob("*.py")}
    paths.update(PROJECT / p for p in ["backend/data/history.py", "backend/data/schemas.py",
        "backend/data/sqlite.py", "backend/data/providers/history.py", "backend/data/providers/base.py",
        "backend/data/providers/_a_share_utils.py", "backend/data/config.py", "backend/data/atomic.py",
        "backend/data/calendar.py", "VERSION", "requirements.txt"] if (PROJECT / p).exists())
    return {str(p.relative_to(PROJECT)).replace("\\", "/"): sha256(p) for p in sorted(paths)}


def _freeze_code(snapshot: Path, hashes: dict) -> str:
    """Preserve the exact uncommitted source used for this experiment."""
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    target = snapshot / "source" / digest
    for name, expected in hashes.items():
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            shutil.copyfile(PROJECT / name, path)
        if sha256(path) != expected:
            raise ValueError(f"冻结代码校验失败: {name}")
    _dump(target / "code-manifest.json", hashes)
    return str(target.resolve())


def run_report(snapshot: Path) -> dict:
    manifest = verify_snapshot(snapshot)
    code_hashes = _code_hashes()
    previous_path = snapshot / "report.json"
    previous = json.loads(previous_path.read_text(encoding="utf-8")) if previous_path.exists() else {}
    quality, (start, end) = data_quality(snapshot)
    provider = HistoryProvider(base_dir=snapshot)
    results = {}
    for name, params in STRATEGIES.items():
        strategy = StrongSectorBreakoutStrategy(**params) if name == "strong_sector_breakout" else MaCrossStrategy(**params)
        result = AShareBacktestEngine(provider, strategy, **ENGINE_CONFIG).run(start_date=start, end_date=end)
        equity = [{"date": str(row.date), "total": float(row.total)} for row in result.equity_curve.itertuples()]
        trades = [trade.to_dict() for trade in result.trade_log]
        orders = [{**asdict(order), "side": order.side.value} for order in result.orders]
        trace = {"equity": equity, "orders": orders, "trades": trades}
        trace_path = snapshot / f"{name}-trace.json"
        _dump(trace_path, trace)
        realized = sum(trade.pnl for trade in result.trade_log)
        remaining_cost = sum(p.cost_basis for p in result.positions.values())
        unrealized = result.final_total_value - result.final_cash - remaining_cost
        residual = result.final_total_value - ENGINE_CONFIG["initial_cash"] - realized - unrealized
        if abs(residual) > 0.01:
            raise ValueError(f"{name} 账户盈亏无法对账: {residual}")
        results[name] = {
            "metrics": calculate_metrics(result.equity_curve, trade_log=result.trade_log),
            "final_cash": result.final_cash, "final_equity": result.final_total_value,
            "order_count": len(result.orders), "open_positions": sum(p.quantity > 0 for p in result.positions.values()),
            "realized_pnl": realized, "unrealized_pnl": unrealized, "reconciliation_residual": residual,
            "trace_sha256": sha256(trace_path),
            "lookahead_diagnostics": result.lookahead_issues,
        }
        strategy = StrongSectorBreakoutStrategy(**params) if name == "strong_sector_breakout" else MaCrossStrategy(**params)
        folds = AShareBacktestEngine(provider, strategy, **ENGINE_CONFIG).walk_forward(
            start_date=start, end_date=end, **WALK_FORWARD_CONFIG)
        results[name]["walk_forward"] = folds.to_dict()
    verify_snapshot(snapshot)
    if _code_hashes() != code_hashes:
        raise ValueError("研究运行期间代码发生变化，请等待代码稳定后重新复算")
    code_snapshot = _freeze_code(snapshot, code_hashes)
    report = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "snapshot": str(snapshot.resolve()),
        "manifest": manifest, "software_version": (PROJECT / "VERSION").read_text().strip(),
        "environment": {"python": platform.python_version(), "pandas": pd.__version__},
        "code_sha256": code_hashes, "code_snapshot": code_snapshot,
        "engine_config": ENGINE_CONFIG, "strategy_config": STRATEGIES,
        "walk_forward_config": WALK_FORWARD_CONFIG,
        "data_quality": quality, "research_window": {"start": start, "end": end},
        "results": results, "limitations": LIMITATIONS}
    report["experiment_sha256"] = hashlib.sha256(json.dumps({
        "inputs": manifest["files"], "code": code_hashes, "engine": ENGINE_CONFIG,
        "strategies": STRATEGIES, "walk_forward": WALK_FORWARD_CONFIG,
        "window": report["research_window"], "environment": report["environment"],
    }, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    repeated = previous.get("experiment_sha256") == report["experiment_sha256"]
    if repeated and previous.get("results") != report["results"]:
        raise ValueError("相同实验输入产生了不同输出，请检查可重复性")
    report["reproducibility"] = {"same_experiment_repeat_verified": repeated}
    _dump(snapshot / "report.json", report)
    return report


def render_markdown(report: dict) -> str:
    q = report["data_quality"]
    lines = ["# V1.1 冻结样本研究与数据质量报告", "",
        "本报告重算了当前执行引擎的研究基线。历史证券状态、公司行动与完整成交数据仍缺失，结果不能用于证明策略盈利。", "",
        f"生成时间（UTC）：{report['generated_at_utc']}；软件版本：{report['software_version']}。", "",
        "## 冻结数据与样本选择", "",
        f"- 本地库 {q['rows']:,} 行、{q['codes']:,} 个代码、{q['dates']} 个日期，覆盖 {q['start']}～{q['end']}。",
        f"- 股票名单 {q['stock_basic_codes']:,} 只；排除非股票代码 {', '.join(q['non_stock_codes']) or '无'}。",
        f"- 研究区间 {report['research_window']['start']}～{report['research_window']['end']}，共 {len(q['selected_dates'])} 个交易日。",
        f"- 选择规则：{q['date_policy']}。没有按收益筛选代码或区间。",
        f"- 独立日历范围 {q['calendar_start']}～{q['calendar_end']}；覆盖不足/缺失日期：{', '.join(q['missing_or_low_coverage_open_dates']) or '无'}。",
        f"- 无效价格 {q['invalid_price_rows']} 行；OHLC 矛盾 {q['inconsistent_ohlc_rows']} 行；无成交量 {q['nonpositive_or_missing_volume_rows']} 行；缺少成交额 {q['missing_amount_rows']:,} 行。无成交量 bar 不参与成交。", "",
        "## 固定配置与结果", "",
        "初始现金 100 万；佣金 0.03%；卖出税费假设 0.05%；单边滑点 0.1%；单票最高20%；收盘确认8%止损/20%止盈后下一交易日开盘尝试退出；T+1、100股整手和涨跌停约束。费用只代表研究配置。", "",
        "两个策略均使用固定默认参数；强势突破 lookback=20、量比>=1.2、每日最多5个信号；均线交叉5/20。未提供历史行业排名，强势突破的行业排名过滤关闭。", "",
        "| 策略 | 账户收益 | 最大回撤 | 平仓样本 | 交易胜率 | 期末持仓 | 期末权益 |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for name, result in report["results"].items():
        m = result["metrics"]
        win = f"{m['win_rate']:.2%}" if m["win_rate"] is not None else "无样本"
        lines.append(f"| {name} | {m['total_return']:.2%} | {m['max_drawdown']:.2%} | {m['trade_count']} | {win} | {result['open_positions']} | {result['final_equity']:,.2f} |")
    lines += ["", "每个策略均已核对：期末权益变化 = 已实现净盈亏 + 剩余持仓浮盈亏，误差不超过0.01。完整净值、订单和平仓记录位于冻结目录的 `*-trace.json`。未平仓收益与交易胜率使用不同口径。首个收盘估值也与初始现金比较，包含首日成交成本。", "",
        "## 滚动折叠检查", "",
        "固定40日训练区间、15日测试区间、15日步长。测试指标用之前的训练日线预热，每折从100万元现金起步，训练持仓不带入测试。未重新拟合或选择参数；重叠的训练区间与独立测试账户不拼接成一条净值。下表收益是测试折的算术平均，胜率按平仓笔数汇总，不能解释为整个区间投资收益。", "",
        "| 策略 | 测试折数 | 测试折平均收益 | 测试平仓样本 | 测试交易胜率 |",
        "|---|---:|---:|---:|---:|"]
    for name, result in report["results"].items():
        m = result["walk_forward"]["out_of_sample"]
        win = f"{m['win_rate']:.2%}" if m["win_rate"] is not None else "无样本"
        lines.append(f"| {name} | {m['fold_count']} | {m['total_return']:.2%} | {m['trade_count']} | {win} |")
    lines += ["", "## 数据限制", ""] + [f"- {item}" for item in report["limitations"]]
    lines += ["", "## 复现", "", f"冻结目录：`{report['snapshot']}`。该目录中的原始数据不会被后续在线补数覆盖；`manifest.json` 保存输入文件 SHA-256，报告 JSON 保存代码哈希、配置、环境与执行轨迹哈希。", "",
        "```powershell", f'.\\.venv\\Scripts\\python.exe -m backend.research.release_report --snapshot "{report["snapshot"]}" --output docs/V1.1-research-report.md', "```", "",
        f"实验指纹：`{report['experiment_sha256']}`。代码副本：`{report['code_snapshot']}`；保留了未提交的执行源码，避免后续修改丢失复现依据。报告中的生成时间不参与实验指纹。", "",
        f"相同输入与代码的重复计算比对：{'已通过（完整结果与轨迹哈希一致）' if report['reproducibility']['same_experiment_repeat_verified'] else '本次为该实验指纹首次计算；再次运行将自动比对完整结果与轨迹哈希'}。", "",
        "冻结文件哈希不符时脚本拒绝运行；代码在计算期间变化也会终止。代码更新后可重算并比较轨迹哈希与实验指纹。冻结包包含本地行情，未提交到 Git；Markdown 与报告 JSON 可以随版本审查。", "",
        "| 冻结文件 | SHA-256 |", "|---|---|"]
    lines += [f"| {name} | `{digest}` |" for name, digest in report["manifest"]["files"].items()]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=PROJECT / "data")
    parser.add_argument("--snapshot", type=Path, help="Use an existing frozen snapshot; do not copy live data")
    parser.add_argument("--output", type=Path, default=PROJECT / "docs" / "V1.1-research-report.md")
    args = parser.parse_args()
    snapshot = args.snapshot
    if snapshot is None:
        snapshot = PROJECT / "data" / "tmp" / ("v1_1-research-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
        freeze_snapshot(args.data_root, snapshot)
    report = run_report(snapshot)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_markdown(report), encoding="utf-8")
    _dump(args.output.with_suffix(".json"), report)
    print(json.dumps({"report": str(args.output), "snapshot": str(snapshot), "window": report["research_window"], "results": report["results"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
