"""Local readiness checks; never wait for an upstream market-data request."""
from __future__ import annotations

from pathlib import Path

_LOADED_VERSION = (Path(__file__).resolve().parents[2] / "VERSION").read_text(encoding="utf-8").strip()


def version() -> str:
    return _LOADED_VERSION


def readiness(runtime=None, initialization_failed: bool = False) -> dict:
    if runtime is None:
        return {"status": "error" if initialization_failed else "starting", "version": version(),
                "checks": {"account": {"status": "error" if initialization_failed else "warning",
                           "message": "服务初始化失败，请查看运行日志" if initialization_failed else "服务正在初始化，请稍后刷新"}}}
    checks = {}
    try:
        account = runtime.paper.snapshot()
        checks["account"] = {"status": "warning" if account.journal_pending else "ok",
            "message": "账户已保存，复盘日志等待恢复" if account.journal_pending else "账户可读取，成交写入受保护"}
    except Exception:
        checks["account"] = {"status": "error", "message": "账户状态无法读取，交易暂停；请查看运行日志并恢复备份"}

    info = {}
    try:
        from backend.market.session import get_trading_session
        info = get_trading_session().describe_cached()
        valid = info["is_trading_day"] is not None
        checks["calendar"] = {"status": "ok" if valid else "warning",
            "message": f"{info['current_trading_day']} · {info['phase']}" if valid else "交易日历未确认，等待后台更新"}
    except Exception:
        checks["calendar"] = {"status": "warning", "message": "交易日历暂不可用"}

    dates = []
    try:
        history = getattr(runtime.provider, "history", None)
        if history is None:
            checks["history"] = {"status": "warning", "message": "演示数据" if runtime.provider.name == "fixture" else "尚无本地全市场日线库"}
        else:
            from backend.api.data_status import history_dates
            dates = history_dates(runtime)
            last = dates[-1] if dates else {}
            latest = last.get("date")
            rows = last.get("valid_rows", 0)
            ratio = last.get("coverage_ratio", 0)
            expected = info.get("expected_daily_date")
            complete = max((item["date"] for item in dates if item["complete"]
                            and (not expected or item["date"] <= expected)), default=None)
            good = bool(complete and expected and complete >= expected)
            checks["history"] = {"status": "ok" if good else "warning", "latest_date": latest,
                "rows": rows, "coverage_ratio": ratio, "expected_date": expected, "latest_complete_date": complete,
                "analysis_threshold": .9,
                "message": f"最近可分析日 {complete or '暂无'}；最新采集 {latest or '暂无'}，有效覆盖 {ratio:.1%}"
                    + ("；达到分析门槛，少量缺失仍在补齐" if last.get("complete") and ratio < 1 else "")}
    except Exception:
        checks["history"] = {"status": "warning", "message": "历史数据覆盖暂不可读取"}

    try:
        history = getattr(runtime.provider, "history", None)
        expected = info.get("expected_daily_date")
        if history is None or not expected:
            checks["indexes"] = {"status": "warning", "message": "演示数据" if runtime.provider.name == "fixture" else "指数日线尚未确认"}
        else:
            codes = ("000300.SH", "000905.SH", "399006.SZ")
            with history._connect() as conn:
                rows = conn.execute("SELECT ts_code,MAX(trade_date) FROM daily WHERE ts_code IN (?,?,?) "
                    "AND trade_date<=? AND open>0 AND high>0 AND low>0 AND close>0 "
                    "AND high>=open AND high>=close AND low<=open AND low<=close AND high>=low "
                    "AND high<1e9 GROUP BY ts_code", (*codes, expected)).fetchall()
            latest = dict(rows)
            matching = sum(latest.get(code) == expected for code in codes)
            checks["indexes"] = {"status": "ok" if matching == len(codes) else "warning",
                "date": expected, "count": matching, "expected_count": len(codes), "latest_dates": latest,
                "message": f"{expected} · {matching}/{len(codes)} 项指数日线已更新"
                    + ("；旧日期不计入当日指数" if matching < len(codes) else "")}
    except Exception:
        checks["indexes"] = {"status": "warning", "message": "指数日线覆盖暂不可读取"}

    try:
        from backend.market.quote_service import get_quote_service
        _, meta = get_quote_service().get()
        expected_quote = info.get("expected_daily_date") if info.get("phase") == "盘前" else info.get("current_trading_day")
        current = bool(meta.quote_date and meta.quote_date == expected_quote and not meta.is_stale)
        source = {"tencent": "腾讯", "sina": "新浪"}.get(meta.source, meta.source)
        checks["quotes"] = {"status": "ok" if current else "warning", "quote_date": meta.quote_date,
            "quote_time": meta.quote_time, "source": meta.source, "rows": meta.rows,
            "message": f"{source} · {meta.quote_date or '日期未确认'} · {meta.rows:,} 只" + (" · 已收盘" if meta.status == "final" else "" if current else "；行情待更新")}
    except Exception:
        checks["quotes"] = {"status": "warning", "message": "行情缓存暂不可用"}

    try:
        from backend.data.auto_ingest import status
        ingest = status(runtime)
        pending = ingest["pending"]
        # A few absent bars (e.g. suspended/new securities) need retries but do
        # not make a day below the analysis gate. Keep the two states distinct.
        by_date = {item["date"]: item for item in dates}
        minor = sorted(day for day in pending if by_date.get(day, {}).get("complete"))
        critical = sorted(set(pending) - set(minor))
        retry_dates = [item.get("next_retry_at") for item in pending.values() if isinstance(item, dict) and item.get("next_retry_at")]
        message = (f"{len(critical)} 个日期未达分析门槛，已安排重试" if critical
                   else f"{len(minor)} 个日期覆盖已达分析门槛，少量缺失继续重试" if minor
                   else "后台正在检查历史数据" if ingest["running"] else "暂无待重试任务")
        checks["ingest"] = {"status": "warning" if critical else "ok", "message": message,
            "running": ingest["running"], "pending_dates": sorted(pending),
            "critical_gap_dates": critical, "minor_missing_dates": minor,
            "next_retry_at": min(retry_dates) if retry_dates else None}
        progress = ingest.get("progress") or {}
        checks["ingest"]["progress"] = progress
        if ingest["running"] and progress.get("total"):
            checks["ingest"]["message"] += f"；检查 {progress.get('window_start', '')}～{progress.get('window_end', '')}：{progress.get('processed', 0):,}/{progress['total']:,}，已落地 {progress.get('downloaded', 0):,} 只"
    except Exception:
        checks["ingest"] = {"status": "warning", "message": "补数状态暂不可读取"}
    overall = "error" if checks["account"]["status"] == "error" else "degraded" if any(c["status"] != "ok" for c in checks.values()) else "ready"
    return {"status": overall, "version": version(), "data_mode": "demo" if runtime.provider.name == "fixture" else "production", "checks": checks}
