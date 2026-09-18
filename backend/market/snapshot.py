"""Full-market delayed snapshot from free quote sources.

Uses AkShare's Sina-backed full-market spot endpoint, which returns all listed
A-shares in one call. The data is delayed (~seconds), never real-time; the
caller must surface that honestly.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd

from backend.market.engine import MarketFactor, MarketState, classify_market, clamp

_TENCENT_URL = "https://qt.gtimg.cn/q="
_TENCENT_BATCH = 300


def _exchange_prefix(ts_code: str) -> str:
    symbol, _, exchange = ts_code.partition(".")
    return f"{exchange.lower()}{symbol}"


def _all_ts_codes() -> list[str]:
    """Universe for the fallback source, taken from the local stock list."""
    from backend.data.config import get_settings

    path = get_settings().db_path.parent / "stock_basic.csv"
    if path.exists():
        try:
            basic = pd.read_csv(path, dtype=str)
            codes = sorted(basic["ts_code"].dropna().astype(str).unique().tolist())
            if codes:
                return codes
        except Exception:  # noqa: BLE001
            pass
    return []


def fetch_full_market_snapshot_tencent(
    ts_codes: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fallback full-market snapshot from Tencent (batched).

    Tencent also returns the price-limit prices and a real quote timestamp,
    which the Sina endpoint does not.
    """
    import requests

    codes = ts_codes or _all_ts_codes()
    if not codes:
        return pd.DataFrame(), {"source": "tencent", "timeliness": "delayed", "rows": 0}

    records: list[dict[str, Any]] = []
    quote_time = ""
    for i in range(0, len(codes), _TENCENT_BATCH):
        batch = codes[i : i + _TENCENT_BATCH]
        query = ",".join(_exchange_prefix(c) for c in batch)
        resp = requests.get(_TENCENT_URL + query, timeout=15)
        resp.encoding = "gbk"
        for line in resp.text.splitlines():
            if "=" not in line:
                continue
            payload = line.split("=", 1)[1].strip().strip('";')
            f = payload.split("~")
            if len(f) < 49 or not f[2]:
                continue
            try:
                price = float(f[3])
                pre_close = float(f[4])
            except (TypeError, ValueError):
                continue
            if price <= 0:
                continue
            symbol = f[2]
            suffix = "SH" if symbol.startswith(("60", "68", "90")) else (
                "BJ" if symbol.startswith(("43", "83", "87", "88", "92")) else "SZ"
            )
            records.append(
                {
                    "ts_code": f"{symbol}.{suffix}",
                    "name": f[1],
                    "price": price,
                    "change": _to_float(f[31]),
                    "pct_chg": _to_float(f[32]),
                    "pre_close": pre_close,
                    "open": _to_float(f[5]),
                    "high": _to_float(f[33]),
                    "low": _to_float(f[34]),
                    "volume": (_to_float(f[6]) or 0.0) * 100.0,   # 手 -> 股
                    "amount": (_to_float(f[37]) or 0.0) * 10000.0,  # 万元 -> 元
                }
            )
            if not quote_time and f[30]:
                raw = str(f[30])
                if len(raw) >= 14:
                    quote_time = f"{raw[8:10]}:{raw[10:12]}:{raw[12:14]}"

    frame = pd.DataFrame(records)
    meta = {
        "source": "tencent",
        "timeliness": "delayed",
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "quote_time": quote_time,
        "rows": len(frame),
    }
    return frame, meta


def _to_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sina_to_ts_code(code: str) -> str:
    code = str(code).strip().lower()
    if len(code) < 3:
        return code
    prefix, symbol = code[:2], code[2:]
    return f"{symbol}.{prefix.upper()}"


def fetch_full_market_snapshot_sina() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Sina-backed full-market snapshot (AkShare)."""
    import akshare as ak

    raw = ak.stock_zh_a_spot()
    if raw is None or raw.empty:
        meta = {
            "source": "sina",
            "timeliness": "delayed",
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "rows": 0,
        }
        return pd.DataFrame(), meta

    col_map = {
        "代码": "code",
        "名称": "name",
        "最新价": "price",
        "涨跌额": "change",
        "涨跌幅": "pct_chg",
        "昨收": "pre_close",
        "今开": "open",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "时间": "quote_time",
    }
    renamed = raw.rename(columns=col_map)
    needed = ["code", "name", "price", "change", "pct_chg", "pre_close", "open", "high", "low", "volume", "amount"]
    for col in needed:
        if col not in renamed.columns:
            renamed[col] = pd.NA

    out = pd.DataFrame(
        {
            "ts_code": renamed["code"].map(_sina_to_ts_code),
            "name": renamed["name"].astype(str),
            "price": pd.to_numeric(renamed["price"], errors="coerce"),
            "change": pd.to_numeric(renamed["change"], errors="coerce"),
            "pct_chg": pd.to_numeric(renamed["pct_chg"], errors="coerce"),
            "pre_close": pd.to_numeric(renamed["pre_close"], errors="coerce"),
            "open": pd.to_numeric(renamed["open"], errors="coerce"),
            "high": pd.to_numeric(renamed["high"], errors="coerce"),
            "low": pd.to_numeric(renamed["low"], errors="coerce"),
            "volume": pd.to_numeric(renamed["volume"], errors="coerce"),
            "amount": pd.to_numeric(renamed["amount"], errors="coerce"),
        }
    )
    out = out.dropna(subset=["ts_code"]).drop_duplicates(subset=["ts_code"])
    quote_time = ""
    if "quote_time" in renamed.columns:
        non_null = renamed["quote_time"].dropna()
        if not non_null.empty:
            quote_time = str(non_null.iloc[0])

    meta = {
        "source": "sina",
        "timeliness": "delayed",
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "quote_time": quote_time,
        "rows": len(out),
    }
    return out, meta


def rank_snapshot(
    df: pd.DataFrame,
    *,
    sort_by: str = "pct_chg",
    ascending: bool = False,
    top_n: int = 100,
    min_amount: float | None = None,
    min_price: float | None = None,
) -> pd.DataFrame:
    """Filter and rank a snapshot DataFrame."""
    if df.empty:
        return df
    filtered = df.copy()
    if min_amount is not None:
        filtered = filtered[filtered["amount"] >= min_amount]
    if min_price is not None:
        filtered = filtered[filtered["price"] >= min_price]
    if sort_by not in filtered.columns:
        sort_by = "pct_chg"
    filtered = filtered.sort_values(sort_by, ascending=ascending)
    return filtered.head(top_n)


def fetch_full_market_snapshot() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Full-market snapshot with automatic source fallback (Sina -> Tencent).

    A single free endpoint is not dependable: Sina started returning HTML
    instead of JSON during testing. Falling back keeps the "当日数据" path
    alive, and the meta records which source actually answered.
    """
    errors: list[str] = []
    sources = (
        ("sina", fetch_full_market_snapshot_sina),
        ("tencent", fetch_full_market_snapshot_tencent),
    )
    for name, fetcher in sources:
        try:
            frame, meta = fetcher()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}: {exc}")
            continue
        if frame is not None and not frame.empty:
            if errors:
                meta["fallback_from"] = "; ".join(errors)
            return frame, meta
        errors.append(f"{name}: empty")
    raise RuntimeError("all quote sources failed: " + "; ".join(errors))


def _limit_pct(ts_code: str) -> float:
    """Approximate limit-up percentage threshold for a board."""
    symbol = ts_code.split(".")[0]
    if ts_code.endswith(".BJ"):
        return 29.5
    if symbol.startswith(("30", "688")):
        return 19.5
    return 9.5


def _limit_masks(df: pd.DataFrame):
    """Return (limit_up, limit_down, broken, reseal) boolean masks.

    Limit prices are derived from the previous close and the board rule, so a
    stock that touched the limit intraday but closed below it counts as broken
    (炸板) and one that closed back at the limit counts as resealed (回封).
    """
    symbol = df["ts_code"].str.split(".").str[0]
    rate = pd.Series(0.10, index=df.index)
    rate.loc[symbol.str.startswith(("30", "688"))] = 0.20
    rate.loc[df["ts_code"].str.endswith(".BJ")] = 0.30
    pre_close = pd.to_numeric(df["pre_close"], errors="coerce")
    up_limit = (pre_close * (1 + rate)).round(2)
    down_limit = (pre_close * (1 - rate)).round(2)
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["price"], errors="coerce")
    limit_up = close >= up_limit
    limit_down = close <= down_limit
    broken = (high >= up_limit) & (~limit_up)
    reseal = (high >= up_limit) & (low < up_limit) & limit_up
    return limit_up, limit_down, broken, reseal


def compute_full_market_state(df: pd.DataFrame, date: str) -> MarketState:
    """Build a full-market ``MarketState`` from a snapshot DataFrame."""
    if df is None or df.empty:
        return MarketState(date=date, state="数据不足", confidence=0.0, risk_level="high")

    total = len(df)
    pct = pd.to_numeric(df["pct_chg"], errors="coerce")
    advance = int((pct > 0).sum())
    decline = int((pct < 0).sum())
    flat = int((pct == 0).sum())

    limit_up_mask, limit_down_mask, broken_mask, _reseal_mask = _limit_masks(df)
    limit_up = int(limit_up_mask.sum())
    limit_down = int(limit_down_mask.sum())
    broken = int(broken_mask.sum())
    attempted = limit_up + broken
    success_rate = (limit_up / attempted) if attempted else None

    amount = pd.to_numeric(df["amount"], errors="coerce")
    total_amount = float(amount.sum(skipna=True))
    breadth = advance / total if total else None

    factors: list[MarketFactor] = []
    score = 50.0
    if breadth is not None:
        score += (breadth - 0.5) * 100.0
        factors.append(
            MarketFactor("breadth", round(breadth, 4), 1.0, f"上涨家数占比 {breadth:.1%}")
        )
    if total:
        ratio = (limit_up - limit_down) / total
        score += ratio * 50.0
        factors.append(
            MarketFactor(
                "limit_differential",
                round(ratio, 4),
                1.0,
                f"涨停 {limit_up} / 跌停 {limit_down}",
            )
        )
    if broken:
        score -= min(15.0, broken * 0.3)
        factors.append(
            MarketFactor("broken_boards", float(broken), 1.0, f"炸板 {broken} 家")
        )
    if success_rate is not None:
        factors.append(
            MarketFactor(
                "limit_up_success",
                round(success_rate, 4),
                1.0,
                f"涨停封板率 {success_rate:.1%}",
            )
        )

    score = clamp(score)
    state, risk_level, exposure = classify_market(score)
    confidence = 1.0 if total >= 3000 else 0.7

    return MarketState(
        date=date,
        advance_count=advance,
        decline_count=decline,
        flat_count=flat,
        total_count=total,
        limit_up_count=limit_up,
        limit_down_count=limit_down,
        broken_count=broken,
        limit_up_success_rate=success_rate,
        total_amount=total_amount,
        breadth=breadth,
        score=round(score, 2),
        state=state,
        confidence=confidence,
        risk_level=risk_level,
        recommended_exposure=exposure,
        factors=tuple(factors),
    )


def compute_full_market_sectors(df: pd.DataFrame, industry_map: dict[str, str]) -> list[dict[str, Any]]:
    """Compute full-market sector strength from a snapshot DataFrame."""
    if df is None or df.empty:
        return []
    work = df.copy()
    work["industry"] = work["ts_code"].map(lambda code: industry_map.get(code, ""))
    work = work[work["industry"] != ""]
    if work.empty:
        return []

    market_avg = float(pd.to_numeric(work["pct_chg"], errors="coerce").mean())
    sectors: list[dict[str, Any]] = []
    for name, group in work.groupby("industry"):
        if len(group) < 2:
            continue
        pct = pd.to_numeric(group["pct_chg"], errors="coerce")
        amount = pd.to_numeric(group["amount"], errors="coerce")
        avg_pct = float(pct.mean())
        breadth = float((pct > 0).mean())
        limit_up = int((pct >= 9.5).sum())
        leader_idx = pct.idxmax()
        leader = str(group.loc[leader_idx, "ts_code"])
        leader_name = str(group.loc[leader_idx, "name"])
        leader_pct = float(pct.loc[leader_idx])
        score = clamp(50.0 + (avg_pct - market_avg) * 2.0 + (breadth - 0.5) * 40.0 + min(12.0, limit_up * 2.0))
        sectors.append(
            {
                "name": name,
                "score": round(score, 2),
                "relative_strength": round(avg_pct - market_avg, 4),
                "breadth": round(breadth, 4),
                "limit_up_count": limit_up,
                "leader": leader,
                "leader_name": leader_name,
                "leader_pct_chg": round(leader_pct, 4),
                "persistence": 0.5,
                "diffusion": round(breadth, 4),
                "weakening": round(float((pct < 0).mean()), 4),
                "risk": "low" if score >= 75 else ("medium" if score >= 55 else "high"),
                "avg_pct_chg": round(avg_pct, 4),
                "total_amount": float(amount.sum()),
                "stock_codes": [str(c) for c in group["ts_code"].tolist()],
            }
        )
    sectors.sort(key=lambda item: item["score"], reverse=True)
    for rank, item in enumerate(sectors, start=1):
        item["rank"] = rank
    return sectors


def compute_full_market_board(df: pd.DataFrame, date: str) -> dict[str, Any]:
    """Compute full-market limit-up / broken / reseal counts from a snapshot."""
    result: dict[str, Any] = {
        "date": date,
        "limit_up_count": 0,
        "broken_count": 0,
        "reseal_count": 0,
        "max_height": 0,
    }
    if df is None or df.empty:
        return result

    limit_up, _limit_down, broken, reseal = _limit_masks(df)
    result["limit_up_count"] = int(limit_up.sum())
    result["broken_count"] = int(broken.sum())
    result["reseal_count"] = int(reseal.sum())
    return result


def compute_full_market_board_environment(
    df: pd.DataFrame,
    date: str,
    *,
    history_extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Full-market board environment (grade + limit-up / broken / reseal).

    A live snapshot only describes one instant, so it cannot derive
    consecutive-limit-up ladders or yesterday's premium by itself.
    ``history_extras`` (from the local multi-day store) fills exactly those
    fields; anything still unknown stays explicitly unavailable.
    """
    from backend.board.engine import grade_board
    from backend.emotion.engine import classify_emotion

    counts = compute_full_market_board(df, date)
    market = compute_full_market_state(df, date)
    phase, exposure, risk, _score = classify_emotion(market)
    extras = history_extras or {}
    ladder = extras.get("ladder")
    return {
        "date": date,
        "grade": grade_board(market.score),
        "score": market.score,
        "emotion_cycle": phase,
        "limit_up_count": counts["limit_up_count"],
        "broken_count": counts["broken_count"],
        "reseal_count": counts["reseal_count"],
        "yesterday_limit_up_premium": extras.get("yesterday_limit_up_premium"),
        "yesterday_premium_available": extras.get("yesterday_limit_up_premium") is not None,
        "leader": extras.get("leader"),
        "leader_name": extras.get("leader_name"),
        "leader_height": extras.get("leader_height", 0),
        "leader_available": bool(extras.get("leader")),
        "ladder": ladder,
        "ladder_available": ladder is not None,
        "ladder_source": extras.get("source"),
        "ladder_note": extras.get("note")
        or "连板梯队需要全市场多日历史，本地历史库尚未覆盖该交易日",
        "risk_level": risk,
        "recommended_exposure": exposure,
        "factors": [f.description for f in market.factors],
    }
