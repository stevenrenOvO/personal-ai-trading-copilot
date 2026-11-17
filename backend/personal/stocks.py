"""Find stocks by name (or code) in the local stock list.

The user remembers names, not codes. Every personal input accepts either; a
name must match exactly one stock or the caller is told which ones matched --
nothing is guessed.
"""

from __future__ import annotations

import re
from typing import Any

_CODE = re.compile(r"(\d{6})(?:\.(SH|SZ|BJ))?")


def _basic(runtime: Any):
    try:
        frame = runtime.provider.stock_basic()
    except Exception:  # noqa: BLE001
        return None
    if frame is None or frame.empty or "name" not in frame or "ts_code" not in frame:
        return None
    return frame[["ts_code", "name"]].dropna().astype(str)


def _suffix(symbol: str) -> str:
    if symbol.startswith(("60", "68", "90")):
        return "SH"
    if symbol.startswith(("43", "83", "87", "88", "92")):
        return "BJ"
    return "SZ"


def search_stocks(runtime: Any, query: Any, limit: int = 10) -> list[dict[str, str]]:
    """Exact name, then names starting with the query, then containing it, then codes."""
    text = str(query or "").strip()
    frame = _basic(runtime)
    if not text or frame is None:
        return []
    upper = text.upper()
    names, codes = frame["name"], frame["ts_code"]
    ranked = []
    for rank, mask in enumerate((names == text, names.str.startswith(text) & (names != text),
                                 names.str.contains(text, regex=False) & ~names.str.startswith(text),
                                 codes.str.startswith(upper))):
        for row in frame[mask].itertuples(index=False):
            ranked.append((rank, len(row.name), row.ts_code, row.name))
    seen, out = set(), []
    for _, _, code, name in sorted(ranked):
        if code not in seen:
            seen.add(code)
            out.append({"ts_code": code, "name": name})
        if len(out) >= limit:
            break
    return out


def resolve_stock(runtime: Any, value: Any) -> tuple[str, str]:
    """Return (ts_code, name) for a name or code; raise ValueError when not unique."""
    text = str(value or "").strip()
    if not text:
        raise ValueError("请填写股票名称或代码")
    frame = _basic(runtime)
    match = _CODE.search(text.upper())
    if match:
        symbol, suffix = match.group(1), match.group(2)
        code = f"{symbol}.{suffix or _suffix(symbol)}"
        name = ""
        if frame is not None:
            hit = frame[frame["ts_code"] == code]
            if hit.empty and not suffix:
                hit = frame[frame["ts_code"].str.startswith(symbol + ".")]
                if len(hit) == 1:
                    code = hit["ts_code"].iloc[0]
            if not hit.empty:
                name = hit["name"].iloc[0]
        return code, name
    found = search_stocks(runtime, text, limit=6)
    exact = [s for s in found if s["name"] == text]
    if len(exact) == 1:
        return exact[0]["ts_code"], exact[0]["name"]
    if len(found) == 1:
        return found[0]["ts_code"], found[0]["name"]
    if not found:
        raise ValueError(f"没有找到「{text}」，请检查名称")
    listed = "、".join(f"{s['name']}（{s['ts_code']}）" for s in found[:5])
    raise ValueError(f"「{text}」匹配到多只：{listed}，请选具体的一只")
