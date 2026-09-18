"""Tests for the fallback quote source (Tencent) parsing."""

from __future__ import annotations

import requests

from backend.market.snapshot import fetch_full_market_snapshot_tencent


def _payload() -> str:
    fields = [""] * 49
    fields[0] = 'v_sh600000="1'
    fields[1] = "浦发银行"
    fields[2] = "600000"
    fields[3] = "9.06"
    fields[4] = "9.10"
    fields[5] = "9.10"
    fields[6] = "456711"
    fields[30] = "20260917161450"
    fields[31] = "-0.04"
    fields[32] = "-0.44"
    fields[33] = "9.14"
    fields[34] = "9.03"
    fields[35] = "9.06/456711/414887057"
    fields[36] = "456711"
    fields[37] = "41489"
    return "~".join(fields) + '";'


class _Resp:
    encoding = "gbk"

    def __init__(self, text: str) -> None:
        self.text = text


def test_tencent_snapshot_parsing(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp(_payload()))
    frame, meta = fetch_full_market_snapshot_tencent(["600000.SH"])
    assert len(frame) == 1
    row = frame.iloc[0]
    assert row["ts_code"] == "600000.SH"
    assert row["name"] == "浦发银行"
    assert row["price"] == 9.06
    assert row["pre_close"] == 9.10
    assert row["pct_chg"] == -0.44
    assert row["volume"] == 456711 * 100       # 手 -> 股
    assert row["amount"] == 41489 * 10000      # 万元 -> 元
    assert meta["source"] == "tencent"
    assert meta["quote_time"] == "16:14:50"


def test_fallback_chain_uses_tencent_when_sina_fails(monkeypatch):
    import backend.market.snapshot as snap

    def boom():
        raise RuntimeError("sina blocked")

    monkeypatch.setattr(snap, "fetch_full_market_snapshot_sina", boom)
    monkeypatch.setattr(
        snap, "fetch_full_market_snapshot_tencent", lambda: (frame_stub(), {"source": "tencent", "rows": 1})
    )
    frame, meta = snap.fetch_full_market_snapshot()
    assert meta["source"] == "tencent"
    assert "sina blocked" in meta["fallback_from"]
    assert len(frame) == 1


def frame_stub():
    import pandas as pd

    return pd.DataFrame({"ts_code": ["600000.SH"], "name": ["浦发银行"], "price": [9.06]})
