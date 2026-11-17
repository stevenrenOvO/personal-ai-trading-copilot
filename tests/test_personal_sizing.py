"""V1.2 personal account: risk profile and stop-distance position sizing.

Expected numbers are worked out by hand for a 12,000-yuan account so the
tests fail if the sizing ever drifts from the user's stated rules.
"""

from __future__ import annotations

import json

import pytest

from backend.personal import RiskProfile, RiskProfileStore, size_position
from tests.test_api_v1 import _call, _make_runtime


def test_default_profile_matches_stated_rules():
    p = RiskProfile().validate()
    assert p.capital == 12_000
    assert (p.risk_per_trade, p.max_risk_per_trade) == (0.01, 0.02)
    assert p.max_single_weight == 0.20
    assert (p.first_entry_min, p.first_entry_max) == (0.10, 0.15)
    assert p.exposure_caps["weak"] == 0.20 and p.exposure_caps["range"] == 0.50
    assert p.lot_size == 100


def test_one_lot_when_risk_and_first_entry_bind():
    # entry 10.00, stop 9.30 (7%), 1% risk = 120 yuan.
    # 1 lot: 70 loss + 5 + 5 commission + 0.465 stamp = 80.47 -> ok
    # 2 lots: 140 + 5 + 5 + 0.93 = 150.93 > 120, and 2,005 > 1,800 first entry.
    r = size_position(RiskProfile(), 10.0, 9.3)
    assert r["verdict"] == "可买"
    assert r["lots"] == 1 and r["shares"] == 100
    assert r["risk_amount"] == pytest.approx(80.465, abs=0.01)
    assert r["stop_pct"] == pytest.approx(0.07)
    assert any("单笔风险" in x for x in r["limited_by"])
    assert any("首仓上限" in x for x in r["limited_by"])
    assert any("低于首仓下限" in w for w in r["warnings"])


def test_first_entry_cap_binds_before_risk():
    # entry 6, stop 5.7 (5%): 3 lots risk 90 + 10 + 0.855 = 100.86 <= 120 and
    # 1,805 <= 2,400 single cap, but 1,805 > 1,800 first-entry cap -> 2 lots.
    r = size_position(RiskProfile(), 6, 5.7)
    assert r["lots"] == 2 and r["value"] == 1200.0
    assert r["limited_by"] == ["首仓上限 15%"]


def test_expensive_stock_cannot_be_bought_and_says_why():
    # entry 25, stop 23.5: one lot 2,500 > single 2,400 and first-entry 1,800;
    # one-lot loss 150 + 10 + 1.175 = 161.18 > 120.
    r = size_position(RiskProfile(), 25, 23.5)
    assert r["verdict"] == "不可买" and r["lots"] == 0
    rules = {b["rule"] for b in r["blocked_by"]}
    assert rules == {"risk", "single_weight", "first_entry"}
    assert r["one_lot"]["risk_amount"] == pytest.approx(161.18, abs=0.01)
    assert r["one_lot_needs_risk_pct"] == pytest.approx(161.175 / 12000, abs=1e-4)


def test_weak_market_exposure_room_blocks_new_position():
    # weak regime cap 20% = 2,400; already 2,000 held -> 400 room < one lot 805.
    r = size_position(RiskProfile(), 8, 7.5, market_state="弱势", current_exposure=2000)
    assert r["verdict"] == "不可买"
    assert {b["rule"] for b in r["blocked_by"]} == {"exposure"}
    assert r["inputs"]["regime"] == "weak"


def test_adding_to_existing_position_uses_single_cap_not_first_entry():
    # 1,600 already in this stock: single cap room 800 -> one 8-yuan lot (805) does not fit.
    r = size_position(RiskProfile(), 8, 7.5, existing_position=1600, current_exposure=1600)
    assert r["verdict"] == "不可买"
    assert {b["rule"] for b in r["blocked_by"]} == {"single_weight"}
    # 1,000 held -> room 1,400 -> one lot fits, two do not.
    r = size_position(RiskProfile(), 8, 7.5, existing_position=1000, current_exposure=1000)
    assert r["lots"] == 1


def test_cash_and_wide_stop_warning():
    r = size_position(RiskProfile(), 10, 9.0, available_cash=500)
    assert r["verdict"] == "不可买"
    assert "cash" in {b["rule"] for b in r["blocked_by"]}
    assert any("超过参考上限" in w for w in r["warnings"])


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError, match="止损价必须低于买入价"):
        size_position(RiskProfile(), 10, 10)
    with pytest.raises(ValueError, match="超过上限"):
        size_position(RiskProfile(), 10, 9.5, risk_pct=0.03)
    with pytest.raises(ValueError):
        size_position(RiskProfile(), "abc", 9)
    with pytest.raises(ValueError):
        size_position(RiskProfile(), float("nan"), 9)


def test_store_defaults_update_and_validation(tmp_path):
    store = RiskProfileStore(tmp_path)
    profile, source = store.load()
    assert source == "default" and profile.capital == 12_000
    saved = store.update({"capital": 15000, "exposure_caps": {"strong": 0.8}})
    assert saved.capital == 15000 and saved.exposure_caps == {"strong": 0.8, "range": 0.5, "weak": 0.2}
    assert saved.updated_at
    loaded, source = store.load()
    assert source == "saved" and loaded.capital == 15000
    before = store.path.read_text(encoding="utf-8")
    with pytest.raises(ValueError):
        store.update({"risk_per_trade": 0.05})  # above the 2% hard maximum
    with pytest.raises(ValueError, match="未知的风控参数"):
        store.update({"leverage": 2})
    with pytest.raises(ValueError):
        store.update({"capital": -1})
    assert store.path.read_text(encoding="utf-8") == before


def test_personal_api(tmp_path):
    runtime = _make_runtime(tmp_path)
    code, data = _call("GET", "/personal/risk-profile", runtime)
    assert code == 200 and data["meta"]["source"] == "default"

    code, data = _call("GET", "/personal/sizing?entry=10&stop=9.3", runtime)
    assert code == 200 and data["data"]["lots"] == 1
    assert data["profile_source"] == "default"

    code, data = _call("GET", "/personal/sizing?entry=10", runtime)
    assert code == 400

    code, data = _call("POST", "/personal/risk-profile", runtime, {"capital": 20000})
    assert code == 200 and data["data"]["capital"] == 20000
    saved = json.loads((tmp_path / "personal" / "risk_profile.json").read_text(encoding="utf-8"))
    assert saved["capital"] == 20000

    code, data = _call("POST", "/personal/risk-profile", runtime, {"max_single_weight": 2})
    assert code == 400
