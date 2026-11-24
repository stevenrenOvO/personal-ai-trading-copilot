"""Static checks of the dashboard script that run without Node.js.

The Node-based checks are skipped when Node is not installed. A duplicated
top-level ``const`` is a SyntaxError that stops the *whole* page script --
on 2026-11-17 a second ``const num`` blanked every tab -- so it is checked
here with plain Python.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

HTML = (Path(__file__).resolve().parents[1] / "backend" / "dashboard" / "static" / "index.html").read_text(encoding="utf-8")
SCRIPT = max(re.findall(r"<script>(.*?)</script>", HTML, re.S), key=len)


def test_no_duplicate_top_level_declarations():
    names = re.findall(r"^    (?:const|let|var|async function|function)\s+([A-Za-z_$][\w$]*)", SCRIPT, re.M)
    dupes = sorted(name for name, count in Counter(names).items() if count > 1)
    assert not dupes, f"declared more than once at top level: {dupes}"


def test_script_delimiters_balance():
    stripped = re.sub(r"//[^\n]*", "", SCRIPT)
    for open_, close in ("{}", "()", "[]"):
        assert stripped.count(open_) == stripped.count(close), open_ + close
    assert SCRIPT.count("`") % 2 == 0


def test_v12_daily_loop_controls_exist_and_are_wired():
    for element_id in ("wl-code", "wl-add", "wl-view", "pc-code", "pc-submit", "pc-result",
                       "pl-date", "pl-form", "pl-submit", "rp-chase", "stock-options",
                       "more-toggle", "more-menu", "pc-box", "pane-today", "pane-watch", "pane-plan", "pane-settings",
                       "td-hero", "td-setup", "td-holds", "td-actions", "date-prev", "date-next", "date-face", "us-view"):
        assert f'id="{element_id}"' in HTML, element_id
    for endpoint in ("/personal/account", "/personal/watchlist/status", "/personal/pretrade",
                     "/personal/plan", "/personal/today", "/personal/today/mark", "/personal/stocks",
                     "/personal/watchlist/hold", "/personal/watchlist/sell", "/personal/usage"):
        assert endpoint in SCRIPT, endpoint
    assert "loadPane(currentPane);" in SCRIPT and "/personal/today" in SCRIPT


def test_today_is_the_default_view():
    assert '<button data-tab="account" data-pane="today" class="active">今日</button>' in HTML
    assert '<section id="account" class="section active">' in HTML
    assert '<section id="dashboard" class="section">' in HTML
    assert 'class="pane active" id="pane-today"' in HTML
    for removed in ('id="ra-view"', 'id="rt-submit"', 'id="sz-submit"', 'data-pane="tools"'):
        assert removed not in HTML, removed
    assert "loadTab(activeTab())" not in SCRIPT or "if (first === 'account') loadTab(first);" in SCRIPT


def test_usage_events_used_in_the_page_are_valid_names():
    import re as _re

    events = set(_re.findall(r"track\('((?:tab|pane|action):[a-z_]+)'\)", SCRIPT))
    events |= set(_re.findall(r"'(action:[a-z_]+)'", SCRIPT))
    assert {"action:pretrade", "action:hold", "action:sell", "action:plan_save"} <= events
    assert all(_re.fullmatch(r"(tab|pane|action):[a-z_]{1,32}", e) for e in events)
    for tab in _re.findall(r'data-tab="([^"]+)"', HTML):
        assert _re.fullmatch(r"[a-z_]{1,32}", tab), tab  # tab:<name> must be a valid event


def test_navigation_keeps_daily_and_research_pages_on_top():
    nav = HTML[HTML.index('<nav id="tabs"'):HTML.index('</nav>')]
    main = nav[:nav.index('id="more-menu"')]
    assert [m for m in re.findall(r'>([^<>]+)</button>', main) if m.strip()] == [
        "今日", "我的股票", "计划", "市场", "机会", "涨停", "复盘", "更多 ▾"]
    more = nav[nav.index('id="more-menu"'):]
    for label in ("设置", "总览", "回测", "AI 助手", "策略健康"):
        assert f">{label}</button>" in more, label
    assert 'id="td-market"' in HTML and "renderPulse(" in SCRIPT
