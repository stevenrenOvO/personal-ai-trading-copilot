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
    for element_id in ("ra-view", "wl-code", "wl-add", "wl-view", "pc-code", "pc-submit", "pc-result",
                       "rt-date", "rt-side", "rt-price", "rt-shares", "rt-submit", "rt-list",
                       "pl-date", "pl-form", "pl-submit"):
        assert f'id="{element_id}"' in HTML, element_id
    for endpoint in ("/personal/account", "/personal/watchlist/status", "/personal/pretrade",
                     "/personal/trades", "/personal/trades/void", "/personal/plan"):
        assert endpoint in SCRIPT, endpoint
    assert "loadDailyLoop();" in SCRIPT
