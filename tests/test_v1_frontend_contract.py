"""V1.0 frontend contract: the UI must use the Step-4 chain, not the baseline.

The old 20-day breakout page was once the main opportunity result; these
assertions stop it (or any re-computed client-side logic) from coming back.
"""

from __future__ import annotations

from pathlib import Path

INDEX = Path(__file__).resolve().parents[1] / "backend" / "dashboard" / "static" / "index.html"


def _html() -> str:
    return INDEX.read_text(encoding="utf-8")


def test_opportunity_pages_call_the_step4_plan_api():
    html = _html()
    assert "/opportunities/plan?date=" in html
    # The baseline scan must not be used as a main opportunity source anywhere.
    assert "`/opportunities?date=" not in html


def test_no_leftover_table_markup_for_mobile():
    html = _html()
    assert "<table" not in html          # tables are rendered as cards instead
    assert "@media" in html              # responsive rules exist
    assert "width=device-width" in html  # viewport is set


def test_page_states_the_execution_disclaimers():
    html = _html()
    for phrase in ("不等于买入建议", "可执行 ≠ 应该买", "T+1", "未接入"):
        assert phrase in html


def test_review_tab_has_paper_and_journal_forms():
    html = _html()
    for element in ("pt-code", "pt-submit", "dj-code", "dj-submit", "review-audit"):
        assert f'id="{element}"' in html


def test_no_debug_or_placeholder_markers_in_frontend():
    html = _html()
    for marker in ("TODO", "FIXME", "console.log(", "debugger"):
        assert marker not in html
