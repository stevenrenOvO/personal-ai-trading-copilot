"""Execute real dashboard handlers with deterministic, offline browser inputs."""

from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.parametrize("scenario", [
    "decision_fields",
    "decision_retry",
    "stale_date",
    "audit_failure",
    "failure_status",
    "date_selection_during_refresh",
    "refresh_waits_for_view",
    "unconfirmed_market",
    "health_without_samples",
    "available_dates",
    "dashboard_basis",
    "refresh_new_complete_day",
    "analysis_coverage_labels",
    "structured_ai_results",
    "structured_ai_empty_and_stale",
    "readiness_index_and_demo_labels",
    "unknown_ai_trade_permission",
])
def test_dashboard_interactions(scenario):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required to execute dashboard JavaScript")
    runner = Path(__file__).with_name("frontend_v11_checks.cjs")
    result = subprocess.run(
        [node, str(runner), scenario], capture_output=True, text=True,
        encoding="utf-8", timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
