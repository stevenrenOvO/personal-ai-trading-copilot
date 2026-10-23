"""Run the isolated V1.1 regression cases corresponding to the original audit.

Usage: python -m docs.v1_1_audit_repro
The V1.0 observations are archived in the Chinese audit report.
"""
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    raise SystemExit(subprocess.call([sys.executable, "-m", "pytest",
        "tests/test_v1_1_regression.py", "-q"], cwd=root))
