"""Immutable backtest results, retained across local service restarts."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import re

from backend.data.atomic import atomic_write, file_lock


class BacktestRunStore:
    """Persist complete runs; listing returns the newest 50 at most."""

    MAX_LISTED = 50

    def __init__(self, root: Path) -> None:
        self.path = Path(root) / "backtest" / "runs"
        self._lock_path = self.path / ".runs"

    @staticmethod
    def _run_id(value) -> str:
        if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", value):
            raise ValueError("回测编号只能包含字母、数字、下划线和连字符")
        return value

    @classmethod
    def _validate(cls, run) -> datetime:
        if not isinstance(run, dict):
            raise ValueError("回测记录必须是 JSON 对象")
        cls._run_id(run.get("run_id"))
        value = run.get("created_at")
        if not isinstance(value, str):
            raise ValueError("回测记录缺少 created_at")
        try:
            created_at = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("回测 created_at 必须是带时区的 ISO 时间") from exc
        if created_at.tzinfo is None or created_at.utcoffset() is None:
            raise ValueError("回测 created_at 必须包含时区")
        return created_at

    @staticmethod
    def _invalid_constant(value):
        raise ValueError(f"invalid JSON number: {value}")

    def _read(self, path: Path) -> dict:
        try:
            run = json.loads(path.read_text(encoding="utf-8"), parse_constant=self._invalid_constant)
            self._validate(run)
            # JSON exponents such as 1e999 parse as infinity without invoking
            # parse_constant. Apply the same finite-number rule as save().
            json.dumps(run, allow_nan=False)
            if run["run_id"] != path.stem:
                raise ValueError("回测编号与文件名不一致")
            return run
        except (OSError, ValueError, TypeError, OverflowError) as exc:
            raise RuntimeError(f"回测记录无法读取，请检查 {path.name}；原文件已保留") from exc

    def save(self, run: dict) -> dict:
        """Save once; conflicting IDs and malformed existing runs never overwrite."""
        self._validate(run)
        # Validate the entire result before taking the lock or touching a run.
        text = json.dumps(run, ensure_ascii=False, allow_nan=False)
        normalized = json.loads(text)
        path = self.path / f"{self._run_id(normalized['run_id'])}.json"
        with file_lock(self._lock_path):
            if path.exists():
                previous = self._read(path)
                if previous != normalized:
                    raise ValueError("相同 run_id 不能覆盖已有回测结果")
                return previous
            atomic_write(path, text)
        return normalized

    def get(self, run_id: str) -> dict | None:
        """Read the complete result, including one outside the recent-run list."""
        path = self.path / f"{self._run_id(run_id)}.json"
        with file_lock(self._lock_path):
            return self._read(path) if path.exists() else None

    def list(self, limit: int = MAX_LISTED) -> list[dict]:
        """Return newest first, without pruning older complete results."""
        if type(limit) is not int or limit < 0:
            raise ValueError("回测列表条数必须为非负整数")
        with file_lock(self._lock_path):
            runs = [self._read(path) for path in self.path.glob("*.json")]
        runs.sort(key=lambda run: (self._validate(run), run["run_id"]), reverse=True)
        return runs[:min(limit, self.MAX_LISTED)]


# Public shorthand used by the API; retain the descriptive name for callers.
BacktestStore = BacktestRunStore
