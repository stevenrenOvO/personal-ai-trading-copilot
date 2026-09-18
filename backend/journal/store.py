"""JSONL persistence for the decision journal and audit trail."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backend.data.config import get_settings
from backend.journal.models import AuditEvent, DecisionRecord


class JsonlStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def append(self, obj: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def load_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows


class DecisionJournal:
    def __init__(self, root: Path | None = None) -> None:
        resolved_root = (Path(root) if root is not None else get_settings().db_path.parent).resolve()
        self.store = JsonlStore(resolved_root / "journal" / "decisions.jsonl")

    def record(self, record: DecisionRecord) -> DecisionRecord:
        self.store.append(record.to_dict())
        return record

    def list(
        self,
        *,
        ts_code: str | None = None,
        trade_date: str | None = None,
        decision: str | None = None,
    ) -> list[DecisionRecord]:
        records = [DecisionRecord.from_dict(row) for row in self.store.load_all()]
        if ts_code:
            records = [item for item in records if item.ts_code == ts_code]
        if trade_date:
            records = [item for item in records if item.trade_date == trade_date]
        if decision:
            records = [item for item in records if item.decision == decision]
        return records


class DecisionAudit:
    def __init__(self, root: Path | None = None) -> None:
        resolved_root = (Path(root) if root is not None else get_settings().db_path.parent).resolve()
        self.store = JsonlStore(resolved_root / "audit" / "events.jsonl")

    def record_event(
        self,
        *,
        event_type: str,
        actor: str,
        target: str,
        payload: dict[str, Any] | None = None,
    ) -> AuditEvent:
        event = AuditEvent.create(
            event_type=event_type,
            actor=actor,
            target=target,
            payload=payload,
        )
        self.store.append(event.to_dict())
        return event

    def list(self, *, event_type: str | None = None, actor: str | None = None) -> list[AuditEvent]:
        events = [AuditEvent.from_dict(row) for row in self.store.load_all()]
        if event_type:
            events = [item for item in events if item.event_type == event_type]
        if actor:
            events = [item for item in events if item.actor == actor]
        return events
