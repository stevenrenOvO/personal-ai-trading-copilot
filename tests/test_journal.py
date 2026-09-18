"""Tests for decision journal and audit trail."""

import tempfile
from pathlib import Path

import pytest

from backend.journal.models import AuditEvent, DecisionRecord
from backend.journal.store import DecisionAudit, DecisionJournal


@pytest.fixture
def journal_root():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


def _decision(**overrides):
    values = {
        "trade_date": "20240103",
        "ts_code": "000001.SZ",
        "action": "BUY",
        "signal_strength": 0.8,
        "risk_allowed": True,
        "risk_violations": [],
        "decision": "APPROVED",
        "reason": "breakout",
    }
    values.update(overrides)
    return DecisionRecord.create(**values)


def test_decision_record_roundtrip():
    rec = _decision()
    assert rec.decision_id.startswith("decision_")
    assert rec.ts_code == "000001.SZ"
    data = rec.to_dict()
    assert DecisionRecord.from_dict(data) == rec


def test_audit_event_roundtrip():
    event = AuditEvent.create(
        event_type="decision_created",
        actor="decision_engine",
        target="000001.SZ",
        payload={"decision": "APPROVED"},
    )
    assert event.event_type == "decision_created"
    assert AuditEvent.from_dict(event.to_dict()) == event


def test_decision_journal_save_and_list(journal_root):
    journal = DecisionJournal(journal_root)
    rec = _decision()
    journal.record(rec)

    records = journal.list()
    assert len(records) == 1
    assert records[0].ts_code == "000001.SZ"


def test_decision_journal_filters(journal_root):
    journal = DecisionJournal(journal_root)
    journal.record(_decision(ts_code="000001.SZ", decision="APPROVED"))
    journal.record(_decision(ts_code="600000.SH", decision="REJECTED"))

    assert len(journal.list(ts_code="000001.SZ")) == 1
    assert len(journal.list(decision="REJECTED")) == 1
    assert len(journal.list(trade_date="20240104")) == 0


def test_decision_audit_records_and_filters(journal_root):
    audit = DecisionAudit(journal_root)
    audit.record_event(
        event_type="decision_created",
        actor="test",
        target="000001.SZ",
        payload={"a": 1},
    )
    audit.record_event(
        event_type="decision_rejected",
        actor="test",
        target="600000.SH",
    )

    assert len(audit.list()) == 2
    assert len(audit.list(event_type="decision_created")) == 1
    assert len(audit.list(actor="other")) == 0

