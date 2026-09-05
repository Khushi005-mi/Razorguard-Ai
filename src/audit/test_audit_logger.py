from risk_engine.models import RiskDecision

from audit.logger import AuditLogger


def test_audit_record_is_persisted(tmp_path):

    log_path = tmp_path / "decisions.jsonl"

    logger = AuditLogger(
        log_path=str(log_path)
    )

    decision = RiskDecision(
        transaction_id="TX_AUDIT_001",
        risk_score=0.72,
        risk_level="HIGH",
        action="FLAG_FOR_REVIEW",
        risk_signals=[
            "High historical refund rate",
            "Frequent refund activity",
        ],
        decision_reason=(
            "Risk score exceeded the high-risk review threshold."
        ),
        model_version="logistic_v1",
        policy_version="policy_v1",
    )

    record = logger.record(decision)

    assert record is not None
    assert record.transaction_id == "TX_AUDIT_001"
    assert record.risk_score == 0.72
    assert record.risk_level == "HIGH"
    assert record.action == "FLAG_FOR_REVIEW"

    assert record.risk_signals == [
        "High historical refund rate",
        "Frequent refund activity",
    ]

    assert record.decision_reason == (
        "Risk score exceeded the high-risk review threshold."
    )

    assert log_path.exists()

    content = log_path.read_text()

    assert "TX_AUDIT_001" in content
    assert "High historical refund rate" in content
    assert (
        "Risk score exceeded the high-risk review threshold."
        in content
    )


def test_duplicate_transaction_is_not_logged_twice(tmp_path):

    log_path = tmp_path / "decisions.jsonl"

    logger = AuditLogger(
        log_path=str(log_path)
    )

    decision = RiskDecision(
        transaction_id="TX_DUPLICATE_001",
        risk_score=0.72,
        risk_level="HIGH",
        action="FLAG_FOR_REVIEW",
        risk_signals=[
            "Frequent refund activity",
        ],
        decision_reason=(
            "Risk score exceeded the high-risk review threshold."
        ),
        model_version="logistic_v1",
        policy_version="policy_v1",
    )

    first_record = logger.record(decision)
    second_record = logger.record(decision)

    assert first_record is not None
    assert second_record is None

    lines = log_path.read_text().strip().splitlines()

    assert len(lines) == 1