import pytest

from risk_engine.engine import RiskEngine
from risk_engine.policy import RiskPolicy

@pytest.fixture
def engine():
    return RiskEngine()


def sample_transaction():
    return {
        "transaction_id": "TX_TEST_001",
        "amount": 2500.0,
        "refund_requested": 1,
        "refund_amount": 2400.0,
        "refund_delay_hours": 2.0,
        "previous_orders": 8,
        "previous_refunds": 5,
        "historical_refund_rate": 0.625,
        "high_value_transaction": 1,
        "rapid_refund": 1,
        "refund_amount_ratio": 0.96,
        "refund_frequency_signal": 3,
    }


def test_engine_returns_decision(engine):

    decision = engine.evaluate(sample_transaction())

    assert decision.transaction_id == "TX_TEST_001"
    assert 0 <= decision.risk_score <= 1
    assert decision.risk_level in {"LOW", "MEDIUM", "HIGH"}
    assert decision.action in {
        "ALLOW",
        "MONITOR",
        "FLAG_FOR_REVIEW",
    }


def test_policy_thresholds():

    policy = RiskPolicy(
        medium_threshold=0.50,
        high_threshold=0.65,
    )

    engine = RiskEngine(policy=policy)

    assert engine.policy_engine.evaluate(0.30).risk_level == "LOW"
    assert engine.policy_engine.evaluate(0.30).action == "ALLOW"

    assert engine.policy_engine.evaluate(0.55).risk_level == "MEDIUM"
    assert engine.policy_engine.evaluate(0.55).action == "MONITOR"

    assert engine.policy_engine.evaluate(0.70).risk_level == "HIGH"
    assert engine.policy_engine.evaluate(0.70).action == "FLAG_FOR_REVIEW"

def test_missing_field_rejected(engine):

    transaction = sample_transaction()

    del transaction["amount"]

    with pytest.raises(ValueError):
        engine.evaluate(transaction)


def test_negative_amount_rejected(engine):

    transaction = sample_transaction()
    transaction["amount"] = -100

    with pytest.raises(ValueError):
        engine.evaluate(transaction)


def test_refund_cannot_exceed_amount(engine):

    transaction = sample_transaction()
    transaction["refund_amount"] = 5000

    with pytest.raises(ValueError):
        engine.evaluate(transaction)