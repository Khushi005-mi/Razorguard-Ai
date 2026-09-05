import pytest

from risk_engine.policy import PolicyEngine, RiskPolicy


def test_low_risk():

    engine = PolicyEngine()

    decision = engine.evaluate(0.30)

    assert decision.risk_level == "LOW"
    assert decision.action == "ALLOW"


def test_medium_risk():

    engine = PolicyEngine()

    decision = engine.evaluate(0.55)

    assert decision.risk_level == "MEDIUM"
    assert decision.action == "MONITOR"


def test_high_risk():

    engine = PolicyEngine()

    decision = engine.evaluate(0.70)

    assert decision.risk_level == "HIGH"
    assert decision.action == "FLAG_FOR_REVIEW"


def test_custom_policy():

    policy = RiskPolicy(
        medium_threshold=0.40,
        high_threshold=0.80,
    )

    engine = PolicyEngine(policy)

    assert engine.evaluate(0.50).risk_level == "MEDIUM"
    assert engine.evaluate(0.85).risk_level == "HIGH"


def test_invalid_policy():

    with pytest.raises(ValueError):

        PolicyEngine(
            RiskPolicy(
                medium_threshold=0.80,
                high_threshold=0.60,
            )
        )


def test_invalid_score():

    engine = PolicyEngine()

    with pytest.raises(ValueError):
        engine.evaluate(1.5)