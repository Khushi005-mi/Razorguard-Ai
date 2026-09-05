from pathlib import Path

import joblib
import pandas as pd

from .models import RiskDecision
from .policy import PolicyEngine, RiskPolicy
from .reasons import generate_reasons


FEATURES = [
    "amount",
    "refund_requested",
    "refund_amount",
    "refund_delay_hours",
    "previous_orders",
    "previous_refunds",
    "historical_refund_rate",
    "high_value_transaction",
    "rapid_refund",
    "refund_amount_ratio",
    "refund_frequency_signal",
]


class RiskEngine:
    def __init__(
        self,
        model_path: str = "models/logistic_regression_baseline.joblib",
        model_version: str = "logistic_v1",
        policy: RiskPolicy | None = None,
    ):
        self.model_path = Path(model_path)
        self.model_version = model_version

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}"
            )

        self.model = joblib.load(self.model_path)
        self.policy_engine = PolicyEngine(policy)

    def validate_transaction(self, transaction: dict) -> None:
        required_fields = ["transaction_id", *FEATURES]

        missing = [
            field
            for field in required_fields
            if field not in transaction
        ]

        if missing:
            raise ValueError(
                f"Missing required fields: {missing}"
            )

        if transaction["amount"] < 0:
            raise ValueError(
                "Transaction amount cannot be negative"
            )

        if transaction["refund_amount"] < 0:
            raise ValueError(
                "Refund amount cannot be negative"
            )

        if transaction["refund_amount"] > transaction["amount"]:
            raise ValueError(
                "Refund amount cannot exceed transaction amount"
            )

    def predict_risk(self, transaction: dict) -> float:
        features = pd.DataFrame(
            [[transaction[feature] for feature in FEATURES]],
            columns=FEATURES,
        )

        risk_score = self.model.predict_proba(features)[0, 1]

        return float(risk_score)

    def evaluate(self, transaction: dict) -> RiskDecision:
        self.validate_transaction(transaction)

        risk_score = self.predict_risk(transaction)

        policy_decision = self.policy_engine.evaluate(
            risk_score
        )

        risk_signals = generate_reasons(transaction)

        return RiskDecision(
            transaction_id=transaction["transaction_id"],
            risk_score=round(risk_score, 6),
            risk_level=policy_decision.risk_level,
            action=policy_decision.action,
            risk_signals=risk_signals,
            decision_reason=policy_decision.decision_reason,
            model_version=self.model_version,
            policy_version=self.policy_engine.policy.policy_version,
        )