from dataclasses import dataclass


@dataclass(frozen=True)
class PolicyDecision:
    risk_level: str
    action: str
    decision_reason: str


@dataclass(frozen=True)
class RiskPolicy:
    medium_threshold: float = 0.50
    high_threshold: float = 0.65
    policy_version: str = "policy_v1"


class PolicyEngine:

    def __init__(self, policy: RiskPolicy | None = None):
        self.policy = policy or RiskPolicy()

        if not (
            0 <= self.policy.medium_threshold
            < self.policy.high_threshold
            <= 1
        ):
            raise ValueError(
                "Policy thresholds must satisfy: 0 <= medium < high <= 1"
            )

    def evaluate(self, risk_score: float) -> PolicyDecision:

        if not 0 <= risk_score <= 1:
            raise ValueError("Risk score must be between 0 and 1")

        if risk_score >= self.policy.high_threshold:
            return PolicyDecision(
                risk_level="HIGH",
                action="FLAG_FOR_REVIEW",
                decision_reason=(
                    "Risk score exceeded the high-risk review threshold."
                ),
            )

        if risk_score >= self.policy.medium_threshold:
            return PolicyDecision(
                risk_level="MEDIUM",
                action="MONITOR",
                decision_reason=(
                    "Risk score exceeded the monitoring threshold "
                    "but remained below the high-risk threshold."
                ),
            )

        return PolicyDecision(
            risk_level="LOW",
            action="ALLOW",
            decision_reason=(
                "Risk score remains below the monitoring threshold."
            ),
        )