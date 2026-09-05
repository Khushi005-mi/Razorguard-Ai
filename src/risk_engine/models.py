from dataclasses import dataclass
from typing import List


@dataclass
class RiskDecision:
    transaction_id: str
    risk_score: float
    risk_level: str
    action: str
    risk_signals: List[str]
    decision_reason: str
    model_version: str
    policy_version: str