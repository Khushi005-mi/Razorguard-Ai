import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from risk_engine.models import RiskDecision

from .models import AuditRecord


class AuditLogger:

    def __init__(
        self,
        log_path: str = "data/audit/decisions.jsonl",
    ):
        self.log_path = Path(log_path)

        self.log_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Temporary idempotency mechanism for the JSONL prototype.
        # Production will use a database-level unique constraint.
        self._existing_transaction_ids = set()

        if self.log_path.exists():

            with self.log_path.open(
                "r",
                encoding="utf-8",
            ) as file:

                for line in file:

                    if not line.strip():
                        continue

                    record = json.loads(line)

                    transaction_id = record.get(
                        "transaction_id"
                    )

                    if transaction_id:
                        self._existing_transaction_ids.add(
                            transaction_id
                        )

    def record(
        self,
        decision: RiskDecision,
    ) -> AuditRecord | None:

        # Prevent duplicate persistence.
        if (
            decision.transaction_id
            in self._existing_transaction_ids
        ):
            return None

        audit_record = AuditRecord(
            transaction_id=decision.transaction_id,
            risk_score=decision.risk_score,
            risk_level=decision.risk_level,
            action=decision.action,
            risk_signals=decision.risk_signals,
            decision_reason=decision.decision_reason,
            model_version=decision.model_version,
            policy_version=decision.policy_version,
            decision_timestamp=datetime.now(timezone.utc),
        )

        with self.log_path.open(
            "a",
            encoding="utf-8",
        ) as file:

            file.write(
                json.dumps(
                    asdict(audit_record),
                    default=str,
                )
                + "\n"
            )

        self._existing_transaction_ids.add(
            decision.transaction_id
        )

        return audit_record