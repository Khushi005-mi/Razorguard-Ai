from typing import Any

from src.db.connection import get_connection
from src.risk_engine.engine import RiskDecision


def create_risk_decision(decision: RiskDecision) -> dict[str, Any]:
    """
    Persist a risk decision and return the inserted decision record.

    The returned decision_id is required when creating the corresponding
    human review record for HIGH-risk transactions.
    """

    conn = None

    try:
        conn = get_connection()

        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO risk_decisions (
                    transaction_id,
                    risk_score,
                    risk_level,
                    action,
                    risk_signals,
                    decision_reason,
                    model_version,
                    policy_version
                )
                VALUES (
                    %s,
                    %s,
                    %s,
                    %s,
                    %s::jsonb,
                    %s,
                    %s,
                    %s
                )
                RETURNING
                    decision_id,
                    transaction_id,
                    risk_score,
                    risk_level,
                    action,
                    risk_signals,
                    decision_reason,
                    model_version,
                    policy_version,
                    decision_timestamp
                """,
                (
                    decision.transaction_id,
                    decision.risk_score,
                    decision.risk_level,
                    decision.action,
                    __import__("json").dumps(decision.risk_signals),
                    decision.decision_reason,
                    decision.model_version,
                    decision.policy_version,
                ),
            )

            row = cursor.fetchone()

        conn.commit()

        if row is None:
            raise RuntimeError(
                "Risk decision was inserted but no record was returned."
            )

        return {
            "decision_id": row[0],
            "transaction_id": row[1],
            "risk_score": float(row[2]),
            "risk_level": row[3],
            "action": row[4],
            "risk_signals": row[5],
            "decision_reason": row[6],
            "model_version": row[7],
            "policy_version": row[8],
            "decision_timestamp": row[9],
        }

    except Exception:
        if conn is not None:
            conn.rollback()
        raise

    finally:
        if conn is not None:
            conn.close()

