from src.db.connection import get_connection


def get_dashboard_summary() -> dict:
    conn = None

    try:
        conn = get_connection()

        with conn.cursor() as cursor:

            # Dashboard counts
            cursor.execute(
                """
                SELECT
                    COUNT(*) AS total_transactions,
                    COUNT(*) FILTER (
                        WHERE risk_level = 'HIGH'
                    ) AS high_risk,
                    COUNT(*) FILTER (
                        WHERE risk_level = 'MEDIUM'
                    ) AS medium_risk,
                    COUNT(*) FILTER (
                        WHERE risk_level = 'LOW'
                    ) AS low_risk
                FROM risk_decisions;
                """
            )

            counts = cursor.fetchone()

            # Recent decisions
            cursor.execute(
                """
                SELECT
                    transaction_id,
                    risk_score,
                    risk_level,
                    action,
                    risk_signals,
                    decision_reason,
                    model_version,
                    policy_version,
                    decision_timestamp
                FROM risk_decisions
                ORDER BY decision_timestamp DESC
                LIMIT 10;
                """
            )

            rows = cursor.fetchall()

        recent_decisions = [
            {
                "transaction_id": row[0],
                "risk_score": float(row[1]),
                "risk_level": row[2],
                "action": row[3],
                "risk_signals": row[4],
                "decision_reason": row[5],
                "model_version": row[6],
                "policy_version": row[7],
                "decision_timestamp": row[8],
            }
            for row in rows
        ]

        return {
            "total_transactions": counts[0],
            "high_risk": counts[1],
            "medium_risk": counts[2],
            "low_risk": counts[3],
            "recent_decisions": recent_decisions,
        }

    finally:
        if conn is not None:
            conn.close()