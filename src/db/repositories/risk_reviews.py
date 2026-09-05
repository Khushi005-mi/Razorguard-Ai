from datetime import datetime, timezone

from src.db.connection import get_connection


def create_risk_review(
    transaction_id: str,
    risk_decision_id: int,
) -> dict:
    conn = None

    try:
        conn = get_connection()

        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO risk_reviews (
                    transaction_id,
                    risk_decision_id
                )
                VALUES (%s, %s)
                RETURNING
                    review_id,
                    transaction_id,
                    risk_decision_id,
                    status,
                    reviewer,
                    review_reason,
                    reviewed_at,
                    created_at;
                """,
                (transaction_id, risk_decision_id),
            )

            row = cursor.fetchone()

        conn.commit()

        return {
            "review_id": row[0],
            "transaction_id": row[1],
            "risk_decision_id": row[2],
            "status": row[3],
            "reviewer": row[4],
            "review_reason": row[5],
            "reviewed_at": row[6],
            "created_at": row[7],
        }

    except Exception:
        if conn is not None:
            conn.rollback()
        raise

    finally:
        if conn is not None:
            conn.close()


def get_pending_reviews() -> list[dict]:
    conn = None

    try:
        conn = get_connection()

        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    rr.review_id,
                    rr.transaction_id,
                    rr.risk_decision_id,
                    rr.status,
                    rr.reviewer,
                    rr.review_reason,
                    rr.reviewed_at,
                    rr.created_at,
                    rd.risk_score,
                    rd.risk_level,
                    rd.action,
                    rd.risk_signals,
                    rd.decision_reason
                FROM risk_reviews rr
                JOIN risk_decisions rd
                    ON rr.risk_decision_id = rd.decision_id
                WHERE rr.status = 'PENDING'
                ORDER BY rd.risk_score DESC, rr.created_at DESC;
                """
            )

            rows = cursor.fetchall()

        return [
            {
                "review_id": row[0],
                "transaction_id": row[1],
                "risk_decision_id": row[2],
                "status": row[3],
                "reviewer": row[4],
                "review_reason": row[5],
                "reviewed_at": row[6],
                "created_at": row[7],
                "risk_score": float(row[8]),
                "risk_level": row[9],
                "action": row[10],
                "risk_signals": row[11],
                "decision_reason": row[12],
            }
            for row in rows
        ]

    finally:
        if conn is not None:
            conn.close()


def update_risk_review(
    review_id: int,
    status: str,
    reviewer: str,
    review_reason: str,
) -> dict | None:
    allowed_statuses = {"CONFIRMED_ABUSE", "FALSE_POSITIVE"}

    if status not in allowed_statuses:
        raise ValueError("Invalid review status.")

    conn = None

    try:
        conn = get_connection()

        with conn.cursor() as cursor:
            cursor.execute(
                """
                UPDATE risk_reviews
                SET
                    status = %s,
                    reviewer = %s,
                    review_reason = %s,
                    reviewed_at = %s
                WHERE review_id = %s
                  AND status = 'PENDING'
                RETURNING
                    review_id,
                    transaction_id,
                    risk_decision_id,
                    status,
                    reviewer,
                    review_reason,
                    reviewed_at,
                    created_at;
                """,
                (
                    status,
                    reviewer,
                    review_reason,
                    datetime.now(timezone.utc),
                    review_id,
                ),
            )

            row = cursor.fetchone()

        if row is None:
            conn.rollback()
            return None

        conn.commit()

        return {
            "review_id": row[0],
            "transaction_id": row[1],
            "risk_decision_id": row[2],
            "status": row[3],
            "reviewer": row[4],
            "review_reason": row[5],
            "reviewed_at": row[6],
            "created_at": row[7],
        }

    except Exception:
        if conn is not None:
            conn.rollback()
        raise

    finally:
        if conn is not None:
            conn.close()