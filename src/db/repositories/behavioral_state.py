from datetime import datetime

from src.db.connection import get_connection


def get_customer_behavioral_state(
    customer_id: str,
    transaction_timestamp: datetime,
) -> dict:
    """
    Calculate customer behavioral features using only transactions
    that occurred before the current transaction.
    """

    query = """
        SELECT
            COUNT(*) AS previous_orders,

            COUNT(*) FILTER (
                WHERE refund_requested = TRUE
            ) AS previous_refunds,

            COALESCE(
                COUNT(*) FILTER (
                    WHERE refund_requested = TRUE
                )::float
                / NULLIF(COUNT(*), 0),
                0
            ) AS historical_refund_rate,

            (
    COUNT(*) FILTER (
        WHERE refund_requested = TRUE
    )
    *
    COALESCE(
        COUNT(*) FILTER (
            WHERE refund_requested = TRUE
        )::float
        / NULLIF(COUNT(*), 0),
        0
    )
) AS refund_frequency_signal
        FROM transactions
        WHERE customer_id = %s
          AND timestamp < %s;
    """

    conn = None

    try:
        conn = get_connection()

        with conn.cursor() as cursor:
            cursor.execute(
                query,
                (
                    transaction_timestamp,
                    customer_id,
                    transaction_timestamp,
                ),
            )

            row = cursor.fetchone()

        if row is None:
            return {
                "previous_orders": 0,
                "previous_refunds": 0,
                "historical_refund_rate": 0.0,
                "refund_frequency_signal": 0,
            }

        return {
            "previous_orders": int(row[0]),
            "previous_refunds": int(row[1]),
            "historical_refund_rate": float(row[2]),
            "refund_frequency_signal": int(row[3]),
        }

    finally:
        if conn is not None:
            conn.close()
