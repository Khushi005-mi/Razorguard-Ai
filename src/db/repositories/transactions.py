from datetime import datetime

from src.api.schemas import TransactionCreate
from src.db.connection import get_connection


def create_transaction(transaction: TransactionCreate) -> dict:
    """
    Persist a validated transaction in PostgreSQL.
    """

    query = """
        INSERT INTO transactions (
            transaction_id,
            customer_id,
            timestamp,
            amount,
            payment_method,
            refund_requested,
            refund_amount,
            refund_delay_hours,
            previous_orders,
            previous_refunds,
            historical_refund_rate,
            high_value_transaction,
            rapid_refund,
            refund_amount_ratio,
            refund_frequency_signal
        )
        VALUES (
            %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s
        )
        RETURNING
            transaction_id,
            customer_id,
            timestamp,
            amount,
            payment_method,
            refund_requested,
            refund_amount,
            refund_delay_hours,
            previous_orders,
            previous_refunds,
            historical_refund_rate,
            high_value_transaction,
            rapid_refund,
            refund_amount_ratio,
            refund_frequency_signal,
            created_at;
    """

    values = (
        transaction.transaction_id,
        transaction.customer_id,
        transaction.timestamp,
        transaction.amount,
        transaction.payment_method,
        transaction.refund_requested,
        transaction.refund_amount,
        transaction.refund_delay_hours,
        transaction.previous_orders,
        transaction.previous_refunds,
        transaction.historical_refund_rate,
        transaction.high_value_transaction,
        transaction.rapid_refund,
        transaction.refund_amount_ratio,
        transaction.refund_frequency_signal,
    )

    conn = None

    try:
        conn = get_connection()

        with conn.cursor() as cursor:
            cursor.execute(query, values)
            row = cursor.fetchone()

        conn.commit()

        columns = [
            "transaction_id",
            "customer_id",
            "timestamp",
            "amount",
            "payment_method",
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
            "created_at",
        ]

        return dict(zip(columns, row))

    except Exception:
        if conn is not None:
            conn.rollback()
        raise

    finally:
        if conn is not None:
            conn.close()
def get_transaction(transaction_id: str) -> dict | None:
    query = """
        SELECT
            transaction_id,
            customer_id,
            timestamp,
            amount,
            payment_method,
            refund_requested,
            refund_amount,
            refund_delay_hours,
            previous_orders,
            previous_refunds,
            historical_refund_rate,
            high_value_transaction,
            rapid_refund,
            refund_amount_ratio,
            refund_frequency_signal
        FROM transactions
        WHERE transaction_id = %s;
    """

    conn = None

    try:
        conn = get_connection()

        with conn.cursor() as cursor:
            cursor.execute(query, (transaction_id,))
            row = cursor.fetchone()

        if row is None:
            return None

        columns = [
            "transaction_id",
            "customer_id",
            "timestamp",
            "amount",
            "payment_method",
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

        return dict(zip(columns, row))

    finally:
        if conn is not None:
            conn.close()