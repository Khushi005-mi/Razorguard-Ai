from datetime import datetime, timezone
from typing import Any

from src.db.connection import get_connection


def normalize_refund_webhook(
    payment_id: str,
    refund_amount: float,
    refund_id: str,
    refund_created_at: int,
) -> dict[str, Any] | None:
    """
    Apply refund information from a Razorpay webhook to an
    existing internal transaction.

    The payment_id must already exist as transaction_id in the
    internal transactions table.

    refund_created_at is the Unix timestamp supplied by Razorpay.
    It is used to derive refund_delay_hours from the internal
    transaction timestamp.

    Returns the updated transaction or None when no matching
    transaction exists.
    """

    conn = None

    try:
        conn = get_connection()

        with conn.cursor() as cursor:
            # ------------------------------------------------
            # 1. Find the internal transaction
            # ------------------------------------------------

            cursor.execute(
                """
                SELECT
                    transaction_id,
                    customer_id,
                    amount,
                    refund_amount,
                    timestamp
                FROM transactions
                WHERE transaction_id = %s;
                """,
                (payment_id,),
            )

            row = cursor.fetchone()

            if row is None:
                conn.rollback()
                return None

            transaction_id = row[0]
            customer_id = row[1]
            transaction_amount = float(row[2])
            existing_refund_amount = float(row[3])
            transaction_timestamp = row[4]

            # ------------------------------------------------
            # 2. Convert Razorpay Unix timestamp to UTC
            # ------------------------------------------------

            refund_timestamp = datetime.fromtimestamp(
                refund_created_at,
                tz=timezone.utc,
            )

            # ------------------------------------------------
            # 3. Calculate refund delay
            # ------------------------------------------------

            refund_delay_hours = (
                refund_timestamp - transaction_timestamp
            ).total_seconds() / 3600

            # A refund cannot logically occur before the
            # transaction. Treat such webhook data as invalid.
            if refund_delay_hours < 0:
                raise ValueError(
                    "Refund timestamp cannot be earlier than "
                    "transaction timestamp."
                )

            # ------------------------------------------------
            # 4. Prevent refund amount from decreasing
            # ------------------------------------------------

            new_refund_amount = max(
                existing_refund_amount,
                refund_amount,
            )

            if new_refund_amount > transaction_amount:
                raise ValueError(
                    "Refund amount cannot exceed transaction amount."
                )

            # ------------------------------------------------
            # 5. Calculate refund amount ratio
            # ------------------------------------------------

            refund_amount_ratio = (
                new_refund_amount / transaction_amount
                if transaction_amount > 0
                else 0.0
            )

            # ------------------------------------------------
            # 6. Update internal transaction
            # ------------------------------------------------

            cursor.execute(
                """
                UPDATE transactions
                SET
                    refund_requested = TRUE,
                    refund_amount = %s,
                    refund_amount_ratio = %s,
                    refund_delay_hours = %s
                WHERE transaction_id = %s
                RETURNING
                    transaction_id,
                    customer_id,
                    amount,
                    refund_requested,
                    refund_amount,
                    refund_amount_ratio,
                    refund_delay_hours;
                """,
                (
                    new_refund_amount,
                    refund_amount_ratio,
                    refund_delay_hours,
                    transaction_id,
                ),
            )

            updated_row = cursor.fetchone()

        conn.commit()

        # ----------------------------------------------------
        # 7. Return normalized transaction data
        # ----------------------------------------------------

        return {
            "transaction_id": updated_row[0],
            "customer_id": updated_row[1],
            "amount": float(updated_row[2]),
            "refund_requested": updated_row[3],
            "refund_amount": float(updated_row[4]),
            "refund_amount_ratio": float(updated_row[5]),
            "refund_delay_hours": float(updated_row[6]),
            "refund_id": refund_id,
        }

    except Exception:
        if conn is not None:
            conn.rollback()
        raise

    finally:
        if conn is not None:
            conn.close()