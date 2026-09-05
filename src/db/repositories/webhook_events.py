from datetime import datetime, timezone
from typing import Any

from src.db.connection import get_connection


def create_webhook_event(
    event_id: str,
    event_type: str,
    transaction_id: str | None,
    payload: dict[str, Any],
) -> dict[str, Any] | None:
    """
    Store a webhook event.

    Returns the created event when this is the first delivery.
    Returns None when the event_id has already been received.
    """
    conn = None

    try:
        conn = get_connection()

        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO webhook_events (
                    event_id,
                    event_type,
                    transaction_id,
                    payload
                )
                VALUES (
                    %s,
                    %s,
                    %s,
                    %s::jsonb
                )
                ON CONFLICT (event_id) DO NOTHING
                RETURNING
                    webhook_event_id,
                    event_id,
                    event_type,
                    transaction_id,
                    payload,
                    status,
                    received_at,
                    processed_at,
                    error_message;
                """,
                (
                    event_id,
                    event_type,
                    transaction_id,
                    __import__("json").dumps(payload),
                ),
            )

            row = cursor.fetchone()

        conn.commit()

        if row is None:
            return None

        return {
            "webhook_event_id": row[0],
            "event_id": row[1],
            "event_type": row[2],
            "transaction_id": row[3],
            "payload": row[4],
            "status": row[5],
            "received_at": row[6],
            "processed_at": row[7],
            "error_message": row[8],
        }

    except Exception:
        if conn is not None:
            conn.rollback()
        raise

    finally:
        if conn is not None:
            conn.close()


def mark_webhook_processed(webhook_event_id: int) -> None:
    conn = None

    try:
        conn = get_connection()

        with conn.cursor() as cursor:
            cursor.execute(
                """
                UPDATE webhook_events
                SET
                    status = 'PROCESSED',
                    processed_at = %s,
                    error_message = NULL
                WHERE webhook_event_id = %s;
                """,
                (
                    datetime.now(timezone.utc),
                    webhook_event_id,
                ),
            )

        conn.commit()

    except Exception:
        if conn is not None:
            conn.rollback()
        raise

    finally:
        if conn is not None:
            conn.close()


def mark_webhook_failed(
    webhook_event_id: int,
    error_message: str,
) -> None:
    conn = None

    try:
        conn = get_connection()

        with conn.cursor() as cursor:
            cursor.execute(
                """
                UPDATE webhook_events
                SET
                    status = 'FAILED',
                    error_message = %s
                WHERE webhook_event_id = %s;
                """,
                (
                    error_message,
                    webhook_event_id,
                ),
            )

        conn.commit()

    except Exception:
        if conn is not None:
            conn.rollback()
        raise

    finally:
        if conn is not None:
            conn.close()
