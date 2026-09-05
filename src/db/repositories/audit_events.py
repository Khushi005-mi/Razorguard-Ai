import json

from src.db.connection import get_connection


def create_audit_event(
    event_type: str,
    transaction_id: str | None,
    event_data: dict,
) -> dict:
    query = """
        INSERT INTO audit_events (
            transaction_id,
            event_type,
            event_data
        )
        VALUES (%s, %s, %s::jsonb)
        RETURNING
            audit_event_id,
            transaction_id,
            event_type,
            event_data,
            event_timestamp;
    """

    conn = None

    try:
        conn = get_connection()

        with conn.cursor() as cursor:
            cursor.execute(
                query,
                (
                    transaction_id,
                    event_type,
                    json.dumps(event_data),
                ),
            )

            row = cursor.fetchone()

        conn.commit()

        columns = [
            "audit_event_id",
            "transaction_id",
            "event_type",
            "event_data",
            "event_timestamp",
        ]

        return dict(zip(columns, row))

    except Exception:
        if conn is not None:
            conn.rollback()
        raise

    finally:
        if conn is not None:
            conn.close()