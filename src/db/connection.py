import os

import psycopg2
from psycopg2.extensions import connection as PostgreSQLConnection


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "dbname=razorguard user=khushimishra host=/tmp port=5432",
)


def get_connection() -> PostgreSQLConnection:
    """
    Create and return a PostgreSQL database connection.
    """
    return psycopg2.connect(DATABASE_URL)
