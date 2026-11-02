import os

import psycopg
from psycopg.rows import dict_row
from dotenv import load_dotenv

load_dotenv()


def get_db_connection():
    """Connect to the PostgreSQL database hosted on Neon."""
    database_url = os.getenv("NEON_DATABASE_URL")

    if not database_url:
        raise RuntimeError(
            "NEON_DATABASE_URL is not configured."
        )

    return psycopg.connect(database_url)


def initialize_database():
    """Create the uploaded_documents table if it doesn't exist."""
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS uploaded_documents (
                    document_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    file_data BYTEA NOT NULL,
                    uploaded_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )


def save_document(user_id, filename, file_type, file_data):
    """Save an uploaded document and its metadata in PostgreSQL."""
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO uploaded_documents
                    (user_id, filename, file_type, file_data)
                VALUES (%s, %s, %s, %s)
                RETURNING document_id
                """,
                (user_id, filename, file_type, file_data),
            )

            return cursor.fetchone()[0]


def get_user_documents(user_id):
    """Retrieve metadata for a user's uploaded documents."""
    with get_db_connection() as connection:
        with connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """
                SELECT document_id, filename, file_type, uploaded_at
                FROM uploaded_documents
                WHERE user_id = %s
                ORDER BY uploaded_at DESC
                """,
                (user_id,),
            )

            return cursor.fetchall()


def get_document_data(user_id, document_id):
    """Retrieve document binary data, checking its owner."""
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT file_data
                FROM uploaded_documents
                WHERE user_id = %s AND document_id = %s
                """,
                (user_id, document_id),
            )

            result = cursor.fetchone()
            return result[0] if result else None