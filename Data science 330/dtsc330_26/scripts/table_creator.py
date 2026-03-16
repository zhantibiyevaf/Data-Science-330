import sqlite3
from pathlib import Path


def minimal_create_database():
    """Create the SQLite database."""

    db_path = Path(__file__).resolve().parents[2] / "data" / "article_grant_db.sqlite"

    connection = sqlite3.connect(db_path)

    print("SQLite version:", connection.execute("SELECT sqlite_version();").fetchall())

    connection.close()


minimal_create_database()