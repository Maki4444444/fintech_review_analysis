"""
db_insert.py
------------
Connects to the bank_reviews PostgreSQL database, creates the banks
and reviews tables, and inserts all 1,800 cleaned and analyzed reviews
from reviews_final.csv.

Usage:
    python scripts/db_insert.py

Input:
    data/clean/reviews_final.csv

Output:
    Populated banks and reviews tables in the bank_reviews database.
"""

import os
import pandas as pd
import psycopg2
from tqdm import tqdm


# ── Database connection config ─────────────────────────────────────────────
DB_CONFIG = {
    "host":     "localhost",
    "database": "bank_reviews",
    "user":     "bank_admin",
    "password": "admin123",
}

# ── Bank metadata ──────────────────────────────────────────────────────────
BANKS = [
    ("Commercial Bank of Ethiopia", "com.combanketh.mobilebanking"),
    ("Bank of Abyssinia",           "com.boa.boaMobileBanking"),
    ("Dashen Bank",                 "com.dashen.dashensuperapp"),
]


def get_connection():
    """Create and return a psycopg2 database connection."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        print("[DB] Connected to bank_reviews successfully.")
        return conn
    except Exception as e:
        print(f"[DB] Connection failed: {e}")
        raise


def create_tables(conn):
    """
    Create banks and reviews tables if they don't already exist.

    Args:
        conn: Active psycopg2 connection.
    """
    cur = conn.cursor()

    # Banks table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS banks (
            bank_id   SERIAL PRIMARY KEY,
            bank_name VARCHAR(255) UNIQUE NOT NULL,
            app_id    VARCHAR(255) NOT NULL
        );
    """)

    # Reviews table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS reviews (
            review_id        SERIAL PRIMARY KEY,
            bank_id          INT REFERENCES banks(bank_id) ON DELETE CASCADE,
            review_text      TEXT,
            rating           INT CHECK (rating BETWEEN 1 AND 5),
            review_date      DATE,
            sentiment_label  VARCHAR(50),
            sentiment_score  FLOAT,
            identified_theme VARCHAR(100),
            source           VARCHAR(50)
        );
    """)

    # Indexes
    cur.execute("CREATE INDEX IF NOT EXISTS idx_reviews_bank_id ON reviews(bank_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_reviews_sentiment ON reviews(sentiment_label);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_reviews_theme ON reviews(identified_theme);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_reviews_date ON reviews(review_date);")

    conn.commit()
    cur.close()
    print("[DB] Tables created successfully.")


def insert_banks(conn):
    """
    Insert bank metadata into the banks table.

    Args:
        conn: Active psycopg2 connection.

    Returns:
        Dict mapping bank_name -> bank_id.
    """
    cur = conn.cursor()
    bank_id_map = {}

    for bank_name, app_id in BANKS:
        cur.execute("""
            INSERT INTO banks (bank_name, app_id)
            VALUES (%s, %s)
            ON CONFLICT (bank_name) DO NOTHING;
        """, (bank_name, app_id))

        cur.execute("SELECT bank_id FROM banks WHERE bank_name = %s;", (bank_name,))
        bank_id_map[bank_name] = cur.fetchone()[0]

    conn.commit()
    cur.close()
    print(f"[DB] Inserted {len(BANKS)} banks.")
    return bank_id_map


def insert_reviews(conn, df: pd.DataFrame, bank_id_map: dict):
    """
    Insert all reviews into the reviews table.

    Uses ON CONFLICT DO NOTHING to make the script safe to re-run
    without creating duplicate entries.

    Args:
        conn: Active psycopg2 connection.
        df: DataFrame from reviews_final.csv.
        bank_id_map: Dict mapping bank_name -> bank_id.
    """
    cur = conn.cursor()
    inserted = 0
    skipped = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Inserting reviews"):
        bank_id = bank_id_map.get(row["bank"])
        if bank_id is None:
            skipped += 1
            continue

        try:
            cur.execute("""
                INSERT INTO reviews (
                    review_id, bank_id, review_text, rating, review_date,
                    sentiment_label, sentiment_score, identified_theme, source
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (review_id) DO NOTHING;
            """, (
                int(row["review_id"]),
                bank_id,
                str(row["review"]),
                int(row["rating"]),
                str(row["date"]),
                str(row["distilbert_label"]),
                float(row["distilbert_score"]),
                str(row["identified_theme"]),
                str(row["source"]),
            ))
            inserted += 1
        except Exception as e:
            print(f"[WARN] Skipped row {row['review_id']}: {e}")
            skipped += 1

    conn.commit()
    cur.close()
    print(f"[DB] Inserted: {inserted} | Skipped: {skipped}")


if __name__ == "__main__":
    # Load data
    filepath = "data/clean/reviews_final.csv"
    if not os.path.exists(filepath):
        print(f"[ERROR] File not found: {filepath}")
        exit(1)

    df = pd.read_csv(filepath)
    print(f"[DATA] Loaded {len(df)} reviews from {filepath}")

    # Run pipeline
    conn = get_connection()
    create_tables(conn)
    bank_id_map = insert_banks(conn)
    insert_reviews(conn, df, bank_id_map)

    conn.close()
    print("[DB] Connection closed. All done.")