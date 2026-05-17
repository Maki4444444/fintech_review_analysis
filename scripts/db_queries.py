"""
db_queries.py
-------------
Runs verification SQL queries against the bank_reviews database
to confirm data integrity after insertion.

Usage:
    python scripts/db_queries.py
"""

import psycopg2
import pandas as pd


DB_CONFIG = {
    "host":     "localhost",
    "database": "bank_reviews",
    "user":     "bank_admin",
    "password": "admin123",
}


def run_query(conn, title: str, query: str) -> pd.DataFrame:
    """
    Execute a SQL query and return results as a DataFrame.

    Args:
        conn: Active psycopg2 connection.
        title: Label printed before results.
        query: SQL query string.

    Returns:
        DataFrame with query results.
    """
    cur = conn.cursor()
    cur.execute(query)
    rows = cur.fetchall()
    cols = [desc[0] for desc in cur.description]
    cur.close()
    df = pd.DataFrame(rows, columns=cols)
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    conn = psycopg2.connect(**DB_CONFIG)
    print("[DB] Connected to bank_reviews successfully.")

    # 1. Total reviews per bank
    run_query(conn, "1. Total Reviews per Bank", """
        SELECT
            b.bank_name,
            COUNT(r.review_id) AS total_reviews
        FROM banks b
        LEFT JOIN reviews r ON b.bank_id = r.bank_id
        GROUP BY b.bank_name
        ORDER BY total_reviews DESC;
    """)

    # 2. Average rating per bank
    run_query(conn, "2. Average Rating per Bank", """
        SELECT
            b.bank_name,
            ROUND(AVG(r.rating)::NUMERIC, 2) AS avg_rating,
            MIN(r.rating) AS min_rating,
            MAX(r.rating) AS max_rating
        FROM banks b
        LEFT JOIN reviews r ON b.bank_id = r.bank_id
        GROUP BY b.bank_name
        ORDER BY avg_rating DESC;
    """)

    # 3. Sentiment distribution per bank
    run_query(conn, "3. Sentiment Distribution per Bank", """
        SELECT
            b.bank_name,
            r.sentiment_label,
            COUNT(*) AS count,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY b.bank_name), 1) AS percentage
        FROM banks b
        JOIN reviews r ON b.bank_id = r.bank_id
        GROUP BY b.bank_name, r.sentiment_label
        ORDER BY b.bank_name, count DESC;
    """)

    # 4. Theme distribution per bank
    run_query(conn, "4. Theme Distribution per Bank", """
        SELECT
            b.bank_name,
            r.identified_theme,
            COUNT(*) AS count
        FROM banks b
        JOIN reviews r ON b.bank_id = r.bank_id
        GROUP BY b.bank_name, r.identified_theme
        ORDER BY b.bank_name, count DESC;
    """)

    # 5. Null check on key columns
    run_query(conn, "5. Null Check on Key Columns", """
        SELECT
            COUNT(*) FILTER (WHERE review_text IS NULL)    AS null_review_text,
            COUNT(*) FILTER (WHERE rating IS NULL)         AS null_rating,
            COUNT(*) FILTER (WHERE sentiment_label IS NULL) AS null_sentiment_label,
            COUNT(*) FILTER (WHERE sentiment_score IS NULL) AS null_sentiment_score,
            COUNT(*) FILTER (WHERE identified_theme IS NULL) AS null_theme,
            COUNT(*) FILTER (WHERE review_date IS NULL)    AS null_review_date
        FROM reviews;
    """)

    # 6. Average sentiment score per bank
    run_query(conn, "6. Average Sentiment Score per Bank", """
        SELECT
            b.bank_name,
            ROUND(AVG(r.sentiment_score)::NUMERIC, 4) AS avg_sentiment_score
        FROM banks b
        JOIN reviews r ON b.bank_id = r.bank_id
        GROUP BY b.bank_name
        ORDER BY avg_sentiment_score DESC;
    """)

    # 7. Reviews per bank per rating
    run_query(conn, "7. Reviews per Bank per Star Rating", """
        SELECT
            b.bank_name,
            r.rating,
            COUNT(*) AS count
        FROM banks b
        JOIN reviews r ON b.bank_id = r.bank_id
        GROUP BY b.bank_name, r.rating
        ORDER BY b.bank_name, r.rating;
    """)

    conn.close()
    print("\n[DB] All verification queries complete.")