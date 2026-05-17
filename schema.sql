-- schema.sql
-- Database schema for Fintech Review Analytics
-- Database: bank_reviews

-- ── Drop tables if they exist (for clean re-runs) ─────────────────────────
DROP TABLE IF EXISTS reviews;
DROP TABLE IF EXISTS banks;

-- ── Banks table ────────────────────────────────────────────────────────────
-- Stores metadata about each Ethiopian bank app
CREATE TABLE banks (
    bank_id     SERIAL PRIMARY KEY,
    bank_name   VARCHAR(255) UNIQUE NOT NULL,
    app_id      VARCHAR(255) NOT NULL
);

-- ── Reviews table ──────────────────────────────────────────────────────────
-- Stores all scraped, preprocessed, and analyzed reviews
-- bank_id is a FOREIGN KEY linking each review to its bank
CREATE TABLE reviews (
    review_id         SERIAL PRIMARY KEY,
    bank_id           INT REFERENCES banks(bank_id) ON DELETE CASCADE,
    review_text       TEXT,
    rating            INT CHECK (rating BETWEEN 1 AND 5),
    review_date       DATE,
    sentiment_label   VARCHAR(50),
    sentiment_score   FLOAT,
    identified_theme  VARCHAR(100),
    source            VARCHAR(50)
);

-- ── Indexes for query performance ──────────────────────────────────────────
CREATE INDEX idx_reviews_bank_id ON reviews(bank_id);
CREATE INDEX idx_reviews_sentiment ON reviews(sentiment_label);
CREATE INDEX idx_reviews_theme ON reviews(identified_theme);
CREATE INDEX idx_reviews_date ON reviews(review_date);