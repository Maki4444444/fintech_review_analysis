"""
sentiment.py
------------
Sentiment analysis pipeline for Ethiopian bank app reviews.
Uses VADER for lexicon-based scoring and DistilBERT for transformer-based
scoring. Results are aggregated by bank and star rating.

Usage:
    python scripts/sentiment.py

Input:
    data/clean/reviews_clean.csv

Output:
    data/clean/reviews_sentiment.csv
"""

import os
import pandas as pd
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def get_vader_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply VADER sentiment analysis to the review column.

    Scores each review with a compound score and maps it to a label:
        compound >= 0.05  -> positive
        compound <= -0.05 -> negative
        else              -> neutral

    Args:
        df: DataFrame with a 'review' column.

    Returns:
        DataFrame with added columns: vader_score, vader_label.
    """
    sia = SentimentIntensityAnalyzer()

    tqdm.pandas(desc="Applying VADER sentiment")
    df["vader_score"] = df["review"].progress_apply(
        lambda x: sia.polarity_scores(str(x))["compound"]
    )

    def _label(score):
        if score >= 0.05:
            return "positive"
        elif score <= -0.05:
            return "negative"
        else:
            return "neutral"

    df["vader_label"] = df["vader_score"].apply(_label)

    print(f"[VADER] Distribution:\n{df['vader_label'].value_counts()}\n")
    return df


def get_distilbert_sentiment(df: pd.DataFrame, batch_size: int = 16) -> pd.DataFrame:
    """
    Apply DistilBERT sentiment analysis to the review column.

    Uses distilbert-base-uncased-finetuned-sst-2-english. Reviews are
    truncated to 512 tokens. Score is positive for POSITIVE label and
    negative for NEGATIVE label to allow directional comparison with VADER.

    Args:
        df: DataFrame with a 'review' column.
        batch_size: Number of reviews to process per batch.

    Returns:
        DataFrame with added columns: distilbert_label, distilbert_score.
    """
    try:
        from transformers import pipeline

        print("[DistilBERT] Loading model...")
        model = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            truncation=True,
            max_length=512,
        )

        results = []
        for i in tqdm(range(0, len(df), batch_size), desc="Applying DistilBERT sentiment"):
            batch = df["review"].iloc[i:i + batch_size].astype(str).tolist()
            preds = model(batch)
            results.extend(preds)

        df["distilbert_label"] = [p["label"].lower() for p in results]
        df["distilbert_score"] = [
            p["score"] if p["label"] == "POSITIVE" else -p["score"]
            for p in results
        ]

        print(f"[DistilBERT] Distribution:\n{df['distilbert_label'].value_counts()}\n")

    except Exception as e:
        print(f"[DistilBERT] Failed: {e}")
        df["distilbert_label"] = None
        df["distilbert_score"] = None

    return df


def aggregate_sentiment(df: pd.DataFrame) -> dict:
    """
    Aggregate sentiment scores by bank and by star rating.

    Args:
        df: DataFrame with vader_score, distilbert_score, bank, rating columns.

    Returns:
        Dict with two DataFrames: 'by_bank' and 'by_rating'.
    """
    by_bank = df.groupby("bank").agg(
        vader_mean=("vader_score", "mean"),
        distilbert_mean=("distilbert_score", "mean"),
        total_reviews=("review", "count"),
    ).round(4)

    by_rating = df.groupby(["bank", "rating"]).agg(
        vader_mean=("vader_score", "mean"),
        distilbert_mean=("distilbert_score", "mean"),
        review_count=("review", "count"),
    ).round(4)

    print("[Aggregation] Sentiment by bank:")
    print(by_bank)
    print("\n[Aggregation] Sentiment by bank and rating:")
    print(by_rating)

    return {"by_bank": by_bank, "by_rating": by_rating}


if __name__ == "__main__":
    df = pd.read_csv("data/clean/reviews_clean.csv")

    df = get_vader_sentiment(df)
    df = get_distilbert_sentiment(df)

    agg = aggregate_sentiment(df)

    os.makedirs("data/clean", exist_ok=True)
    df.to_csv("data/clean/reviews_sentiment.csv", index=False)
    print("Saved → data/clean/reviews_sentiment.csv")

    agg["by_bank"].to_csv("data/clean/sentiment_by_bank.csv")
    agg["by_rating"].to_csv("data/clean/sentiment_by_rating.csv")
    print("Saved → data/clean/sentiment_by_bank.csv")
    print("Saved → data/clean/sentiment_by_rating.csv")