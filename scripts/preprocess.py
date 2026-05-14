"""
preprocess.py
-------------
Loads raw scraped reviews and produces a clean, analysis-ready CSV.

Usage:
    python scripts/preprocess.py

Input:
    data/raw/reviews_raw.csv

Output:
    data/clean/reviews_clean.csv
    data/clean/<bank_name>_clean.csv (one per bank)
"""

import os
import re
import pandas as pd
from tqdm import tqdm
from deep_translator import GoogleTranslator


def translate_amharic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect Amharic reviews using Ethiopic Unicode range and translate
    them to English using deep-translator's Google Translate backend.
    Emojis and non-Amharic text are preserved untouched.

    Args:
        df: DataFrame containing a 'review' column.

    Returns:
        DataFrame with Amharic reviews replaced by English translations.
    """
    translator = GoogleTranslator(source="am", target="en")
    translated_count = [0]

    def _contains_amharic(text):
        if not isinstance(text, str):
            return False
        return bool(re.search(r'[\u1200-\u137F]', text))

    def _translate_row(text):
        if not _contains_amharic(text):
            return text
        try:
            translated = translator.translate(text)
            translated_count[0] += 1
            return translated
        except Exception as e:
            print(f"[translate] Error on: {text[:30]}... → {e}")
            return text

    tqdm.pandas(desc="Translating Amharic reviews")
    df["review"] = df["review"].progress_apply(_translate_row)

    print(f"[translate] Amharic reviews translated: {translated_count[0]}")
    return df


def preprocess_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and normalize raw scraped reviews.

    Steps:
        1. Drop duplicate review IDs.
        2. Drop rows with missing review text or rating.
        3. Normalize date to YYYY-MM-DD string.
        4. Select and return only the five required columns.

    Args:
        df: Raw DataFrame from scrape_reviews().

    Returns:
        Cleaned DataFrame with columns: review, rating, date, bank, source.
    """
    original_count = len(df)

    # 1. Deduplicate by review ID
    df = df.drop_duplicates(subset="id")
    dupes_removed = original_count - len(df)

    # 2. Drop rows missing review text or rating
    df = df.dropna(subset=["review", "rating"])
    nulls_removed = original_count - dupes_removed - len(df)

    # 3. Normalize date to YYYY-MM-DD
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    # 4. Keep only required columns
    df = df[["review", "rating", "date", "bank", "source"]].reset_index(drop=True)

    print(f"[preprocess] Original rows     : {original_count}")
    print(f"[preprocess] Duplicates removed: {dupes_removed}")
    print(f"[preprocess] Nulls removed     : {nulls_removed}")
    print(f"[preprocess] Final row count   : {len(df)}")

    return df


if __name__ == "__main__":
    raw_df = pd.read_csv("data/raw/reviews_raw.csv")
    translated_df = translate_amharic(raw_df)
    clean_df = preprocess_reviews(translated_df)

    os.makedirs("data/clean", exist_ok=True)

    for bank_name, group in clean_df.groupby("bank"):
        filename = bank_name.lower().replace(" ", "_") + "_clean.csv"
        group.to_csv(f"data/clean/{filename}", index=False)
        print(f"Saved {len(group)} reviews → data/clean/{filename}")

    clean_df.to_csv("data/clean/reviews_clean.csv", index=False)
    print(f"\nSaved combined {len(clean_df)} reviews → data/clean/reviews_clean.csv")