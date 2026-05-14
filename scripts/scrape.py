"""
scrape.py
---------
Scrapes Google Play Store reviews for Ethiopian bank apps and
saves the raw results to data/raw/reviews_raw.csv.

Usage:
    python scripts/scrape.py

Output:
    data/raw/reviews_raw.csv
"""

import os
import pandas as pd
from google_play_scraper import reviews, Sort
from tqdm import tqdm


APP_IDS = {
    "Commercial Bank of Ethiopia": "com.combanketh.mobilebanking",
    "Bank of Abyssinia": "com.boa.boaMobileBanking",
    "Dashen Bank": "com.dashen.dashensuperapp",
}


def scrape_reviews(app_ids: dict, count: int = 600) -> pd.DataFrame:
    """
    Scrape reviews from Google Play Store for multiple apps.

    Args:
        app_ids: Dict mapping bank name -> Play Store app ID.
        count: Number of reviews to request per app.

    Returns:
        Raw DataFrame with all scraped reviews.
    """
    all_reviews = []

    for bank_name, app_id in tqdm(app_ids.items(), desc="Scraping banks"):
        result, _ = reviews(
            app_id,
            lang="en",
            country="et",
            sort=Sort.NEWEST,
            count=count,
        )

        for entry in result:
            all_reviews.append({
                "id": entry.get("reviewId"),
                "review": entry.get("content"),
                "rating": entry.get("score"),
                "date": entry.get("at"),
                "bank": bank_name,
                "source": "Google Play",
            })

    return pd.DataFrame(all_reviews)


if __name__ == "__main__":
    raw_df = scrape_reviews(APP_IDS, count=600)

    os.makedirs("data/raw", exist_ok=True)
    raw_df.to_csv("data/raw/reviews_raw.csv", index=False)
    print("Saved → data/raw/reviews_raw.csv")