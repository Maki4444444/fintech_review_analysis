"""
thematic.py
-----------
Thematic analysis pipeline for Ethiopian bank app reviews.
Extracts keywords per bank using TF-IDF and spaCy, then maps
reviews to business-relevant themes using a keyword-driven theme map
built from the extracted keywords.

Usage:
    python scripts/thematic.py

Input:
    data/clean/reviews_sentiment.csv

Output:
    data/clean/reviews_final.csv
"""

import os
import re
import pandas as pd
import spacy
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Load spaCy model
nlp = spacy.load("en_core_web_sm")
STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


# ── Theme maps per bank ────────────────────────────────────────────────────────
# These were derived by inspecting the top TF-IDF keywords and n-grams
# extracted from each bank's reviews and grouping semantically related terms
# into business-relevant categories.

CBE_THEME_MAP = {
    "Account Access Issues": [
        "login", "password", "otp", "sign", "access", "locked",
        "verification", "authenticate", "credential", "unlock"
    ],
    "Transaction Performance": [
        "transfer", "transaction", "send", "receive", "payment",
        "slow", "failed", "pending", "delay", "money"
    ],
    "App Stability": [
        "crash", "freeze", "bug", "error", "stuck", "force",
        "close", "stop", "working", "update", "fix", "problem",
        "issue", "version", "not working", "open"
    ],
    "UI & Design": [
        "ui", "interface", "design", "easy", "navigation", "layout",
        "button", "screen", "user", "friendly", "look", "simple",
        "fast", "speed", "quick", "smooth", "load"
    ],
    "Customer Support": [
        "support", "service", "help", "response", "agent",
        "complain", "call", "feedback", "contact", "resolve"
    ],
    "Positive Experience": [
    "good", "great", "excellent", "best", "nice", "love",
    "amazing", "awesome", "perfect", "wonderful", "happy",
    "satisfied", "thank", "well", "superb", "fantastic",
    "like", "useful", "helpful", "recommend", "impressive",
    "convenient", "reliable", "works", "working", "ok",
    "okay", "fine", "better", "improved", "smooth"
    ],
    "Negative Experience": [
        "worst", "bad", "terrible", "horrible", "useless", "disappoint",
        "poor", "hate", "awful", "waste", "pathetic", "disgusting"
    ],
}

BOA_THEME_MAP = {
    "Login & Authentication": [
        "login", "otp", "password", "sign", "access", "verification",
        "authenticate", "locked", "credential", "fingerprint"
    ],
    "Transaction Issues": [
        "transfer", "transaction", "payment", "send", "receive",
        "failed", "pending", "delay", "money", "slow", "balance"
    ],
    "App Stability & Bugs": [
        "crash", "freeze", "bug", "error", "stop", "working",
        "update", "fix", "close", "force", "issue", "problem",
        "not working", "version", "open", "phone"
    ],
    "UI & User Experience": [
        "ui", "interface", "design", "easy", "navigation", "layout",
        "simple", "friendly", "screen", "button", "look", "fast",
        "speed", "smooth", "quick", "load", "mobile"
    ],
    "Feature Requests": [
        "feature", "add", "need", "want", "request", "option",
        "improve", "upgrade", "new", "wish", "missing", "developer"
    ],
    "Positive Experience": [
    "good", "great", "excellent", "best", "nice", "love",
    "amazing", "awesome", "perfect", "wonderful", "happy",
    "satisfied", "thank", "well", "superb", "fantastic",
    "like", "useful", "helpful", "recommend", "impressive",
    "convenient", "reliable", "works", "working", "ok",
    "okay", "fine", "better", "improved", "smooth"
    ],
    "Negative Experience": [
        "worst", "bad", "terrible", "horrible", "useless", "disappoint",
        "poor", "hate", "awful", "waste", "pathetic"
    ],
}

DASHEN_THEME_MAP = {
    "Account & Login Problems": [
        "login", "password", "otp", "access", "sign", "locked",
        "verification", "account", "credential", "unlock"
    ],
    "Transfer & Payment Issues": [
        "transfer", "payment", "send", "receive", "money",
        "failed", "pending", "delay", "transaction", "slow", "balance"
    ],
    "App Performance": [
        "crash", "freeze", "bug", "slow", "error", "stuck",
        "working", "update", "fix", "force", "stop", "problem",
        "issue", "not working", "version", "open"
    ],
    "UI & Design": [
        "ui", "interface", "design", "easy", "navigation",
        "simple", "friendly", "screen", "button", "layout",
        "fast", "speed", "smooth", "quick", "load", "super"
    ],
    "Customer Service": [
        "support", "service", "help", "agent", "response",
        "complain", "call", "contact", "resolve", "feedback"
    ],
     "Positive Experience": [
    "good", "great", "excellent", "best", "nice", "love",
    "amazing", "awesome", "perfect", "wonderful", "happy",
    "satisfied", "thank", "well", "superb", "fantastic",
    "like", "useful", "helpful", "recommend", "impressive",
    "convenient", "reliable", "works", "working", "ok",
    "okay", "fine", "better", "improved", "smooth"
],
    "Negative Experience": [
        "worst", "bad", "terrible", "horrible", "useless", "disappoint",
        "poor", "hate", "awful", "waste", "pathetic"
    ],
}
BANK_THEME_MAPS = {
    "Commercial Bank of Ethiopia": CBE_THEME_MAP,
    "Bank of Abyssinia": BOA_THEME_MAP,
    "Dashen Bank": DASHEN_THEME_MAP,
}


# ── NLP preprocessing ─────────────────────────────────────────────────────────

def preprocess_text(text: str) -> str:
    """
    Clean and normalize review text for keyword extraction.

    Steps:
        1. Lowercase
        2. Remove non-alphabetic characters (keep spaces)
        3. Tokenize and lemmatize
        4. Remove stopwords and short tokens

    Args:
        text: Raw review string.

    Returns:
        Cleaned, lemmatized string.
    """
    text = re.sub(r"[^a-zA-Z\s]", " ", str(text).lower())
    tokens = text.split()
    tokens = [
        LEMMATIZER.lemmatize(t)
        for t in tokens
        if t not in STOP_WORDS and len(t) > 2
    ]
    return " ".join(tokens)


# ── TF-IDF keyword extraction ──────────────────────────────────────────────────

def extract_tfidf_keywords(texts: list, top_n: int = 30) -> list:
    """
    Extract top keywords from a list of texts using TF-IDF.

    Args:
        texts: List of preprocessed review strings.
        top_n: Number of top keywords to return.

    Returns:
        List of (keyword, score) tuples sorted by TF-IDF weight descending.
    """
    vectorizer = TfidfVectorizer(
        max_features=200,
        ngram_range=(1, 2),
        stop_words="english",
    )
    X = vectorizer.fit_transform(texts)
    scores = X.toarray().mean(axis=0)
    vocab = vectorizer.get_feature_names_out()

    keyword_scores = sorted(
        zip(vocab, scores), key=lambda x: x[1], reverse=True
    )
    return keyword_scores[:top_n]


# ── spaCy keyword extraction ───────────────────────────────────────────────────

def extract_spacy_keywords(texts: list, top_n: int = 30) -> list:
    """
    Extract top noun and noun-chunk keywords from texts using spaCy.

    Args:
        texts: List of raw review strings.
        top_n: Number of top keywords to return.

    Returns:
        List of (keyword, count) tuples sorted by frequency descending.
    """
    freq = defaultdict(int)
    combined = " ".join(texts[:500])  # limit for performance

    doc = nlp(combined[:100000])  # spaCy has character limits

    for token in doc:
        if (
            token.pos_ in ("NOUN", "PROPN")
            and token.text.lower() not in STOP_WORDS
            and len(token.text) > 2
        ):
            freq[token.lemma_.lower()] += 1

    return sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_n]


# ── Theme assignment ───────────────────────────────────────────────────────────

def assign_theme(review: str, theme_map: dict) -> str:
    """
    Assign a theme to a review based on keyword matching.

    Iterates through the theme map and returns the first theme whose
    keywords appear in the review. Returns 'Other' if no match is found.

    Args:
        review: Raw review string.
        theme_map: Dict mapping theme names to keyword lists.

    Returns:
        Theme label string.
    """
    review_lower = str(review).lower()
    for theme, keywords in theme_map.items():
        if any(kw in review_lower for kw in keywords):
            return theme
    return "Other"


def run_thematic_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run full thematic analysis pipeline per bank.

    For each bank:
        1. Preprocesses review text
        2. Extracts TF-IDF and spaCy keywords
        3. Assigns themes using the bank-specific theme map

    Args:
        df: DataFrame with 'review' and 'bank' columns.

    Returns:
        DataFrame with added 'clean_text' and 'identified_theme' columns.
    """
    df["clean_text"] = df["review"].apply(preprocess_text)
    df["identified_theme"] = "Other"

    for bank_name, theme_map in BANK_THEME_MAPS.items():
        bank_mask = df["bank"] == bank_name
        bank_df = df[bank_mask].copy()

        print(f"\n{'='*60}")
        print(f"Bank: {bank_name} ({len(bank_df)} reviews)")
        print(f"{'='*60}")

        # TF-IDF keywords
        tfidf_keywords = extract_tfidf_keywords(
            bank_df["clean_text"].tolist(), top_n=30
        )
        print(f"\n[TF-IDF] Top keywords:")
        for kw, score in tfidf_keywords[:15]:
            print(f"  {kw:<30} {score:.4f}")

        # spaCy keywords
        spacy_keywords = extract_spacy_keywords(
            bank_df["review"].tolist(), top_n=30
        )
        print(f"\n[spaCy] Top noun keywords:")
        for kw, count in spacy_keywords[:15]:
            print(f"  {kw:<30} {count}")

        # Assign themes
        df.loc[bank_mask, "identified_theme"] = bank_df["review"].apply(
            lambda x: assign_theme(x, theme_map)
        )

        # Theme distribution
        theme_dist = df[bank_mask]["identified_theme"].value_counts()
        print(f"\n[Themes] Distribution:")
        print(theme_dist)

    return df


if __name__ == "__main__":
    df = pd.read_csv("data/clean/reviews_sentiment.csv")

    df = run_thematic_analysis(df)

    # Select final output columns
    final_cols = [
        "review", "rating", "date", "bank", "source",
        "vader_score", "vader_label",
        "distilbert_label", "distilbert_score",
        "identified_theme",
    ]
    df_final = df[final_cols].reset_index(drop=True)
    df_final.insert(0, "review_id", df_final.index + 1)

    os.makedirs("data/clean", exist_ok=True)
    df_final.to_csv("data/clean/reviews_final.csv", index=False)
    print("\nSaved → data/clean/reviews_final.csv")