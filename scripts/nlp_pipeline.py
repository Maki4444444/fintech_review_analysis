"""
nlp_pipeline.py
---------------
Modular NLP preprocessing pipeline for Ethiopian bank app reviews.
Handles tokenization, stop-word removal, and lemmatization.

Used by thematic.py for keyword extraction and theme assignment.

Usage:
    from scripts.nlp_pipeline import tokenize, remove_stopwords, lemmatize, preprocess
"""

import re
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("punkt_tab", quiet=True)

# Load spaCy model and NLTK tools
nlp = spacy.load("en_core_web_sm")
STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def tokenize(text: str) -> list:
    """
    Tokenize a review string into a list of lowercase word tokens.
    Removes non-alphabetic characters before tokenizing.

    Args:
        text: Raw review string.

    Returns:
        List of lowercase word tokens.
    """
    text = re.sub(r"[^a-zA-Z\s]", " ", str(text).lower())
    tokens = word_tokenize(text)
    return tokens


def remove_stopwords(tokens: list) -> list:
    """
    Remove English stopwords and short tokens from a token list.

    Args:
        tokens: List of word tokens.

    Returns:
        Filtered list of tokens with stopwords and tokens
        shorter than 3 characters removed.
    """
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 2]


def lemmatize(tokens: list) -> list:
    """
    Lemmatize a list of tokens using NLTK WordNetLemmatizer.

    Args:
        tokens: List of word tokens.

    Returns:
        List of lemmatized tokens.
    """
    return [LEMMATIZER.lemmatize(t) for t in tokens]


def preprocess(text: str) -> str:
    """
    Full NLP preprocessing pipeline: tokenize → remove stopwords → lemmatize.

    Args:
        text: Raw review string.

    Returns:
        Cleaned, lemmatized string ready for keyword extraction.
    """
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)
    return " ".join(tokens)


def spacy_noun_extract(texts: list, top_n: int = 30) -> list:
    """
    Extract top nouns and proper nouns from a list of texts using spaCy.

    Args:
        texts: List of raw review strings.
        top_n: Number of top keywords to return.

    Returns:
        List of (keyword, count) tuples sorted by frequency descending.
    """
    from collections import defaultdict
    freq = defaultdict(int)
    combined = " ".join(texts[:500])

    doc = nlp(combined[:100000])
    for token in doc:
        if (
            token.pos_ in ("NOUN", "PROPN")
            and token.text.lower() not in STOP_WORDS
            and len(token.text) > 2
        ):
            freq[token.lemma_.lower()] += 1

    return sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_n]