"""
Feature Engineering Module
===========================
Builds feature matrices for traditional ML models:
- TF-IDF vector features (unigram + bigram)
- Handcrafted statistical features (linguistic cues)
- Combined sparse + dense feature matrix
"""

import re
import string

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

# First-person pronouns for mental health text analysis
FIRST_PERSON_PRONOUNS = {"i", "me", "my", "myself", "mine", "we", "us", "our", "ourselves"}

# Negative emotion words (curated for mental health context)
NEGATIVE_WORDS = {
    "sad", "depressed", "hopeless", "worthless", "alone", "lonely", "anxious",
    "afraid", "scared", "worried", "angry", "hate", "die", "death", "kill",
    "suicide", "hurt", "pain", "suffering", "cry", "crying", "tired", "exhausted",
    "empty", "numb", "broken", "lost", "failed", "failure", "useless", "helpless",
    "miserable", "terrible", "awful", "worse", "worst", "never", "nothing",
    "nobody", "guilt", "guilty", "shame", "panic", "stress", "stressed",
}


def build_tfidf_vectorizer(max_features=10000, ngram_range=(1, 2),
                           min_df=3, max_df=0.95):
    """Create a configured TF-IDF vectorizer."""
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True,
    )


def extract_handcrafted_features(texts: pd.Series, raw_texts: pd.Series = None) -> pd.DataFrame:
    """
    Extract linguistic and statistical features from text.

    Features:
        - char_count: character count
        - word_count: word count
        - avg_word_length: average word length
        - sentence_count: number of sentences
        - exclamation_count: number of '!'
        - question_count: number of '?'
        - uppercase_ratio: ratio of uppercase characters
        - unique_word_ratio: unique words / total words (lexical diversity)
        - first_person_pronoun_count: count of first-person pronouns
        - negative_word_count: count of negative emotion words
    """
    features = pd.DataFrame()

    features["char_count"] = texts.str.len()
    features["word_count"] = texts.str.split().str.len()
    features["avg_word_length"] = texts.apply(
        lambda t: np.mean([len(w) for w in t.split()]) if len(t.split()) > 0 else 0
    )
    features["sentence_count"] = texts.apply(
        lambda t: max(len(re.split(r"[.!?]+", t)) - 1, 1)
    )
    features["exclamation_count"] = texts.str.count("!")
    features["question_count"] = texts.str.count(r"\?")
    upper_source = raw_texts if raw_texts is not None else texts
    features["uppercase_ratio"] = upper_source.apply(
        lambda t: sum(1 for c in str(t) if c.isupper()) / max(len(str(t)), 1)
    )
    features["unique_word_ratio"] = texts.apply(
        lambda t: len(set(t.lower().split())) / max(len(t.split()), 1)
    )
    features["first_person_pronoun_count"] = texts.apply(
        lambda t: sum(1 for w in t.lower().split() if w in FIRST_PERSON_PRONOUNS)
    )
    features["negative_word_count"] = texts.apply(
        lambda t: sum(1 for w in t.lower().split() if w in NEGATIVE_WORDS)
    )

    return features


def build_feature_matrix(train_texts: pd.Series, test_texts: pd.Series,
                         train_raw: pd.Series = None, test_raw: pd.Series = None,
                         tfidf_params=None):
    """
    Build combined TF-IDF + handcrafted feature matrices for train and test.

    Args:
        train_texts: Series of cleaned training texts
        test_texts: Series of cleaned test texts
        tfidf_params: Optional dict of TfidfVectorizer parameters

    Returns:
        (X_train, X_test, tfidf_vectorizer)
        - X_train: sparse matrix (n_train, tfidf_features + handcrafted_features)
        - X_test: sparse matrix (n_test, tfidf_features + handcrafted_features)
        - tfidf_vectorizer: fitted TfidfVectorizer instance
    """
    params = tfidf_params or {}
    tfidf = build_tfidf_vectorizer(**params)

    # TF-IDF features
    X_train_tfidf = tfidf.fit_transform(train_texts)
    X_test_tfidf = tfidf.transform(test_texts)

    print(f"[Features] TF-IDF shape: train={X_train_tfidf.shape}, test={X_test_tfidf.shape}")

    # Handcrafted features
    train_hf = extract_handcrafted_features(train_texts, raw_texts=train_raw)
    test_hf = extract_handcrafted_features(test_texts, raw_texts=test_raw)

    print(f"[Features] Handcrafted features: {train_hf.shape[1]} columns")

    # Combine: sparse TF-IDF + dense handcrafted -> sparse matrix
    X_train = hstack([X_train_tfidf, csr_matrix(train_hf.values)], format="csr")
    X_test = hstack([X_test_tfidf, csr_matrix(test_hf.values)], format="csr")

    print(f"[Features] Combined shape: train={X_train.shape}, test={X_test.shape}")

    return X_train, X_test, tfidf
