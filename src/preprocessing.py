"""
Text Preprocessing Pipeline
============================
Provides text cleaning and normalization for mental health statements:
- Lowercasing
- URL / email / HTML removal
- Special character and markdown cleanup
- Optional lemmatization via NLTK WordNet
- Data splitting with stratification
"""

import re
import string

import nltk
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Ensure required NLTK data is available
for resource in ["wordnet", "omw-1.4", "averaged_perceptron_tagger",
                 "averaged_perceptron_tagger_eng", "punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"corpora/{resource}" if "wordnet" in resource or "omw" in resource
                       else f"taggers/{resource}" if "tagger" in resource
                       else f"tokenizers/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

from nltk.stem import WordNetLemmatizer

# Initialize lemmatizer once (reused across calls)
_lemmatizer = WordNetLemmatizer()


def preprocess_text(text: str, lemmatize: bool = True) -> str:
    """
    Full preprocessing pipeline for a single text string.

    Steps:
        1. Lowercase
        2. Remove URLs (http/https/www)
        3. Remove email addresses
        4. Remove HTML tags
        5. Remove Markdown formatting symbols
        6. Remove special characters (keep basic punctuation and alphanumerics)
        7. Collapse extra whitespace
        8. (Optional) Lemmatize words

    Args:
        text: Raw input text
        lemmatize: Whether to apply lemmatization (default True)

    Returns:
        Cleaned text string
    """
    if not isinstance(text, str) or len(text) == 0:
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # 3. Remove email addresses
    text = re.sub(r"\S+@\S+\.\S+", " ", text)

    # 4. Remove HTML tags (e.g., <br>, <p>, etc.)
    text = re.sub(r"<[^>]+>", " ", text)

    # 5. Remove Markdown formatting (headers, bullets, checkboxes, bold/italic)
    text = re.sub(r"#{1,6}\s*", "", text)          # headers
    text = re.sub(r"\*{1,3}|_{1,3}", "", text)     # bold/italic markers
    text = re.sub(r"- \[[ x]\]", "", text)          # checkboxes
    text = re.sub(r"^[\-\*\+]\s+", "", text, flags=re.MULTILINE)  # bullets

    # 6. Remove special characters — keep letters, digits, basic punctuation
    #    We preserve .,!?' as they carry sentiment information
    text = re.sub(r"[^a-z0-9\s.,!?']", " ", text)

    # 7. Collapse multiple spaces into one
    text = re.sub(r"\s+", " ", text).strip()

    # 8. Lemmatization (optional)
    if lemmatize and len(text) > 0:
        words = text.split()
        words = [_lemmatizer.lemmatize(w) for w in words]
        text = " ".join(words)

    return text


def preprocess_dataframe(df: pd.DataFrame, text_col: str = "statement",
                         lemmatize: bool = True) -> pd.DataFrame:
    """
    Apply preprocessing to an entire DataFrame's text column.

    Args:
        df: Input DataFrame
        text_col: Name of the column containing raw text
        lemmatize: Whether to apply lemmatization

    Returns:
        New DataFrame with an added 'clean_text' column
    """
    result = df.copy()
    result["clean_text"] = result[text_col].apply(
        lambda x: preprocess_text(x, lemmatize=lemmatize)
    )
    # Remove rows that became empty after preprocessing
    result = result[result["clean_text"].str.len() > 0].reset_index(drop=True)
    return result


def split_data(df: pd.DataFrame, test_size: float = 0.2,
               random_state: int = 42, label_col: str = "status"):
    """
    Perform stratified train/test split to maintain class proportions.

    Args:
        df: Cleaned DataFrame
        test_size: Fraction of data for testing (default 0.2)
        random_state: Random seed for reproducibility
        label_col: Name of the label column

    Returns:
        (train_df, test_df) — two DataFrames with the same columns
    """
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[label_col],
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print(f"[Split] Train: {len(train_df)}, Test: {len(test_df)}")
    print(f"[Split] Train label distribution:")
    for label, count in train_df[label_col].value_counts().items():
        print(f"  {label:25s} {count:6d}  ({count/len(train_df)*100:.1f}%)")

    return train_df, test_df


def encode_labels(y_series: pd.Series):
    """
    Encode string labels to integer indices and return the mapping.

    Args:
        y_series: Pandas Series of string labels

    Returns:
        (encoded_array, label_to_idx, idx_to_label)
        - encoded_array: numpy int array of indices
        - label_to_idx: dict mapping label string -> int
        - idx_to_label: dict mapping int -> label string
    """
    unique_labels = sorted(y_series.unique())
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    encoded = y_series.map(label_to_idx).values.astype(np.int64)
    return encoded, label_to_idx, idx_to_label
