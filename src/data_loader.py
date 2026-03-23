"""
Data Loading and Cleaning Module
=================================
Handles loading the raw CSV dataset and performing cleaning operations:
- Remove invalid rows (empty, non-text, invalid labels)
- Decode HTML entities
- Normalize label variations
- Deduplicate entries
"""

import html
import re
import pandas as pd

# The 7 valid mental health status categories
VALID_LABELS = [
    "Normal",
    "Depression",
    "Suicidal",
    "Anxiety",
    "Bipolar",
    "Stress",
    "Personality disorder",
]


def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Load the raw CSV file and keep only the relevant columns.

    Args:
        filepath: Path to 'Combined Data.csv'

    Returns:
        DataFrame with columns ['statement', 'status']
    """
    df = pd.read_csv(filepath)
    # Keep only the two columns we need
    df = df[["statement", "status"]].copy()
    return df


def normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize label variations to a canonical set of 7 classes.
    Handles common typos and casing differences found in the dataset:
      - 'Bi-Polar' / 'bi-polar' / 'bipolar' -> 'Bipolar'
      - 'personality disorder' / 'Personality Disorder' -> 'Personality disorder'
      - leading/trailing whitespace in labels

    Args:
        df: DataFrame with a 'status' column

    Returns:
        DataFrame with normalized 'status' values
    """
    result = df.copy()
    # Strip whitespace from labels
    result["status"] = result["status"].astype(str).str.strip()

    # Build a case-insensitive mapping to canonical labels
    label_map = {}
    for label in VALID_LABELS:
        label_map[label.lower()] = label

    # Add known variants
    label_map["bi-polar"] = "Bipolar"
    label_map["personality_disorder"] = "Personality disorder"
    label_map["personalitydisorder"] = "Personality disorder"

    # Apply mapping (case-insensitive)
    result["status"] = result["status"].apply(
        lambda x: label_map.get(x.strip().lower(), x)
    )
    return result


def clean_html_entities(text: str) -> str:
    """
    Decode HTML entities such as &#x200B;, &nbsp;, &amp; etc.

    Args:
        text: Raw text string

    Returns:
        Text with HTML entities decoded
    """
    if not isinstance(text, str):
        return ""
    # Decode HTML entities (e.g., &amp; -> &, &#x200B; -> zero-width space)
    text = html.unescape(text)
    # Remove zero-width characters that remain after decoding
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    return text


def clean_dataset(df: pd.DataFrame, min_text_length: int = 5) -> pd.DataFrame:
    """
    Perform full cleaning pipeline on the dataset:
    1. Normalize labels to canonical 7-class set
    2. Filter rows to only valid labels
    3. Decode HTML entities in statement text
    4. Remove empty / too-short statements
    5. Remove exact duplicates

    Args:
        df: Raw DataFrame with ['statement', 'status'] columns
        min_text_length: Minimum character length for valid statements

    Returns:
        Cleaned DataFrame ready for preprocessing
    """
    print(f"[Clean] Starting with {len(df)} rows")

    # Step 1: Normalize label variations
    df = normalize_labels(df)

    # Step 2: Keep only rows with valid labels
    df = df[df["status"].isin(VALID_LABELS)].copy()
    print(f"[Clean] After label filtering: {len(df)} rows")

    # Step 3: Clean HTML entities in the statement column
    df["statement"] = df["statement"].apply(clean_html_entities)

    # Step 4: Remove rows where statement is empty or too short
    df["statement"] = df["statement"].str.strip()
    df = df[df["statement"].str.len() >= min_text_length].copy()
    print(f"[Clean] After removing short/empty statements: {len(df)} rows")

    # Step 5: Remove exact duplicate (statement, status) pairs
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["statement", "status"]).copy()
    print(f"[Clean] Removed {before_dedup - len(df)} duplicates, final: {len(df)} rows")

    # Reset index for clean output
    df = df.reset_index(drop=True)
    return df


def load_and_clean(filepath: str) -> pd.DataFrame:
    """
    Convenience function: load raw CSV and return cleaned DataFrame.

    Args:
        filepath: Path to 'Combined Data.csv'

    Returns:
        Cleaned DataFrame with ['statement', 'status'] columns
    """
    raw_df = load_raw_data(filepath)
    clean_df = clean_dataset(raw_df)

    # Print class distribution summary
    print("\n[Clean] Class distribution:")
    dist = clean_df["status"].value_counts()
    for label, count in dist.items():
        print(f"  {label:25s} {count:6d}  ({count/len(clean_df)*100:.1f}%)")

    return clean_df


if __name__ == "__main__":
    import os

    # Default path relative to project root
    data_path = os.path.join(os.path.dirname(__file__), "..", "Combined Data.csv")
    df = load_and_clean(data_path)
    print(f"\nFinal dataset shape: {df.shape}")
