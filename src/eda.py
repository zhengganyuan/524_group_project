"""
Exploratory Data Analysis (EDA)
================================
Generates 6 key EDA plots and saves them to figures/:
1. Class distribution bar chart
2. Text length distribution (box + histogram)
3. Word frequency analysis (top 20 per class)
4. Class similarity heatmap (cosine similarity of TF-IDF centroids)
5. N-gram analysis (top bigrams/trigrams per class)
6. Cleaning statistics summary
"""

import os
import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.data_loader import load_raw_data, clean_dataset, VALID_LABELS

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "Combined Data.csv")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")

# Common English stop words for word frequency analysis
STOP_WORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself",
    "she", "her", "hers", "herself", "it", "its", "itself", "they", "them",
    "their", "theirs", "themselves", "what", "which", "who", "whom", "this",
    "that", "these", "those", "am", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "having", "do", "does", "did", "doing",
    "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
    "while", "of", "at", "by", "for", "with", "about", "against", "between",
    "through", "during", "before", "after", "above", "below", "to", "from",
    "up", "down", "in", "out", "on", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why",
    "how", "all", "both", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
    "very", "s", "t", "can", "will", "just", "don", "should", "now", "d",
    "ll", "m", "o", "re", "ve", "y", "ain", "aren", "couldn", "didn",
    "doesn", "hadn", "hasn", "haven", "isn", "ma", "mightn", "mustn",
    "needn", "shan", "shouldn", "wasn", "weren", "won", "wouldn",
    "also", "would", "could", "one", "like", "get", "go", "know",
    "really", "even", "much", "still", "back", "thing", "things",
    "im", "dont", "ive", "cant", "thats", "got", "going", "want",
}


def ensure_figures_dir():
    os.makedirs(FIGURES_DIR, exist_ok=True)


# ── Plot 1: Class Distribution ──────────────────────────────────────────────

def plot_class_distribution(df):
    """Bar chart showing sample count per class with percentage labels."""
    ensure_figures_dir()
    counts = df["status"].value_counts()
    total = len(df)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("Set2", n_colors=len(counts))
    bars = ax.bar(range(len(counts)), counts.values, color=colors)

    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index, rotation=25, ha="right")
    ax.set_ylabel("Number of Samples")
    ax.set_title("Class Distribution of Mental Health Status Labels")

    for bar, count in zip(bars, counts.values):
        pct = count / total * 100
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
                f"{count:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    fname = "01_class_distribution.png"
    fig.savefig(os.path.join(FIGURES_DIR, fname), dpi=150)
    plt.close(fig)
    print(f"[EDA] Saved: figures/{fname}")


# ── Plot 2: Text Length Distribution ─────────────────────────────────────────

def plot_text_length_distribution(df):
    """Box plot + histogram of word count per class."""
    ensure_figures_dir()
    df = df.copy()
    df["word_count"] = df["statement"].str.split().str.len()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Box plot
    order = VALID_LABELS
    sns.boxplot(data=df, x="status", y="word_count", order=order, ax=ax1,
                palette="Set2", showfliers=False)
    ax1.set_xticklabels(order, rotation=25, ha="right")
    ax1.set_ylabel("Word Count")
    ax1.set_title("Word Count Distribution by Class (Box Plot)")

    # Histogram overlay
    for label in order:
        subset = df[df["status"] == label]["word_count"]
        ax2.hist(subset, bins=50, alpha=0.5, label=label, range=(0, 300))
    ax2.set_xlabel("Word Count")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Word Count Distribution by Class (Histogram)")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    fname = "02_text_length_distribution.png"
    fig.savefig(os.path.join(FIGURES_DIR, fname), dpi=150)
    plt.close(fig)
    print(f"[EDA] Saved: figures/{fname}")


# ── Plot 3: Word Frequency Analysis ─────────────────────────────────────────

def plot_word_frequency(df, top_n=20):
    """Top-N most frequent words per class (after stop word removal)."""
    ensure_figures_dir()
    labels = VALID_LABELS
    n_classes = len(labels)
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    axes = axes.flatten()

    for i, label in enumerate(labels):
        texts = df[df["status"] == label]["statement"]
        counter = Counter()
        for text in texts:
            words = re.findall(r"[a-z]+", text.lower())
            counter.update(w for w in words if w not in STOP_WORDS and len(w) > 2)

        top_words = counter.most_common(top_n)
        if not top_words:
            continue
        words, counts = zip(*top_words)

        axes[i].barh(range(len(words)), counts, color=sns.color_palette("Set2")[i])
        axes[i].set_yticks(range(len(words)))
        axes[i].set_yticklabels(words, fontsize=8)
        axes[i].invert_yaxis()
        axes[i].set_title(label, fontsize=11)
        axes[i].set_xlabel("Frequency")

    # Hide the extra subplot
    if n_classes < len(axes):
        for j in range(n_classes, len(axes)):
            axes[j].set_visible(False)

    fig.suptitle(f"Top {top_n} Words per Class (Stop Words Removed)", fontsize=14)
    plt.tight_layout()
    fname = "03_word_frequency.png"
    fig.savefig(os.path.join(FIGURES_DIR, fname), dpi=150)
    plt.close(fig)
    print(f"[EDA] Saved: figures/{fname}")


# ── Plot 4: Class Similarity Heatmap ────────────────────────────────────────

def plot_class_similarity_heatmap(df):
    """Cosine similarity between TF-IDF class centroids."""
    ensure_figures_dir()
    tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
    X = tfidf.fit_transform(df["statement"])
    labels = df["status"].values

    centroids = []
    label_order = VALID_LABELS
    for label in label_order:
        mask = labels == label
        centroid = X[mask].mean(axis=0)
        centroids.append(np.asarray(centroid).flatten())

    centroid_matrix = np.vstack(centroids)
    sim_matrix = cosine_similarity(centroid_matrix)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(sim_matrix, annot=True, fmt=".2f", cmap="YlOrRd",
                xticklabels=label_order, yticklabels=label_order, ax=ax)
    ax.set_title("Cosine Similarity Between Class TF-IDF Centroids")
    plt.tight_layout()
    fname = "04_class_similarity_heatmap.png"
    fig.savefig(os.path.join(FIGURES_DIR, fname), dpi=150)
    plt.close(fig)
    print(f"[EDA] Saved: figures/{fname}")


# ── Plot 5: N-gram Analysis ─────────────────────────────────────────────────

def _is_url_fragment(ngram):
    """Check if an n-gram contains URL fragments or non-text noise."""
    url_indicators = {
        "http", "https", "www", "com", "org", "net", "edu", "gov",
        "reddit", "youtube", "twitpic", "tinyurl", "plurk", "imgur",
        "qualtrics", "forms", "gle", "bit", "ly", "poll",
    }
    tokens = ngram.split()
    # If any token is a URL domain/protocol keyword, skip it
    if any(t in url_indicators for t in tokens):
        return True
    # Skip n-grams with hex-like strings or long digit sequences
    if any(len(t) > 6 and sum(c.isdigit() for c in t) > len(t) // 2 for t in tokens):
        return True
    # Skip n-grams where all tokens are the same word repeated
    if len(set(tokens)) == 1 and len(tokens) > 1:
        return True
    return False


def plot_ngram_analysis(df, top_n=10):
    """Top bigrams and trigrams per class (URL/noise fragments filtered)."""
    ensure_figures_dir()
    labels = VALID_LABELS

    fig, axes = plt.subplots(len(labels), 2, figsize=(18, 4 * len(labels)))

    for i, label in enumerate(labels):
        texts = df[df["status"] == label]["statement"]

        for j, (n, gram_label) in enumerate([(2, "Bigram"), (3, "Trigram")]):
            vec = CountVectorizer(ngram_range=(n, n), stop_words="english",
                                  max_features=5000)
            X = vec.fit_transform(texts)
            freqs = X.sum(axis=0).A1
            names = np.array(vec.get_feature_names_out())

            # Filter out URL fragments and noise, then take top_n
            ranked_idx = freqs.argsort()[::-1]
            filtered_idx = [k for k in ranked_idx if not _is_url_fragment(names[k])]
            top_idx = filtered_idx[:top_n]

            display_count = len(top_idx)
            axes[i, j].barh(range(display_count), freqs[top_idx][::-1],
                            color=sns.color_palette("Set2")[i])
            axes[i, j].set_yticks(range(display_count))
            axes[i, j].set_yticklabels(names[top_idx][::-1], fontsize=8)
            axes[i, j].set_title(f"{label} — Top {top_n} {gram_label}s", fontsize=10)
            axes[i, j].set_xlabel("Frequency")

    fig.suptitle("N-gram Analysis by Class", fontsize=14, y=1.001)
    plt.tight_layout()
    fname = "05_ngram_analysis.png"
    fig.savefig(os.path.join(FIGURES_DIR, fname), dpi=150)
    plt.close(fig)
    print(f"[EDA] Saved: figures/{fname}")


# ── Plot 6: Cleaning Statistics ──────────────────────────────────────────────

def plot_cleaning_stats(raw_df, clean_df):
    """Summary table of data removed during cleaning, matching clean_dataset() pipeline."""
    ensure_figures_dir()

    raw_total = len(raw_df)
    clean_total = len(clean_df)

    # Replay the exact same cleaning steps as clean_dataset() to get accurate counts
    from src.data_loader import normalize_labels, clean_html_entities

    df = raw_df.copy()
    df["statement"] = df["statement"].astype(str)
    df["status"] = df["status"].astype(str).str.strip()

    # Step 1: Normalize + filter labels
    df = normalize_labels(df)
    after_label = len(df[df["status"].isin(VALID_LABELS)])
    invalid_label = raw_total - after_label

    # Step 2: HTML clean + short/empty removal (same order as clean_dataset)
    df = df[df["status"].isin(VALID_LABELS)].copy()
    df["statement"] = df["statement"].apply(clean_html_entities)
    df["statement"] = df["statement"].str.strip()
    before_short = len(df)
    df = df[df["statement"].str.len() >= 5].copy()
    empty_statement = before_short - len(df)

    # Step 3: Deduplication
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["statement", "status"])
    duplicates = before_dedup - len(df)

    stats = {
        "Raw Total Rows": raw_total,
        "Invalid Labels Removed": invalid_label,
        "Short/Empty Statements Removed": empty_statement,
        "Duplicates Removed": duplicates,
        "Final Clean Rows": clean_total,
        "Removal Rate": f"{(1 - clean_total / raw_total) * 100:.1f}%",
    }

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    table_data = [[k, v] for k, v in stats.items()]
    table = ax.table(cellText=table_data, colLabels=["Metric", "Value"],
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Style header
    for col in range(2):
        table[0, col].set_facecolor("#4472C4")
        table[0, col].set_text_props(color="white", fontweight="bold")

    ax.set_title("Data Cleaning Summary", fontsize=14, pad=20)
    plt.tight_layout()
    fname = "06_cleaning_stats.png"
    fig.savefig(os.path.join(FIGURES_DIR, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[EDA] Saved: figures/{fname}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    """Run full EDA pipeline and save all plots to figures/."""
    print("=" * 60)
    print("Exploratory Data Analysis")
    print("=" * 60)

    # Load raw data (for cleaning stats comparison)
    raw_df = load_raw_data(DATA_PATH)
    print(f"[EDA] Raw data: {len(raw_df)} rows")

    # Clean data
    clean_df = clean_dataset(raw_df)
    print(f"[EDA] Clean data: {len(clean_df)} rows")

    # Generate all plots
    print("\n[EDA] Generating plots...")
    plot_class_distribution(clean_df)
    plot_text_length_distribution(clean_df)
    plot_word_frequency(clean_df)
    plot_class_similarity_heatmap(clean_df)
    plot_ngram_analysis(clean_df)
    plot_cleaning_stats(raw_df, clean_df)

    print(f"\n[EDA] All 6 plots saved to figures/")
    print("=" * 60)


if __name__ == "__main__":
    main()
