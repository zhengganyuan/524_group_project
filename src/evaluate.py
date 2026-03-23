"""
Unified Evaluation & Visualization
=====================================
Generates all evaluation metrics and plots required by the assignment:
- Performance comparison table
- Confusion matrices
- Per-class F1 bar chart
- ROC-AUC curves
- Training curves (TextCNN)
- Feature importance (XGBoost)
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize

FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")


def ensure_figures_dir():
    os.makedirs(FIGURES_DIR, exist_ok=True)


def compute_metrics(y_true, y_pred, y_prob=None, num_classes=7):
    """Compute standard classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")

    macro_auc = None
    if y_prob is not None:
        y_bin = label_binarize(y_true, classes=list(range(num_classes)))
        try:
            macro_auc = roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr")
        except ValueError:
            macro_auc = None

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_auc": macro_auc,
    }


def print_comparison_table(results_dict):
    """Print a formatted comparison table of all models."""
    print("\n" + "=" * 75)
    print(f"{'Model':<20} {'Accuracy':>10} {'Macro-F1':>10} {'Weighted-F1':>12} {'Macro-AUC':>10}")
    print("-" * 75)
    for name, metrics in results_dict.items():
        auc_str = f"{metrics['macro_auc']:.4f}" if metrics["macro_auc"] else "N/A"
        print(f"{name:<20} {metrics['accuracy']:>10.4f} {metrics['macro_f1']:>10.4f} "
              f"{metrics['weighted_f1']:>12.4f} {auc_str:>10}")
    print("=" * 75)


def plot_confusion_matrix(y_true, y_pred, class_names, title, filename):
    """Plot and save a confusion matrix heatmap."""
    ensure_figures_dir()
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Plot] Saved: figures/{os.path.basename(path)}")


def plot_per_class_f1(results_dict, y_true_dict, y_pred_dict, class_names, filename="per_class_f1.png"):
    """Plot per-class F1 comparison bar chart across models."""
    ensure_figures_dir()
    from sklearn.metrics import f1_score as f1

    model_names = list(results_dict.keys())
    n_classes = len(class_names)
    x = np.arange(n_classes)
    width = 0.8 / len(model_names)

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, name in enumerate(model_names):
        f1_scores = f1(y_true_dict[name], y_pred_dict[name], average=None,
                       labels=list(range(n_classes)))
        ax.bar(x + i * width, f1_scores, width, label=name)

    ax.set_xlabel("Class")
    ax.set_ylabel("F1-Score")
    ax.set_title("Per-Class F1 Score Comparison")
    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Plot] Saved: figures/{os.path.basename(path)}")


def plot_roc_curves(y_true, y_prob_dict, class_names, filename="roc_curves.png"):
    """Plot One-vs-Rest ROC curves for selected models."""
    ensure_figures_dir()
    n_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fig, axes = plt.subplots(1, len(y_prob_dict), figsize=(8 * len(y_prob_dict), 7))
    if len(y_prob_dict) == 1:
        axes = [axes]

    for ax, (model_name, y_prob) in zip(axes, y_prob_dict.items()):
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{class_names[i]} (AUC={roc_auc:.2f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curves — {model_name}")
        ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Plot] Saved: figures/{os.path.basename(path)}")


def plot_training_curves(history, filename="textcnn_training_curves.png"):
    """Plot TextCNN training loss and accuracy curves."""
    ensure_figures_dir()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], "b-", label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "r-", label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("TextCNN — Loss Curves")
    ax1.legend()

    ax2.plot(epochs, history["train_acc"], "b-", label="Train Accuracy")
    ax2.plot(epochs, history["val_acc"], "r-", label="Val Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("TextCNN — Accuracy Curves")
    ax2.legend()

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Plot] Saved: figures/{os.path.basename(path)}")


def plot_feature_importance(model, feature_names=None, top_n=20,
                            filename="xgboost_feature_importance.png"):
    """Plot top-N XGBoost feature importances."""
    ensure_figures_dir()
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]

    if feature_names is not None:
        labels = [feature_names[i] for i in indices]
    else:
        labels = [f"feature_{i}" for i in indices]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(top_n), importances[indices])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"XGBoost — Top {top_n} Features")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Plot] Saved: figures/{os.path.basename(path)}")


# ── Main Entry Point ─────────────────────────────────────────────────────────

def main():
    """
    Load trained models from models/, regenerate predictions, and produce
    all evaluation plots and the comparison table.

    Expects: python src/train.py has been run first (models/ populated).
    """
    import pickle
    import torch
    from src.data_loader import load_and_clean
    from src.preprocessing import preprocess_dataframe, split_data, encode_labels
    from src.feature_engineering import build_feature_matrix, build_tfidf_vectorizer
    from src.models.neural_net import (
        TextCNN, build_vocab, create_data_loaders,
        evaluate_model, get_prediction_probs,
    )

    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(PROJECT_ROOT, "Combined Data.csv")
    MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

    print("=" * 60)
    print("Evaluation & Visualization Pipeline")
    print("=" * 60)

    # ── 1. Reproduce the same data split as training ────────────────────
    df = load_and_clean(DATA_PATH)
    df = preprocess_dataframe(df)
    train_df, test_df = split_data(df)

    y_train, label_to_idx, idx_to_label = encode_labels(train_df["status"])
    y_test, _, _ = encode_labels(test_df["status"])
    class_names = [idx_to_label[i] for i in range(len(idx_to_label))]
    num_classes = len(class_names)

    # ── 2. Build feature matrices ───────────────────────────────────────
    X_train, X_test, tfidf = build_feature_matrix(
        train_df["clean_text"], test_df["clean_text"],
        train_raw=train_df["statement"], test_raw=test_df["statement"],
    )
    tfidf_only = build_tfidf_vectorizer()
    X_train_tfidf = tfidf_only.fit_transform(train_df["clean_text"])
    X_test_tfidf = tfidf_only.transform(test_df["clean_text"])

    # ── 3. Load saved models ────────────────────────────────────────────
    def load_pkl(name):
        with open(os.path.join(MODELS_DIR, f"{name}.pkl"), "rb") as f:
            return pickle.load(f)

    majority_clf = load_pkl("majority_clf")
    nb_clf = load_pkl("nb_clf")
    xgb_model = load_pkl("xgb_model")
    xgb_no_smote = load_pkl("xgb_no_smote")

    # TextCNN
    vocab = load_pkl("vocab")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    textcnn = TextCNN(vocab_size=len(vocab), embed_dim=100,
                      num_classes=num_classes).to(device)
    textcnn.load_state_dict(
        torch.load(os.path.join(MODELS_DIR, "textcnn.pt"),
                   map_location=device, weights_only=True)
    )
    textcnn.eval()

    # Training history
    with open(os.path.join(MODELS_DIR, "textcnn_history.pkl"), "rb") as f:
        history = pickle.load(f)

    # ── 4. Generate predictions ─────────────────────────────────────────
    # Majority Class
    maj_pred = majority_clf.predict(X_test_tfidf)

    # Naive Bayes
    nb_pred = nb_clf.predict(X_test_tfidf)
    nb_prob = nb_clf.predict_proba(X_test_tfidf)

    # XGBoost
    xgb_pred = xgb_model.predict(X_test)
    xgb_prob = xgb_model.predict_proba(X_test)

    # XGBoost (no SMOTE)
    xgb_ns_pred = xgb_no_smote.predict(X_test)
    xgb_ns_prob = xgb_no_smote.predict_proba(X_test)

    # TextCNN
    train_texts = train_df["clean_text"].tolist()
    test_texts = test_df["clean_text"].tolist()
    train_labels = torch.tensor(y_train, dtype=torch.long)
    test_labels = torch.tensor(y_test, dtype=torch.long)
    _, test_loader = create_data_loaders(
        train_texts, train_labels, test_texts, test_labels,
        vocab, max_len=256, batch_size=64,
    )
    cnn_criterion = torch.nn.CrossEntropyLoss()
    _, _, cnn_pred, _ = evaluate_model(textcnn, test_loader, cnn_criterion, device)
    cnn_prob = get_prediction_probs(textcnn, test_loader, device)

    # ── 5. Compute metrics ──────────────────────────────────────────────
    results = {
        "Majority Class": compute_metrics(y_test, maj_pred),
        "Naive Bayes": compute_metrics(y_test, nb_pred, nb_prob, num_classes),
        "XGBoost (no SMOTE)": compute_metrics(y_test, xgb_ns_pred, xgb_ns_prob, num_classes),
        "XGBoost": compute_metrics(y_test, xgb_pred, xgb_prob, num_classes),
        "TextCNN": compute_metrics(y_test, cnn_pred, cnn_prob, num_classes),
    }

    # ── 6. Print comparison table ───────────────────────────────────────
    print_comparison_table(results)

    # Print detailed classification reports for main models
    for name, preds in [("XGBoost (no SMOTE)", xgb_ns_pred), ("XGBoost", xgb_pred), ("TextCNN", cnn_pred)]:
        print(f"\n{'-' * 40}")
        print(f"Classification Report - {name}")
        print(f"{'-' * 40}")
        print(classification_report(y_test, preds, target_names=class_names))

    # ── 7. Generate all plots ───────────────────────────────────────────
    print("\n[Eval] Generating plots...")

    # 7a. Confusion matrices (XGBoost + TextCNN)
    plot_confusion_matrix(y_test, xgb_pred, class_names,
                          "XGBoost — Confusion Matrix", "cm_xgboost.png")
    plot_confusion_matrix(y_test, cnn_pred, class_names,
                          "TextCNN — Confusion Matrix", "cm_textcnn.png")

    # 7b. Per-class F1 bar chart
    y_true_dict = {n: y_test for n in results}
    y_pred_dict = {
        "Majority Class": maj_pred,
        "Naive Bayes": nb_pred,
        "XGBoost (no SMOTE)": xgb_ns_pred,
        "XGBoost": xgb_pred,
        "TextCNN": cnn_pred,
    }
    plot_per_class_f1(results, y_true_dict, y_pred_dict, class_names)

    # 7c. ROC-AUC curves (XGBoost + TextCNN)
    plot_roc_curves(y_test, {"XGBoost (no SMOTE)": xgb_ns_prob, "XGBoost": xgb_prob, "TextCNN": cnn_prob}, class_names)

    # 7d. TextCNN training curves
    plot_training_curves(history)

    # 7e. XGBoost feature importance
    tfidf_names = tfidf.get_feature_names_out().tolist()
    handcrafted_names = [
        "char_count", "word_count", "avg_word_length", "sentence_count",
        "exclamation_count", "question_count", "uppercase_ratio",
        "unique_word_ratio", "first_person_pronoun_count", "negative_word_count",
    ]
    all_feature_names = tfidf_names + handcrafted_names
    plot_feature_importance(xgb_model, feature_names=all_feature_names)

    print(f"\n[Eval] All plots saved to figures/")
    print("=" * 60)


if __name__ == "__main__":
    main()
