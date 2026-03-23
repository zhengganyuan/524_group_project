"""
Unified Training Entry
========================
Orchestrates the full training pipeline:
1. Load & clean data
2. Preprocess text
3. Split data (stratified)
4. Build features
5. Train all models (Baselines + XGBoost + TextCNN)
6. Save trained models
"""

import os
import pickle

import numpy as np
import torch
import torch.nn as nn
from imblearn.over_sampling import SMOTE
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils.class_weight import compute_class_weight

from src.data_loader import load_and_clean
from src.preprocessing import preprocess_dataframe, split_data, encode_labels
from src.feature_engineering import build_feature_matrix, build_tfidf_vectorizer
from src.models.xgboost_model import create_xgboost_model, train_xgboost
from src.models.neural_net import (
    TextCNN, build_vocab, load_glove_embeddings,
    create_data_loaders, train_one_epoch, evaluate_model,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "Combined Data.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
GLOVE_PATH = os.path.join(PROJECT_ROOT, "data", "glove.6B.100d.txt")


def train_baselines(X_train_tfidf, y_train, X_test_tfidf, y_test):
    """Train baseline models: Majority Class + Naive Bayes."""
    print("\n" + "=" * 60)
    print("Training Baselines")
    print("=" * 60)

    # Majority Class
    majority_clf = DummyClassifier(strategy="most_frequent")
    majority_clf.fit(X_train_tfidf, y_train)
    maj_acc = majority_clf.score(X_test_tfidf, y_test)
    print(f"[Baseline] Majority Class accuracy: {maj_acc:.4f}")

    # Naive Bayes
    nb_clf = MultinomialNB(alpha=1.0)
    nb_clf.fit(X_train_tfidf, y_train)
    nb_acc = nb_clf.score(X_test_tfidf, y_test)
    print(f"[Baseline] Naive Bayes accuracy: {nb_acc:.4f}")

    return majority_clf, nb_clf


def train_xgboost_pipeline(X_train, y_train, X_test, y_test, use_smote=True):
    """Train XGBoost with optional SMOTE oversampling."""
    print("\n" + "=" * 60)
    print(f"Training XGBoost (SMOTE={use_smote})")
    print("=" * 60)

    X_tr, y_tr = X_train, y_train
    if use_smote:
        smote = SMOTE(random_state=42)
        X_tr, y_tr = smote.fit_resample(X_train, y_train)
        print(f"[SMOTE] Resampled: {X_train.shape[0]} -> {X_tr.shape[0]}")

    model = create_xgboost_model()
    model = train_xgboost(model, X_tr, y_tr, X_test, y_test, use_class_weight=True)

    acc = model.score(X_test, y_test)
    print(f"[XGBoost] Test accuracy: {acc:.4f}")
    return model


def train_textcnn_pipeline(train_df, label_to_idx,
                           glove_path=None, device=None):
    """Train TextCNN with GloVe embeddings and internal validation split."""
    print("\n" + "=" * 60)
    print("Training TextCNN")
    print("=" * 60)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TextCNN] Using device: {device}")

    # Split training data 90/10 for early stopping to avoid test set leakage
    from sklearn.model_selection import train_test_split as _split
    train_sub, val_sub = _split(
        train_df, test_size=0.1, random_state=42, stratify=train_df["status"]
    )
    train_sub = train_sub.reset_index(drop=True)
    val_sub = val_sub.reset_index(drop=True)
    print(f"[TextCNN] Internal split: {len(train_sub)} train, {len(val_sub)} val")

    train_texts = train_sub["clean_text"].tolist()
    val_texts = val_sub["clean_text"].tolist()
    train_labels = torch.tensor(
        train_sub["status"].map(label_to_idx).values, dtype=torch.long
    )
    val_labels = torch.tensor(
        val_sub["status"].map(label_to_idx).values, dtype=torch.long
    )

    # Build vocab
    vocab = build_vocab(train_texts, max_size=20000)

    # Load GloVe if available
    embed_dim = 100
    pretrained = None
    if glove_path and os.path.exists(glove_path):
        pretrained = load_glove_embeddings(glove_path, vocab, embed_dim)
    else:
        print("[TextCNN] GloVe not found, using random initialization")

    # Create data loaders (val_loader for early stopping, NOT test set)
    train_loader, val_loader = create_data_loaders(
        train_texts, train_labels, val_texts, val_labels,
        vocab, max_len=256, batch_size=64,
    )

    # Build model
    model = TextCNN(
        vocab_size=len(vocab),
        embed_dim=embed_dim,
        num_classes=len(label_to_idx),
        pretrained_embeddings=pretrained,
    ).to(device)

    # Class weights for loss function
    class_weights = compute_class_weight(
        "balanced",
        classes=np.arange(len(label_to_idx)),
        y=train_sub["status"].map(label_to_idx).values,
    )
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32).to(device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=2
    )

    # Training loop with early stopping
    best_val_acc = 0
    patience_counter = 0
    patience = 3
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(20):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, _, _ = evaluate_model(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"  Epoch {epoch+1:2d} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # Restore best model
    model.load_state_dict(best_state)
    print(f"[TextCNN] Best validation accuracy: {best_val_acc:.4f}")

    return model, vocab, history


def save_models(models_dict, save_dir):
    """Save all trained models to disk."""
    os.makedirs(save_dir, exist_ok=True)
    for name, obj in models_dict.items():
        path = os.path.join(save_dir, f"{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        print(f"[Save] {name} -> models/{name}.pkl")


def main():
    """Run the full training pipeline."""
    # 1. Load and clean
    df = load_and_clean(DATA_PATH)

    # 2. Preprocess text
    df = preprocess_dataframe(df)

    # 3. Split
    train_df, test_df = split_data(df)

    # 4. Encode labels
    y_train, label_to_idx, idx_to_label = encode_labels(train_df["status"])
    y_test, _, _ = encode_labels(test_df["status"])

    # 5. Build features for traditional models
    X_train, X_test, tfidf = build_feature_matrix(
        train_df["clean_text"], test_df["clean_text"],
        train_raw=train_df["statement"], test_raw=test_df["statement"],
    )

    # Also get TF-IDF only (for Naive Bayes which needs non-negative values)
    tfidf_only = build_tfidf_vectorizer()
    X_train_tfidf = tfidf_only.fit_transform(train_df["clean_text"])
    X_test_tfidf = tfidf_only.transform(test_df["clean_text"])

    # 6. Train baselines
    majority_clf, nb_clf = train_baselines(X_train_tfidf, y_train, X_test_tfidf, y_test)

    # 7. Train XGBoost (with and without SMOTE for comparison)
    xgb_model = train_xgboost_pipeline(X_train, y_train, X_test, y_test, use_smote=True)
    xgb_no_smote = train_xgboost_pipeline(X_train, y_train, X_test, y_test, use_smote=False)

    # 8. Train TextCNN
    glove = GLOVE_PATH if os.path.exists(GLOVE_PATH) else None
    textcnn_model, vocab, history = train_textcnn_pipeline(
        train_df, label_to_idx, glove_path=glove,
    )

    # 9. Save models
    save_models({
        "majority_clf": majority_clf,
        "nb_clf": nb_clf,
        "xgb_model": xgb_model,
        "xgb_no_smote": xgb_no_smote,
        "tfidf_vectorizer": tfidf,
        "tfidf_only": tfidf_only,
        "label_to_idx": label_to_idx,
        "idx_to_label": idx_to_label,
        "vocab": vocab,
    }, MODELS_DIR)

    # Save TextCNN separately (PyTorch)
    torch.save(textcnn_model.state_dict(), os.path.join(MODELS_DIR, "textcnn.pt"))
    print("[Save] textcnn -> models/textcnn.pt")

    # Save training history
    with open(os.path.join(MODELS_DIR, "textcnn_history.pkl"), "wb") as f:
        pickle.dump(history, f)

    print("\n" + "=" * 60)
    print("Training complete! All models saved to models/")
    print("=" * 60)

    return {
        "majority_clf": majority_clf,
        "nb_clf": nb_clf,
        "xgb_model": xgb_model,
        "xgb_no_smote": xgb_no_smote,
        "textcnn_model": textcnn_model,
        "tfidf": tfidf,
        "label_to_idx": label_to_idx,
        "idx_to_label": idx_to_label,
        "vocab": vocab,
        "history": history,
        "train_df": train_df,
        "test_df": test_df,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


if __name__ == "__main__":
    main()
