"""
XGBoost Multi-Class Classifier
================================
Wrapper for XGBoost with class-weight support and SMOTE integration.
"""

import numpy as np
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


def create_xgboost_model(num_classes=7, random_state=42):
    """Create an XGBoost classifier with default hyperparameters from the plan."""
    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=num_classes,
        eval_metric="mlogloss",
        random_state=random_state,
        n_jobs=-1,
    )


def train_xgboost(model, X_train, y_train, X_val=None, y_val=None,
                   use_class_weight=True):
    """
    Train the XGBoost model with optional class weighting.

    Args:
        model: XGBClassifier instance
        X_train: Training feature matrix
        y_train: Training labels (integer-encoded)
        X_val: Optional validation feature matrix
        y_val: Optional validation labels
        use_class_weight: Whether to use balanced sample weights

    Returns:
        Fitted model
    """
    fit_params = {}

    if use_class_weight:
        sample_weights = compute_sample_weight("balanced", y_train)
        fit_params["sample_weight"] = sample_weights

    if X_val is not None and y_val is not None:
        fit_params["eval_set"] = [(X_val, y_val)]
        fit_params["verbose"] = False

    model.fit(X_train, y_train, **fit_params)
    return model
