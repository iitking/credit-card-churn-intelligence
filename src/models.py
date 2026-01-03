"""
Model factory for Credit Card Churn Intelligence.
Defines candidate machine learning models with hyperparameter configurations
and class imbalance handling.
"""

from typing import Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier
)

# Optional advanced gradient boosting libraries with graceful fallback
try:
    from xgboost import XGBClassifier
    # Verify native library can actually be loaded
    _dummy = XGBClassifier()
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMClassifier
    # Verify native library can actually be loaded
    _dummy = LGBMClassifier()
    HAS_LIGHTGBM = True
except Exception:
    HAS_LIGHTGBM = False


def get_models(scale_pos_weight: float = 5.22, random_state: int = 42) -> Dict[str, Any]:
    """
    Returns a dictionary of candidate models configured for imbalanced churn classification.
    scale_pos_weight is approximately (Negatives / Positives) = 8500 / 1627 ≈ 5.22.
    """
    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            C=0.5,
            random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=250,
            learning_rate=0.08,
            max_depth=5,
            subsample=0.85,
            random_state=random_state
        ),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=250,
            max_depth=6,
            min_samples_leaf=20,
            random_state=random_state
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=250,
            max_depth=14,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1
        )
    }

    if HAS_XGBOOST:
        models["XGBoost"] = XGBClassifier(
            n_estimators=280,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.80,
            min_child_weight=3,
            gamma=0.5,
            reg_alpha=0.5,
            reg_lambda=2.0,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            eval_metric="logloss",
            n_jobs=-1
        )

    if HAS_LIGHTGBM:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=280,
            max_depth=6,
            learning_rate=0.08,
            num_leaves=31,
            subsample=0.85,
            colsample_bytree=0.80,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1
        )

    return models


def create_ensemble_model(best_estimators: list) -> VotingClassifier:
    """
    Creates a Soft Voting Classifier combining the top models.
    """
    return VotingClassifier(
        estimators=best_estimators,
        voting="soft",
        n_jobs=-1
    )
