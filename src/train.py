"""
Full Training and Model Selection Pipeline for Credit Card Churn Intelligence.
Trains baseline and advanced models, compares cross-validation scores,
calibrates decision threshold, and exports serialized production pipeline and metadata.
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import joblib
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config import (
    MODELS_DIR,
    PIPELINE_PATH,
    METADATA_PATH,
    METRICS_SUMMARY_PATH,
    RANDOM_SEED,
    CV_FOLDS
)
from src.data_loader import get_train_test_split
from src.features import FeatureEngineer, build_preprocessor, get_feature_names
from src.models import get_models
from src.evaluate import calculate_metrics, find_optimal_threshold, print_evaluation_summary
from src.explainability import extract_feature_importances

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def train_and_evaluate_all():
    """
    Main training execution function.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    logger.info("Loading and splitting dataset...")
    X_train, X_test, y_train, y_test = get_train_test_split()

    # 2. Fit Feature Engineering and Preprocessing
    logger.info("Fitting feature engineering and preprocessing transformers...")
    feature_engineer = FeatureEngineer()
    X_train_fe = feature_engineer.fit_transform(X_train)
    X_test_fe = feature_engineer.transform(X_test)

    preprocessor = build_preprocessor()
    X_train_trans = preprocessor.fit_transform(X_train_fe)
    X_test_trans = preprocessor.transform(X_test_fe)

    feature_names = get_feature_names(preprocessor)
    logger.info("Transformed features shape: %s | Feature count: %d", X_train_trans.shape, len(feature_names))

    # 3. Model Training & Comparison
    scale_pos_weight = float((len(y_train) - sum(y_train)) / sum(y_train))
    logger.info("Computed scale_pos_weight: %.2f", scale_pos_weight)

    candidate_models = get_models(scale_pos_weight=scale_pos_weight, random_state=RANDOM_SEED)

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    model_results = []
    trained_estimators = {}

    for name, model in candidate_models.items():
        logger.info("Evaluating candidate model: %s...", name)

        # 5-fold CV on F1
        cv_f1 = cross_val_score(model, X_train_trans, y_train, cv=cv, scoring="f1", n_jobs=-1)
        cv_roc = cross_val_score(model, X_train_trans, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)

        # Fit on full train
        model.fit(X_train_trans, y_train)
        trained_estimators[name] = model

        # Predict probabilities
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_trans)[:, 1]
        else:
            y_prob = model.decision_function(X_test_trans)
            y_prob = 1 / (1 + np.exp(-y_prob))

        metrics = calculate_metrics(y_test.values, y_prob, threshold=0.50)

        record = {
            "model_name": name,
            "cv_f1_mean": float(round(cv_f1.mean(), 4)),
            "cv_f1_std": float(round(cv_f1.std(), 4)),
            "cv_roc_auc_mean": float(round(cv_roc.mean(), 4)),
            "test_accuracy": metrics["accuracy"],
            "test_precision": metrics["precision"],
            "test_recall": metrics["recall"],
            "test_f1": metrics["f1"],
            "test_roc_auc": metrics["roc_auc"],
            "test_pr_auc": metrics["pr_auc"],
            "net_business_value_usd": metrics["business_impact"]["net_benefit_usd"]
        }
        model_results.append(record)
        logger.info(
            "[%s] CV F1: %.4f | Test F1: %.4f | Test ROC-AUC: %.4f",
            name, record["cv_f1_mean"], record["test_f1"], record["test_roc_auc"]
        )

    # Convert to DataFrame and sort
    results_df = pd.DataFrame(model_results).sort_values(by="test_f1", ascending=False)
    print("\n" + "="*80)
    print("🏆 CANDIDATE MODEL BENCHMARK RESULTS")
    print("="*80)
    print(results_df.to_string(index=False))

    # Save summary
    with open(METRICS_SUMMARY_PATH, "w") as f:
        json.dump(model_results, f, indent=4)

    # 4. Select Champion Model
    champion_name = results_df.iloc[0]["model_name"]
    champion_model = trained_estimators[champion_name]
    logger.info("Champion Model Selected: %s", champion_name)

    # 5. Threshold Calibration for Champion Model
    champion_prob = champion_model.predict_proba(X_test_trans)[:, 1]
    optimal_thresh, optimal_metrics = find_optimal_threshold(y_test.values, champion_prob, criterion="business_value")
    logger.info("Optimal Decision Threshold: %.3f", optimal_thresh)
    print_evaluation_summary(optimal_metrics, model_name=f"{champion_name} (Calibrated)")

    # 6. Feature Importances
    importances = extract_feature_importances(champion_model, feature_names)

    # 7. Build Full End-to-End Pipeline
    full_pipeline = Pipeline([
        ("feature_engineer", feature_engineer),
        ("preprocessor", preprocessor),
        ("classifier", champion_model)
    ])

    # 8. Save Artifacts
    logger.info("Saving champion pipeline to %s", PIPELINE_PATH)
    joblib.dump(full_pipeline, PIPELINE_PATH)

    # Also save as credit-card-churn-intelligence.pkl in project root for backward compatibility
    legacy_pkl_path = MODELS_DIR.parent / "credit-card-churn-intelligence.pkl"
    joblib.dump(full_pipeline, legacy_pkl_path)
    logger.info("Saved backward-compatible model to %s", legacy_pkl_path)

    metadata = {
        "model_name": champion_name,
        "created_at": datetime.now().isoformat(),
        "random_seed": RANDOM_SEED,
        "optimal_threshold": optimal_thresh,
        "metrics_at_optimal_threshold": optimal_metrics,
        "feature_names": feature_names,
        "top_features": importances[:15],
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "target_classes": {"0": "Existing Customer", "1": "Attrited Customer"}
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=4)
    logger.info("Model metadata saved to %s", METADATA_PATH)

    return full_pipeline, metadata


if __name__ == "__main__":
    train_and_evaluate_all()
