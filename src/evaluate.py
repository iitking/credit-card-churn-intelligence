"""
Evaluation and Business Cost/Benefit Analysis for Credit Card Churn Intelligence.
Computes technical ML metrics and translates predictions into banking ROI and expected value.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config import (
    CUSTOMER_LIFETIME_VALUE,
    RETENTION_CAMPAIGN_COST,
    RETENTION_SUCCESS_RATE,
    DEFAULT_DECISION_THRESHOLD
)


def calculate_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = DEFAULT_DECISION_THRESHOLD
) -> Dict[str, Any]:
    """
    Computes comprehensive evaluation metrics for binary classification at a specific threshold.
    """
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    # Financial / Business Metrics
    # Saved revenue from caught churners saved by campaign
    saved_revenue = tp * (CUSTOMER_LIFETIME_VALUE * RETENTION_SUCCESS_RATE)
    # Total cost of campaigns sent to all predicted churners (TP + FP)
    campaign_cost = (tp + fp) * RETENTION_CAMPAIGN_COST
    # Net business value
    net_value = saved_revenue - campaign_cost

    return {
        "threshold": float(threshold),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        },
        "business_impact": {
            "saved_revenue_usd": float(round(saved_revenue, 2)),
            "retention_costs_usd": float(round(campaign_cost, 2)),
            "net_benefit_usd": float(round(net_value, 2)),
            "roi_ratio": float(round(saved_revenue / max(campaign_cost, 1.0), 2))
        }
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    criterion: str = "business_value"
) -> Tuple[float, Dict[str, Any]]:
    """
    Finds the optimal decision threshold between 0.05 and 0.95.
    Criterion: 'business_value' or 'f1'.
    """
    best_score = -float("inf")
    best_thresh = DEFAULT_DECISION_THRESHOLD
    best_metrics = {}

    thresholds = np.linspace(0.05, 0.90, 86)

    for thresh in thresholds:
        m = calculate_metrics(y_true, y_prob, threshold=thresh)
        if criterion == "business_value":
            score = m["business_impact"]["net_benefit_usd"]
        else:
            score = m["f1"]

        if score > best_score:
            best_score = score
            best_thresh = float(round(thresh, 3))
            best_metrics = m

    return best_thresh, best_metrics


def print_evaluation_summary(metrics: Dict[str, Any], model_name: str = "Champion Model"):
    """
    Prints a formatted evaluation report to stdout.
    """
    cm = metrics["confusion_matrix"]
    bi = metrics["business_impact"]

    print("\n" + "="*60)
    print(f"📊 EVALUATION REPORT: {model_name} (Threshold: {metrics['threshold']:.2f})")
    print("="*60)
    print(f"• Accuracy:  {metrics['accuracy'] * 100:.2f}%")
    print(f"• Precision: {metrics['precision'] * 100:.2f}%")
    print(f"• Recall:    {metrics['recall'] * 100:.2f}%")
    print(f"• F1 Score:  {metrics['f1'] * 100:.2f}%")
    print(f"• ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"• PR-AUC:    {metrics['pr_auc']:.4f}")
    print("\n🗂 Confusion Matrix:")
    print(f"  [TN: {cm['true_negatives']:<5} | FP: {cm['false_positives']:<5}]")
    print(f"  [FN: {cm['false_negatives']:<5} | TP: {cm['true_positives']:<5}]")
    print("\n💰 Banking Financial Impact:")
    print(f"  • Gross Saved Revenue: ${bi['saved_revenue_usd']:,.2f}")
    print(f"  • Campaign Costs:      ${bi['retention_costs_usd']:,.2f}")
    print(f"  • Net Business Value:  ${bi['net_benefit_usd']:,.2f}")
    print(f"  • Campaign ROI:        {bi['roi_ratio']:.2f}x")
    print("="*60 + "\n")
