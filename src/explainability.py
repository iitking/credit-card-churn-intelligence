"""
Explainability and Retention Prescriptions for Credit Card Churn Intelligence.
Extracts model feature importances, local risk factor breakdowns, and prescribes
actionable banking retention interventions.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd


def extract_feature_importances(model, feature_names: List[str]) -> List[Dict[str, Any]]:
    """
    Extracts sorted feature importances from tree-based or linear models.
    """
    importances = None

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])

    if importances is None or len(importances) != len(feature_names):
        return []

    # Normalize to percentages
    total = np.sum(importances)
    normalized = (importances / total) * 100 if total > 0 else importances

    records = [
        {"feature": name, "importance": float(round(score, 3))}
        for name, score in zip(feature_names, normalized)
    ]
    return sorted(records, key=lambda x: x["importance"], reverse=True)


def explain_customer_risk(
    customer_data: Dict[str, Any],
    churn_probability: float
) -> Dict[str, Any]:
    """
    Analyzes an individual customer's attributes against banking heuristics
    to explain the driving churn factors and suggest targeted retention actions.
    """
    risk_factors: List[str] = []
    positive_factors: List[str] = []
    retention_actions: List[str] = []

    # 1. Transaction Count & Velocity
    trans_ct = customer_data.get("Total_Trans_Ct", 60)
    if trans_ct < 45:
        risk_factors.append(f"Low transaction frequency ({trans_ct} transactions/year; active median is ~68).")
        retention_actions.append("Trigger 'Swipe & Win' 5x reward points campaign on everyday merchant spend.")
    elif trans_ct > 75:
        positive_factors.append(f"High card usage ({trans_ct} transactions/year).")

    # 2. Activity Drop (Change Q4 vs Q1)
    amt_chng = customer_data.get("Total_Amt_Chng_Q4_Q1", 0.75)
    ct_chng = customer_data.get("Total_Ct_Chng_Q4_Q1", 0.75)
    if amt_chng < 0.60 or ct_chng < 0.60:
        risk_factors.append(f"Substantial drop in quarterly spending velocity (Q4/Q1 transaction change: {ct_chng:.2f}).")
        retention_actions.append("Send immediate personalized merchant discount offers in dining & travel categories.")

    # 3. Customer Service Contacts
    contacts = customer_data.get("Contacts_Count_12_mon", 2)
    if contacts >= 3:
        risk_factors.append(f"Elevated support contacts ({contacts} calls in 12 months), signaling customer friction.")
        retention_actions.append("Initiate proactive outbound concierge call from senior relationship manager.")

    # 4. Inactivity
    months_inactive = customer_data.get("Months_Inactive_12_mon", 2)
    if months_inactive >= 3:
        risk_factors.append(f"Extended account inactivity ({months_inactive} months inactive in past year).")
        retention_actions.append("Offer 90-day 0% APR promo or fee rebate upon card reactivation.")

    # 5. Revolving Balance & Utilization
    revolving = customer_data.get("Total_Revolving_Bal", 1000)
    utilization = customer_data.get("Avg_Utilization_Ratio", 0.25)
    if revolving < 500 and utilization < 0.05:
        risk_factors.append("Near-zero credit card utilization (<5%), indicating customer is migrating spend to competitors.")
        retention_actions.append("Offer competitive balance transfer rate or temporary credit limit increase.")
    elif utilization > 0.60:
        positive_factors.append(f"Healthy credit line utilization ({utilization*100:.1f}%).")

    # 6. Total Relationship Products
    products = customer_data.get("Total_Relationship_Count", 3)
    if products <= 2:
        risk_factors.append(f"Weak product stickiness (only {products} total bank products held).")
        retention_actions.append("Cross-sell High-Yield Savings or auto-loan discount to increase customer lock-in.")
    elif products >= 5:
        positive_factors.append(f"Strong multi-product banking relationship ({products} products).")

    # Determine priority tier
    if churn_probability >= 0.75:
        urgency = "CRITICAL: Immediate Senior Outreach Required within 24 Hours"
    elif churn_probability >= 0.50:
        urgency = "HIGH: Automated Retention Workflow + Targeted Offer Triggered"
    elif churn_probability >= 0.25:
        urgency = "MODERATE: Monitor & Enroll in Engagement Nurture Track"
    else:
        urgency = "LOW: Healthy Account - Standard Loyalty Rewards"

    if not retention_actions:
        retention_actions.append("Maintain standard engagement and quarterly satisfaction pulse surveys.")

    return {
        "urgency_level": urgency,
        "key_risk_drivers": risk_factors,
        "positive_signals": positive_factors,
        "prescribed_retention_actions": retention_actions
    }
