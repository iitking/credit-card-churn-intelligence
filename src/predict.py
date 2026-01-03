"""
Inference Engine for Credit Card Churn Intelligence.
Provides high-performance single customer and batch prediction routines,
incorporating calibrated decision thresholds and automated retention action prescriptions.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Union
import joblib
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config import (
    PIPELINE_PATH,
    METADATA_PATH,
    DEFAULT_DECISION_THRESHOLD,
    RISK_TIERS,
    ID_COLS,
    BENCHMARK_COLS,
    TARGET_COL
)
from src.features import FeatureEngineer
from src.explainability import explain_customer_risk

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ChurnPredictor:
    """
    Production-ready predictor supporting single-record and batch inference.
    """

    def __init__(self, pipeline_path: Path = PIPELINE_PATH, metadata_path: Path = METADATA_PATH):
        self.pipeline_path = Path(pipeline_path)
        self.metadata_path = Path(metadata_path)
        self.pipeline = None
        self.metadata = {}
        self.threshold = DEFAULT_DECISION_THRESHOLD
        self._load_artifacts()

    def _load_artifacts(self):
        # Look for pipeline
        if not self.pipeline_path.exists():
            fallback_pkl = self.pipeline_path.parent.parent / "credit-card-churn-intelligence.pkl"
            if fallback_pkl.exists():
                self.pipeline_path = fallback_pkl
            else:
                raise FileNotFoundError(
                    f"Pipeline model file not found at {self.pipeline_path} or fallback."
                )

        logger.info("Loading model pipeline from %s...", self.pipeline_path)
        self.pipeline = joblib.load(self.pipeline_path)

        # Load metadata if exists
        if self.metadata_path.exists():
            with open(self.metadata_path, "r") as f:
                self.metadata = json.load(f)
            self.threshold = self.metadata.get("optimal_threshold", DEFAULT_DECISION_THRESHOLD)
            logger.info("Loaded metadata. Calibrated threshold: %.3f", self.threshold)
        else:
            logger.info("Metadata file not found, using default threshold: %.3f", self.threshold)

    @staticmethod
    def get_risk_tier(probability: float) -> Dict[str, str]:
        """
        Maps probability to risk tier metadata.
        """
        for tier_key, tier_val in RISK_TIERS.items():
            if tier_val["min"] <= probability <= tier_val["max"]:
                return {
                    "tier_code": tier_key,
                    "label": tier_val["label"],
                    "badge": tier_val["badge"]
                }
        return {"tier_code": "CRITICAL", "label": "Critical Risk", "badge": "danger"}

    def predict_single(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict churn risk for a single customer dictionary.
        """
        df_input = pd.DataFrame([customer_data])

        # Drop ID or benchmark cols if accidentally passed
        for c in ID_COLS + BENCHMARK_COLS + [TARGET_COL]:
            if c in df_input.columns:
                df_input = df_input.drop(columns=[c])

        # Probability inference
        prob = float(self.pipeline.predict_proba(df_input)[0, 1])
        prediction = int(prob >= self.threshold)
        status = "Attrited Customer" if prediction == 1 else "Existing Customer"
        risk_tier = self.get_risk_tier(prob)

        # Risk explanation & recommendations
        explanation = explain_customer_risk(customer_data, prob)

        return {
            "churn_probability": float(round(prob, 4)),
            "predicted_status": status,
            "is_churn": bool(prediction == 1),
            "decision_threshold": float(self.threshold),
            "risk_tier": risk_tier,
            "urgency_level": explanation["urgency_level"],
            "key_risk_drivers": explanation["key_risk_drivers"],
            "positive_signals": explanation["positive_signals"],
            "prescribed_retention_actions": explanation["prescribed_retention_actions"]
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run inference over a DataFrame of customers.
        Returns the original DataFrame enriched with predictions.
        """
        df_result = df.copy()

        # Prepare feature DataFrame
        drop_cols = [c for c in ID_COLS + BENCHMARK_COLS + [TARGET_COL] if c in df_result.columns]
        X = df_result.drop(columns=drop_cols)

        # Predict
        probs = self.pipeline.predict_proba(X)[:, 1]
        preds = (probs >= self.threshold).astype(int)

        df_result["Churn_Probability"] = np.round(probs, 4)
        df_result["Predicted_Attrition"] = np.where(preds == 1, "Attrited Customer", "Existing Customer")
        df_result["Risk_Tier"] = [self.get_risk_tier(p)["label"] for p in probs]

        return df_result


if __name__ == "__main__":
    predictor = ChurnPredictor()
    sample_customer = {
        "Customer_Age": 54,
        "Gender": "F",
        "Dependent_count": 3,
        "Education_Level": "Uneducated",
        "Marital_Status": "Married",
        "Income_Category": "Less than $40K",
        "Card_Category": "Blue",
        "Months_on_book": 36,
        "Total_Relationship_Count": 2,
        "Months_Inactive_12_mon": 3,
        "Contacts_Count_12_mon": 4,
        "Credit_Limit": 1438.3,
        "Total_Revolving_Bal": 0,
        "Avg_Open_To_Buy": 1438.3,
        "Total_Amt_Chng_Q4_Q1": 0.45,
        "Total_Trans_Amt": 1200,
        "Total_Trans_Ct": 24,
        "Total_Ct_Chng_Q4_Q1": 0.40,
        "Avg_Utilization_Ratio": 0.0
    }

    result = predictor.predict_single(sample_customer)
    print("\n--- SAMPLE PREDICTION RESULT ---")
    print(json.dumps(result, indent=2))
