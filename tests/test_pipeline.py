"""
Unit tests for ChurnPredictor inference engine.
"""

from pathlib import Path
import pytest
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config import PIPELINE_PATH
from src.predict import ChurnPredictor


@pytest.fixture
def predictor_instance():
    """Returns a ChurnPredictor instance if model exists."""
    if not PIPELINE_PATH.exists():
        pytest.skip("Model pipeline not trained yet. Skipping predictor tests.")
    return ChurnPredictor()


def test_predict_single(predictor_instance):
    """Verify single customer prediction returns complete payload."""
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

    res = predictor_instance.predict_single(sample_customer)

    assert "churn_probability" in res
    assert 0.0 <= res["churn_probability"] <= 1.0
    assert "predicted_status" in res
    assert res["predicted_status"] in ["Attrited Customer", "Existing Customer"]
    assert "risk_tier" in res
    assert "key_risk_drivers" in res
    assert "prescribed_retention_actions" in res
    assert len(res["prescribed_retention_actions"]) > 0


def test_predict_batch(predictor_instance):
    """Verify batch prediction returns dataframe with expected extra columns."""
    sample_batch = pd.DataFrame([
        {
            "Customer_Age": 45,
            "Gender": "M",
            "Dependent_count": 2,
            "Education_Level": "Graduate",
            "Marital_Status": "Married",
            "Income_Category": "$60K - $80K",
            "Card_Category": "Blue",
            "Months_on_book": 39,
            "Total_Relationship_Count": 5,
            "Months_Inactive_12_mon": 1,
            "Contacts_Count_12_mon": 3,
            "Credit_Limit": 12691.0,
            "Total_Revolving_Bal": 777.0,
            "Avg_Open_To_Buy": 11914.0,
            "Total_Amt_Chng_Q4_Q1": 1.335,
            "Total_Trans_Amt": 1144.0,
            "Total_Trans_Ct": 42,
            "Total_Ct_Chng_Q4_Q1": 1.625,
            "Avg_Utilization_Ratio": 0.061
        }
    ])

    result_df = predictor_instance.predict_batch(sample_batch)

    assert "Churn_Probability" in result_df.columns
    assert "Predicted_Attrition" in result_df.columns
    assert "Risk_Tier" in result_df.columns
    assert len(result_df) == 1
