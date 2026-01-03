"""
Unit tests for FastAPI service endpoints.
"""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api import app

client = TestClient(app)


def test_root():
    """Verify root endpoint responds with metadata."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "version" in data


def test_health():
    """Verify health endpoint responds."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data


def test_predict_endpoint():
    """Verify /predict endpoint returns churn probability and recommendations."""
    payload = {
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
        "Total_Revolving_Bal": 0.0,
        "Total_Amt_Chng_Q4_Q1": 0.45,
        "Total_Trans_Amt": 1200.0,
        "Total_Trans_Ct": 24,
        "Total_Ct_Chng_Q4_Q1": 0.40
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "churn_probability" in data
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert "risk_tier" in data
    assert "prescribed_retention_actions" in data
    assert len(data["prescribed_retention_actions"]) > 0


def test_predict_batch_endpoint():
    """Verify /predict/batch endpoint processes list of customers."""
    payload = {
        "customers": [
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
                "Total_Amt_Chng_Q4_Q1": 1.335,
                "Total_Trans_Amt": 1144.0,
                "Total_Trans_Ct": 42,
                "Total_Ct_Chng_Q4_Q1": 1.625
            }
        ]
    }

    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["total_customers"] == 1
    assert "projected_churn_rate_pct" in data
    assert len(data["results"]) == 1
