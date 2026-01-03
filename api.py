"""
Production FastAPI Service for Credit Card Churn Intelligence.
Provides RESTful endpoints for single and batch customer inference, health checks,
and model performance telemetry.
Compatible with Pydantic v2 and FastAPI modern lifespan handlers.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import json

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.config import METADATA_PATH, DEFAULT_DECISION_THRESHOLD
from src.predict import ChurnPredictor

# Global Predictor instance
predictor: Optional[ChurnPredictor] = None


def get_predictor_instance() -> Optional[ChurnPredictor]:
    """Helper to lazily load predictor if not loaded by lifespan."""
    global predictor
    if predictor is None:
        try:
            predictor = ChurnPredictor()
        except Exception as e:
            print(f"Warning: Model could not be loaded: {e}")
    return predictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    get_predictor_instance()
    yield


# Initialize FastAPI app
app = FastAPI(
    title="Credit Card Churn Intelligence API",
    description="Enterprise REST API for customer churn risk prediction, explainability, and proactive retention actions.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for cross-platform integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# Pydantic Schemas (Pydantic v2 compatible)
# ==========================================
class CustomerFeatures(BaseModel):
    Customer_Age: int = Field(..., ge=18, le=100)
    Gender: str = Field(..., pattern="^(M|F)$")
    Dependent_count: int = Field(..., ge=0, le=10)
    Education_Level: str = Field(...)
    Marital_Status: str = Field(...)
    Income_Category: str = Field(...)
    Card_Category: str = Field(...)
    Months_on_book: int = Field(..., ge=1, le=120)
    Total_Relationship_Count: int = Field(..., ge=1, le=10)
    Months_Inactive_12_mon: int = Field(..., ge=0, le=12)
    Contacts_Count_12_mon: int = Field(..., ge=0, le=12)
    Credit_Limit: float = Field(..., gt=0)
    Total_Revolving_Bal: float = Field(..., ge=0)
    Avg_Open_To_Buy: Optional[float] = Field(None)
    Total_Amt_Chng_Q4_Q1: float = Field(..., ge=0)
    Total_Trans_Amt: float = Field(..., gt=0)
    Total_Trans_Ct: int = Field(..., ge=1)
    Total_Ct_Chng_Q4_Q1: float = Field(..., ge=0)
    Avg_Utilization_Ratio: Optional[float] = Field(None, ge=0.0, le=1.0)


class BatchPredictionRequest(BaseModel):
    customers: List[CustomerFeatures]


class RiskTierInfo(BaseModel):
    tier_code: str
    label: str
    badge: str


class PredictionResult(BaseModel):
    churn_probability: float
    predicted_status: str
    is_churn: bool
    decision_threshold: float
    risk_tier: RiskTierInfo
    urgency_level: str
    key_risk_drivers: List[str]
    positive_signals: List[str]
    prescribed_retention_actions: List[str]


class BatchPredictionResponse(BaseModel):
    total_customers: int
    predicted_churners: int
    projected_churn_rate_pct: float
    results: List[Dict[str, Any]]


# ==========================================
# Endpoints
# ==========================================
@app.get("/", tags=["General"])
def read_root():
    return {
        "message": "Welcome to Credit Card Churn Intelligence API",
        "docs_url": "/docs",
        "health_check": "/health",
        "version": "1.0.0"
    }


@app.get("/health", tags=["Health"])
def health_check():
    pred = get_predictor_instance()
    status_str = "ready" if pred is not None else "model_not_loaded"
    return {
        "status": status_str,
        "model_loaded": pred is not None
    }


@app.get("/metrics", tags=["Telemetry"])
def get_model_metrics():
    if not METADATA_PATH.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model metadata not found. Train the model first."
        )
    with open(METADATA_PATH, "r") as f:
        meta = json.load(f)
    return meta


@app.post("/predict", response_model=PredictionResult, tags=["Inference"])
def predict_churn(customer: CustomerFeatures):
    pred = get_predictor_instance()
    if pred is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Predictor engine is not initialized."
        )

    data_dict = customer.model_dump() if hasattr(customer, "model_dump") else customer.dict()
    # Compute derived defaults if omitted
    if data_dict.get("Avg_Open_To_Buy") is None:
        data_dict["Avg_Open_To_Buy"] = max(0.0, data_dict["Credit_Limit"] - data_dict["Total_Revolving_Bal"])
    if data_dict.get("Avg_Utilization_Ratio") is None:
        data_dict["Avg_Utilization_Ratio"] = round(data_dict["Total_Revolving_Bal"] / max(data_dict["Credit_Limit"], 1.0), 3)

    try:
        result = pred.predict_single(data_dict)
        return result
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch_churn(batch: BatchPredictionRequest):
    pred = get_predictor_instance()
    if pred is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Predictor engine is not initialized."
        )

    records = [c.model_dump() if hasattr(c, "model_dump") else c.dict() for c in batch.customers]
    for r in records:
        if r.get("Avg_Open_To_Buy") is None:
            r["Avg_Open_To_Buy"] = max(0.0, r["Credit_Limit"] - r["Total_Revolving_Bal"])
        if r.get("Avg_Utilization_Ratio") is None:
            r["Avg_Utilization_Ratio"] = round(r["Total_Revolving_Bal"] / max(r["Credit_Limit"], 1.0), 3)

    df_batch = pd.DataFrame(records)
    scored_df = pred.predict_batch(df_batch)

    churn_count = int((scored_df["Predicted_Attrition"] == "Attrited Customer").sum())
    total_count = len(scored_df)
    churn_rate = (churn_count / total_count * 100) if total_count > 0 else 0.0

    return {
        "total_customers": total_count,
        "predicted_churners": churn_count,
        "projected_churn_rate_pct": round(churn_rate, 2),
        "results": scored_df.to_dict(orient="records")
    }
