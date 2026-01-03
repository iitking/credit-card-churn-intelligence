"""
Configuration module for Credit Card Churn Intelligence System.
Centralizes paths, column mappings, categorical hierarchies, hyperparameter presets,
and business threshold settings.
"""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT
DATA_PATH = DATA_DIR / "BankChurners.csv"
MODELS_DIR = PROJECT_ROOT / "models"
PIPELINE_PATH = MODELS_DIR / "best_churn_pipeline.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
METRICS_SUMMARY_PATH = MODELS_DIR / "metrics_summary.json"

# Target & Identifiers
TARGET_COL = "Attrition_Flag"
TARGET_MAPPING = {
    "Existing Customer": 0,
    "Attrited Customer": 1
}
INVERSE_TARGET_MAPPING = {v: k for k, v in TARGET_MAPPING.items()}

ID_COLS = ["CLIENTNUM"]
BENCHMARK_COLS = [
    "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1",
    "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2"
]

# Categorical Hierarchies (Ordinal)
ORDINAL_CATEGORIES = {
    "Income_Category": [
        "Unknown",
        "Less than $40K",
        "$40K - $60K",
        "$60K - $80K",
        "$80K - $120K",
        "$120K +"
    ],
    "Card_Category": [
        "Blue",
        "Silver",
        "Gold",
        "Platinum"
    ],
    "Education_Level": [
        "Unknown",
        "Uneducated",
        "High School",
        "College",
        "Graduate",
        "Post-Graduate",
        "Doctorate"
    ]
}

HIERARCHY_COLS = list(ORDINAL_CATEGORIES.keys())
NOMINAL_COLS = ["Gender", "Marital_Status"]

# Base Raw Numerical Columns
BASE_NUMERICAL_COLS = [
    "Customer_Age",
    "Dependent_count",
    "Months_on_book",
    "Total_Relationship_Count",
    "Months_Inactive_12_mon",
    "Contacts_Count_12_mon",
    "Credit_Limit",
    "Total_Revolving_Bal",
    "Avg_Open_To_Buy",
    "Total_Amt_Chng_Q4_Q1",
    "Total_Trans_Amt",
    "Total_Trans_Ct",
    "Total_Ct_Chng_Q4_Q1",
    "Avg_Utilization_Ratio"
]

# Engineered Features
ENGINEERED_COLS = [
    "Avg_Transaction_Amt",
    "Trans_Count_Per_Month",
    "Trans_Amt_Per_Month",
    "Inactive_Ratio",
    "Contact_Intensity",
    "Revolving_to_Credit_Ratio",
    "Total_Products_Per_Year"
]

# Random State
RANDOM_SEED = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# Optimal Decision Threshold for Churn (Calibrated for Business Value)
DEFAULT_DECISION_THRESHOLD = 0.35

# Business Risk Tiers
RISK_TIERS = {
    "LOW": {"min": 0.00, "max": 0.25, "label": "Low Risk", "badge": "success"},
    "MODERATE": {"min": 0.25, "max": 0.50, "label": "Moderate Risk", "badge": "info"},
    "HIGH": {"min": 0.50, "max": 0.75, "label": "High Risk", "badge": "warning"},
    "CRITICAL": {"min": 0.75, "max": 1.00, "label": "Critical Risk", "badge": "danger"}
}

# Financial ROI Constants (Bank Credit Card Churn Economics)
CUSTOMER_LIFETIME_VALUE = 1200.0   # Average customer net revenue over 2 years
RETENTION_CAMPAIGN_COST = 60.0     # Cost per targeted retention campaign
RETENTION_SUCCESS_RATE = 0.40      # Percentage of churners persuaded to stay
