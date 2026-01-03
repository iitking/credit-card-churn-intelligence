"""
Feature Engineering and Preprocessing Pipeline for Credit Card Churn Intelligence.
Defines domain-specific feature transformers and scikit-learn preprocessing pipelines.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config import (
    ORDINAL_CATEGORIES,
    HIERARCHY_COLS,
    NOMINAL_COLS,
    BASE_NUMERICAL_COLS,
    ENGINEERED_COLS
)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom Scikit-Learn Transformer that derives domain-specific banking features:
    - Avg_Transaction_Amt: Average dollar value per transaction
    - Trans_Count_Per_Month: Transaction frequency relative to customer tenure
    - Trans_Amt_Per_Month: Spending velocity relative to tenure
    - Inactive_Ratio: Proportion of months inactive in the past year
    - Contact_Intensity: Interaction frequency relative to inactivity
    - Revolving_to_Credit_Ratio: Proportion of revolving balance against credit line
    - Total_Products_Per_Year: Account deepening rate
    """

    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_df = X.copy()

        # 1. Average Transaction Amount
        if "Total_Trans_Amt" in X_df.columns and "Total_Trans_Ct" in X_df.columns:
            X_df["Avg_Transaction_Amt"] = (
                X_df["Total_Trans_Amt"] / (X_df["Total_Trans_Ct"] + 1e-5)
            )
        else:
            X_df["Avg_Transaction_Amt"] = 0.0

        # 2. Transaction Count Per Month of Tenure
        if "Total_Trans_Ct" in X_df.columns and "Months_on_book" in X_df.columns:
            X_df["Trans_Count_Per_Month"] = (
                X_df["Total_Trans_Ct"] / (X_df["Months_on_book"] + 1e-5)
            )
        else:
            X_df["Trans_Count_Per_Month"] = 0.0

        # 3. Transaction Amount Per Month of Tenure
        if "Total_Trans_Amt" in X_df.columns and "Months_on_book" in X_df.columns:
            X_df["Trans_Amt_Per_Month"] = (
                X_df["Total_Trans_Amt"] / (X_df["Months_on_book"] + 1e-5)
            )
        else:
            X_df["Trans_Amt_Per_Month"] = 0.0

        # 4. Inactive Ratio
        if "Months_Inactive_12_mon" in X_df.columns:
            X_df["Inactive_Ratio"] = (
                X_df["Months_Inactive_12_mon"] / 12.0
            )
        else:
            X_df["Inactive_Ratio"] = 0.0

        # 5. Contact Intensity
        if "Contacts_Count_12_mon" in X_df.columns and "Months_Inactive_12_mon" in X_df.columns:
            X_df["Contact_Intensity"] = (
                X_df["Contacts_Count_12_mon"] / (X_df["Months_Inactive_12_mon"] + 1.0)
            )
        else:
            X_df["Contact_Intensity"] = 0.0

        # 6. Revolving Balance to Credit Limit Ratio
        if "Total_Revolving_Bal" in X_df.columns and "Credit_Limit" in X_df.columns:
            X_df["Revolving_to_Credit_Ratio"] = (
                X_df["Total_Revolving_Bal"] / (X_df["Credit_Limit"] + 1e-5)
            )
        else:
            X_df["Revolving_to_Credit_Ratio"] = 0.0

        # 7. Total Products Per Year
        if "Total_Relationship_Count" in X_df.columns and "Months_on_book" in X_df.columns:
            X_df["Total_Products_Per_Year"] = (
                X_df["Total_Relationship_Count"] / ((X_df["Months_on_book"] / 12.0) + 1e-5)
            )
        else:
            X_df["Total_Products_Per_Year"] = 0.0

        return X_df


def build_preprocessor() -> ColumnTransformer:
    """
    Constructs the ColumnTransformer for categorical and numerical features.
    """
    # Ordinal encoder categories
    ordinal_cats = [ORDINAL_CATEGORIES[col] for col in HIERARCHY_COLS]

    ordinal_pipeline = Pipeline([
        ("ordinal_enc", OrdinalEncoder(
            categories=ordinal_cats,
            handle_unknown="use_encoded_value",
            unknown_value=-1
        ))
    ])

    # One-Hot encoder for nominal categories
    nominal_pipeline = Pipeline([
        ("onehot_enc", OneHotEncoder(
            drop="first",
            sparse_output=False,
            handle_unknown="ignore"
        ))
    ])

    # Numerical pipeline with standard scaling
    all_num_cols = BASE_NUMERICAL_COLS + ENGINEERED_COLS
    numerical_pipeline = Pipeline([
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("ordinal", ordinal_pipeline, HIERARCHY_COLS),
            ("nominal", nominal_pipeline, NOMINAL_COLS),
            ("numerical", numerical_pipeline, all_num_cols),
        ],
        remainder="drop"
    )

    return preprocessor


def build_full_feature_pipeline() -> Pipeline:
    """
    Creates full preprocessing pipeline including feature engineering and transformations.
    """
    return Pipeline([
        ("feature_engineer", FeatureEngineer()),
        ("preprocessor", build_preprocessor())
    ])


def get_feature_names(preprocessor: ColumnTransformer) -> list:
    """
    Extracts output feature names after ColumnTransformer fit.
    """
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        # Fallback names
        names = []
        for name, trans, cols in preprocessor.transformers_:
            if name == "ordinal":
                names.extend([f"ordinal__{c}" for c in cols])
            elif name == "nominal":
                if hasattr(trans.named_steps["onehot_enc"], "get_feature_names_out"):
                    names.extend(trans.named_steps["onehot_enc"].get_feature_names_out(cols))
                else:
                    names.extend([f"nominal__{c}" for c in cols])
            elif name == "numerical":
                names.extend([f"num__{c}" for c in cols])
        return names
