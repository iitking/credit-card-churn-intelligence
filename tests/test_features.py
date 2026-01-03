"""
Unit tests for feature engineering transformer and scikit-learn preprocessor.
"""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features import FeatureEngineer, build_preprocessor, get_feature_names
from src.data_loader import get_train_test_split


def test_feature_engineer():
    """Verify feature engineering generates all derived banking ratios."""
    fe = FeatureEngineer()
    df_dummy = pd.DataFrame([{
        "Total_Trans_Amt": 1200,
        "Total_Trans_Ct": 40,
        "Months_on_book": 24,
        "Months_Inactive_12_mon": 2,
        "Contacts_Count_12_mon": 3,
        "Total_Revolving_Bal": 500,
        "Credit_Limit": 5000,
        "Total_Relationship_Count": 4
    }])

    transformed = fe.transform(df_dummy)

    assert "Avg_Transaction_Amt" in transformed.columns
    assert "Trans_Count_Per_Month" in transformed.columns
    assert "Trans_Amt_Per_Month" in transformed.columns
    assert "Inactive_Ratio" in transformed.columns
    assert "Contact_Intensity" in transformed.columns
    assert "Revolving_to_Credit_Ratio" in transformed.columns
    assert "Total_Products_Per_Year" in transformed.columns

    # Check math
    assert np.isclose(transformed["Avg_Transaction_Amt"].iloc[0], 1200 / 40, atol=1e-3)
    assert np.isclose(transformed["Trans_Count_Per_Month"].iloc[0], 40 / 24, atol=1e-3)
    assert np.isclose(transformed["Inactive_Ratio"].iloc[0], 2 / 12, atol=1e-3)


def test_preprocessor_fit_transform():
    """Verify full preprocessor transforms train and test sets cleanly."""
    X_train, X_test, _, _ = get_train_test_split(test_size=0.2, random_seed=42)

    fe = FeatureEngineer()
    X_train_fe = fe.fit_transform(X_train)
    X_test_fe = fe.transform(X_test)

    prep = build_preprocessor()
    X_train_trans = prep.fit_transform(X_train_fe)
    X_test_trans = prep.transform(X_test_fe)

    assert X_train_trans.shape[0] == len(X_train)
    assert X_test_trans.shape[0] == len(X_test)
    assert not np.isnan(X_train_trans).any()
    assert not np.isnan(X_test_trans).any()

    feature_names = get_feature_names(prep)
    assert len(feature_names) == X_train_trans.shape[1]
