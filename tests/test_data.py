"""
Unit tests for data loading, cleaning, and preparation.
"""

from pathlib import Path
import pytest
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config import DATA_PATH, TARGET_COL, ID_COLS
from src.data_loader import load_raw_data, clean_data, prepare_features_and_target, get_train_test_split


def test_load_raw_data():
    """Verify raw dataset is loaded and not empty."""
    df = load_raw_data(DATA_PATH)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 1000
    assert TARGET_COL in df.columns


def test_clean_data():
    """Verify identifiers and naive bayes columns are removed."""
    df = load_raw_data(DATA_PATH)
    cleaned = clean_data(df)

    for col in ID_COLS:
        assert col not in cleaned.columns

    nb_cols = [c for c in cleaned.columns if "naive_bayes" in c.lower()]
    assert len(nb_cols) == 0


def test_prepare_features_and_target():
    """Verify target is mapped to binary 0 and 1."""
    df = load_raw_data(DATA_PATH)
    X, y = prepare_features_and_target(df)

    assert set(y.unique()).issubset({0, 1})
    assert len(X) == len(y)
    assert TARGET_COL not in X.columns


def test_get_train_test_split():
    """Verify stratified split maintains class balance."""
    X_train, X_test, y_train, y_test = get_train_test_split(test_size=0.2, random_seed=42)

    assert len(X_train) + len(X_test) > 1000
    train_rate = y_train.mean()
    test_rate = y_test.mean()

    # Stratified split should have roughly identical churn rate
    assert abs(train_rate - test_rate) < 0.02
