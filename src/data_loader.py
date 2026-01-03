"""
Data Loader Module for Credit Card Churn Intelligence.
Handles ingestion, cleaning, validation, and train-test splits.
"""

import logging
from typing import Tuple, Optional, Union
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config import (
    DATA_PATH,
    TARGET_COL,
    TARGET_MAPPING,
    ID_COLS,
    BENCHMARK_COLS,
    RANDOM_SEED,
    TEST_SIZE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_raw_data(filepath: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Load raw credit card churn dataset from CSV.
    """
    path = Path(filepath) if filepath else DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    logger.info("Loading dataset from: %s", path)
    df = pd.read_csv(path)
    logger.info("Loaded dataset successfully with shape: %s", df.shape)
    return df


def clean_data(df: pd.DataFrame, drop_target: bool = False) -> pd.DataFrame:
    """
    Clean dataset:
    - Drop identifier columns (e.g. CLIENTNUM)
    - Drop benchmark classifier columns (Naive Bayes outputs)
    - Handle whitespace in column names
    """
    cleaned_df = df.copy()

    # Strip whitespaces from column names
    cleaned_df.columns = [c.strip() for c in cleaned_df.columns]

    # Columns to drop
    cols_to_drop = []
    for col in ID_COLS + BENCHMARK_COLS:
        if col in cleaned_df.columns:
            cols_to_drop.append(col)

    # Any other pattern matching naive bayes
    nb_cols = [c for c in cleaned_df.columns if "naive_bayes" in c.lower()]
    for c in nb_cols:
        if c not in cols_to_drop:
            cols_to_drop.append(c)

    if drop_target and TARGET_COL in cleaned_df.columns:
        cols_to_drop.append(TARGET_COL)

    if cols_to_drop:
        cleaned_df = cleaned_df.drop(columns=cols_to_drop)
        logger.info("Dropped columns: %s", cols_to_drop)

    return cleaned_df


def prepare_features_and_target(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepares X (features) and y (binary encoded target).
    """
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataframe.")

    cleaned_df = clean_data(df)

    X = cleaned_df.drop(columns=[TARGET_COL])
    raw_y = cleaned_df[TARGET_COL]

    # Map target to 0 and 1
    if raw_y.dtype == object or str(raw_y.iloc[0]) in TARGET_MAPPING:
        y = raw_y.map(TARGET_MAPPING)
        if y.isna().any():
            unmapped = raw_y[y.isna()].unique()
            raise ValueError(f"Unrecognized target values found: {unmapped}")
    else:
        y = raw_y.astype(int)

    logger.info("Features shape: %s | Target distribution: %s", X.shape, dict(y.value_counts()))
    return X, y


def get_train_test_split(
    filepath: Optional[Union[str, Path]] = None,
    test_size: float = TEST_SIZE,
    random_seed: int = RANDOM_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Loads data and creates a stratified train-test split.
    """
    df = load_raw_data(filepath)
    X, y = prepare_features_and_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_seed,
        stratify=y
    )

    logger.info(
        "Stratified split completed -> Train: %s, Test: %s (Test Churn Rate: %.2f%%)",
        X_train.shape,
        X_test.shape,
        (y_test.sum() / len(y_test)) * 100
    )
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_train_test_split()
    print("Train features sample:")
    print(X_train.head(2))
