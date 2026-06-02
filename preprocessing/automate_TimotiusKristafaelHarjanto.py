"""
Automated Preprocessing Pipeline — Adult Income Dataset
Student : Timotius Kristafael Harjanto
Course  : Machine Learning System Production (SMSML) — Dicoding
"""

import os
import sys
import logging
import urllib.request
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("preprocessing.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RAW_DATA_URL: str = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
)
RAW_DATA_PATH: str = os.path.join(os.path.dirname(__file__), "..", "dataset_raw", "adult.csv")
PROCESSED_DIR: str = os.path.join(os.path.dirname(__file__), "dataset_preprocessed")

COLUMN_NAMES: list[str] = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income",
]

CATEGORICAL_COLUMNS: list[str] = [
    "workclass", "education", "marital_status", "occupation",
    "relationship", "race", "sex", "native_country",
]

NUMERICAL_COLUMNS: list[str] = [
    "age", "education_num", "capital_gain",
    "capital_loss", "hours_per_week",
]

TARGET_COLUMN: str = "income"


# ---------------------------------------------------------------------------
# Pipeline functions
# ---------------------------------------------------------------------------

def load_data(
    url: str = RAW_DATA_URL,
    save_path: str = RAW_DATA_PATH,
) -> pd.DataFrame:
    """Download (if needed) and load the Adult Income CSV dataset.

    Args:
        url:       Remote URL for the UCI Adult dataset.
        save_path: Local path where the raw file is cached.

    Returns:
        Raw DataFrame with labelled columns.
    """
    logger.info("=== load_data ===")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if not os.path.exists(save_path):
        logger.info("Downloading dataset from %s …", url)
        try:
            urllib.request.urlretrieve(url, save_path)
            logger.info("Saved raw dataset → %s", save_path)
        except Exception as exc:
            logger.error("Download failed: %s", exc)
            raise

    df = pd.read_csv(
        save_path,
        names=COLUMN_NAMES,
        sep=",",
        skipinitialspace=True,
        na_values="?",
    )
    logger.info("Dataset loaded — shape: %s", df.shape)
    logger.info("Columns: %s", df.columns.tolist())
    logger.info("Target distribution:\n%s", df[TARGET_COLUMN].value_counts())
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values, duplicates, and cap outliers.

    Args:
        df: Raw input DataFrame.

    Returns:
        Cleaned DataFrame.
    """
    logger.info("=== clean_data ===")
    initial_shape = df.shape

    # --- Missing values ---
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    logger.info("Missing values before cleaning:\n%s", missing_cols)
    df = df.dropna()
    logger.info("Rows after dropping NaN: %d", len(df))

    # --- Duplicates ---
    n_dup = df.duplicated().sum()
    if n_dup > 0:
        logger.info("Removing %d duplicate rows", n_dup)
        df = df.drop_duplicates()

    # --- Outlier capping (3×IQR rule) ---
    for col in NUMERICAL_COLUMNS:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
        n_out = ((df[col] < lo) | (df[col] > hi)).sum()
        if n_out:
            logger.info("Capping %d outliers in '%s' [%.2f, %.2f]", n_out, col, lo, hi)
        df[col] = df[col].clip(lower=lo, upper=hi)

    logger.info("clean_data complete — shape: %s → %s", initial_shape, df.shape)
    return df.reset_index(drop=True)


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Derive new features from existing columns.

    Args:
        df: Cleaned DataFrame.

    Returns:
        DataFrame enriched with engineered features.
    """
    logger.info("=== feature_engineering ===")

    # Age bucket
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 25, 35, 50, 65, 100],
        labels=["Young", "YoungAdult", "MiddleAge", "Senior", "Elderly"],
    )

    # Capital net gain
    df["capital_net"] = df["capital_gain"] - df["capital_loss"]
    df["has_capital_gain"] = (df["capital_gain"] > 0).astype(int)
    df["has_capital_loss"] = (df["capital_loss"] > 0).astype(int)

    # Higher education flag
    higher_ed = {
        "Bachelors", "Some-college", "Masters", "Doctorate", "Prof-school",
    }
    df["higher_education"] = df["education"].isin(higher_ed).astype(int)

    # Work-hours category
    df["work_hours_category"] = pd.cut(
        df["hours_per_week"],
        bins=[0, 35, 40, 50, 168],
        labels=["PartTime", "FullTime", "Overtime", "Heavy"],
    )

    # Married flag
    married_statuses = {"Married-civ-spouse", "Married-AF-spouse"}
    df["is_married"] = df["marital_status"].isin(married_statuses).astype(int)

    new_cols = [
        "age_group", "capital_net", "has_capital_gain",
        "has_capital_loss", "higher_education", "work_hours_category", "is_married",
    ]
    logger.info("New features added: %s", new_cols)
    logger.info("feature_engineering complete — shape: %s", df.shape)
    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode categorical columns and binarise the target.

    Args:
        df: DataFrame after feature engineering.

    Returns:
        Fully encoded DataFrame.
    """
    logger.info("=== encode_features ===")

    # Binarise target: 1 if ">50K", else 0
    df[TARGET_COLUMN] = df[TARGET_COLUMN].str.strip()
    df[TARGET_COLUMN] = (df[TARGET_COLUMN].str.rstrip(".") == ">50K").astype(int)
    logger.info("Target distribution after binarisation:\n%s", df[TARGET_COLUMN].value_counts())

    # Encode categorical + ordinal-category engineered columns
    cat_cols = CATEGORICAL_COLUMNS + ["age_group", "work_hours_category"]
    le = LabelEncoder()
    for col in cat_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))
            logger.info("Encoded: %s", col)

    logger.info("encode_features complete — shape: %s", df.shape)
    return df


def scale_features(
    df: pd.DataFrame,
    target: str = TARGET_COLUMN,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler]:
    """Split into train/test and apply StandardScaler to numerical columns.

    Args:
        df:     Encoded DataFrame.
        target: Name of the target column.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, fitted_scaler).
    """
    logger.info("=== scale_features ===")

    # Drop fnlwgt (census weight — not a predictive feature)
    drop_cols = [target, "fnlwgt"] if "fnlwgt" in df.columns else [target]
    X = df.drop(columns=drop_cols)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info("Train: %s | Test: %s", X_train.shape, X_test.shape)

    num_cols = [c for c in NUMERICAL_COLUMNS if c in X_train.columns]
    num_cols.append("capital_net")  # engineered numerical feature

    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    logger.info("scale_features complete — numerical columns scaled: %s", num_cols)
    return X_train, X_test, y_train, y_test, scaler


def save_processed_dataset(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    path: str = PROCESSED_DIR,
) -> None:
    """Persist train and test splits as CSV files.

    Args:
        X_train, X_test: Feature matrices.
        y_train, y_test: Target series.
        path:            Output directory.
    """
    logger.info("=== save_processed_dataset ===")
    os.makedirs(path, exist_ok=True)

    train_df = pd.concat(
        [X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1
    )
    test_df = pd.concat(
        [X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1
    )

    train_path = os.path.join(path, "train.csv")
    test_path = os.path.join(path, "test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    # Save feature list for downstream scripts
    feature_path = os.path.join(path, "feature_names.txt")
    with open(feature_path, "w") as f:
        f.write("\n".join(X_train.columns.tolist()))

    logger.info("train.csv saved  — shape: %s", train_df.shape)
    logger.info("test.csv  saved  — shape: %s", test_df.shape)
    logger.info("feature_names.txt saved")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Execute the full preprocessing pipeline end-to-end."""
    logger.info("=" * 60)
    logger.info("Adult Income Preprocessing Pipeline — START")
    logger.info("=" * 60)

    df = load_data()
    df = clean_data(df)
    df = feature_engineering(df)
    df = encode_features(df)
    X_train, X_test, y_train, y_test, _scaler = scale_features(df)
    save_processed_dataset(X_train, X_test, y_train, y_test)

    logger.info("=" * 60)
    logger.info("Pipeline completed successfully!")
    logger.info("  Training samples : %d", len(X_train))
    logger.info("  Test samples     : %d", len(X_test))
    logger.info("  Features         : %d", X_train.shape[1])
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
