"""
RazorGuard AI - ML Data Preparation

Loads the leakage-safe train, validation, and test datasets
created during Day 1 and prepares the feature matrices and
target vectors required by the modeling pipeline.

This module intentionally contains NO model-training logic.

Pipeline:

    Raw / Processed Data
            ↓
    Customer-level split
            ↓
    This module
            ↓
    X_train, y_train
    X_validation, y_validation
    X_test, y_test
            ↓
    Model Training
"""


from pathlib import Path
from typing import Tuple

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("data/processed/splits")

TRAIN_FILE = DATA_DIR / "train.csv"
VALIDATION_FILE = DATA_DIR / "validation.csv"
TEST_FILE = DATA_DIR / "test.csv"


# Features available to the model at prediction time.
#
# IMPORTANT:
# We deliberately exclude:
#   - transaction_id
#   - customer_id
#   - customer_segment
#   - abuse_probability
#
# The last two are hidden variables from our synthetic
# environment and must never reach the model.
FEATURE_COLUMNS = [
    "amount",
    "refund_requested",
    "refund_amount",
    "refund_delay_hours",
    "previous_orders",
    "previous_refunds",
    "historical_refund_rate",
    "high_value_transaction",
    "rapid_refund",
    "refund_amount_ratio",
    "refund_frequency_signal",
]

TARGET_COLUMN = "abuse_label"


# ============================================================
# TYPE ALIAS
# ============================================================

Dataset = Tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
]


# ============================================================
# DATA LOADING
# ============================================================

def load_split(file_path: Path) -> pd.DataFrame:
    """
    Load one dataset split from disk.

    Parameters
    ----------
    file_path:
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.

    Raises
    ------
    FileNotFoundError
        If the requested dataset does not exist.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}\n"
            "Run the Day 1 dataset generation and split "
            "pipeline before preparing the ML data."
        )

    return pd.read_csv(file_path)


# ============================================================
# COLUMN VALIDATION
# ============================================================

def validate_columns(df: pd.DataFrame, dataset_name: str) -> None:
    """
    Verify that the dataset contains every column required
    by the modeling pipeline.

    Parameters
    ----------
    df:
        Dataset to validate.

    dataset_name:
        Human-readable name used in error messages.
    """

    required_columns = set(
        FEATURE_COLUMNS + [TARGET_COLUMN]
    )

    missing_columns = (
        required_columns - set(df.columns)
    )

    if missing_columns:
        missing = ", ".join(
            sorted(missing_columns)
        )

        raise ValueError(
            f"{dataset_name} is missing required columns: "
            f"{missing}"
        )


# ============================================================
# FEATURE / TARGET SEPARATION
# ============================================================

def split_features_and_target(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate model features (X) from the prediction target (y).

    X contains only information that RazorGuard could use
    when making a prediction.

    y contains the ground-truth abuse label.
    """

    X = df[FEATURE_COLUMNS].copy()

    y = df[TARGET_COLUMN].copy()

    return X, y


# DATASET LOADING PIPELINE


def load_modeling_data() -> Dataset:
    """
    Load and prepare train, validation, and test datasets.

    Returns
    -------
    Dataset
        X_train, y_train,
        X_validation, y_validation,
        X_test, y_test
    """

    train_df = load_split(TRAIN_FILE)
    validation_df = load_split(VALIDATION_FILE)
    test_df = load_split(TEST_FILE)

    validate_columns(
        train_df,
        "Training dataset",
    )

    validate_columns(
        validation_df,
        "Validation dataset",
    )

    validate_columns(
        test_df,
        "Test dataset",
    )

    X_train, y_train = split_features_and_target(
        train_df
    )

    X_validation, y_validation = split_features_and_target(
        validation_df
    )

    X_test, y_test = split_features_and_target(
        test_df
    )

    return (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
    )


# ============================================================
# DATASET SUMMARY
# ============================================================

def print_dataset_summary(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """
    Print a concise summary of the datasets entering
    the modeling pipeline.
    """

    print("\n")
    print("RAZORGUARD AI — ML DATA PREPARATION")
    

    print("\nDATASET SHAPES")
    print("-" * 60)

    print(
        f"Training:   X={X_train.shape}, "
        f"y={y_train.shape}"
    )

    print(
        f"Validation: X={X_validation.shape}, "
        f"y={y_validation.shape}"
    )

    print(
        f"Test:       X={X_test.shape}, "
        f"y={y_test.shape}"
    )

    print("\nFEATURES")
    print("-" * 60)

    for index, feature in enumerate(
        FEATURE_COLUMNS,
        start=1,
    ):
        print(
            f"{index:2}. {feature}"
        )

    print("\nTARGET")
    print("-" * 60)

    print(
        f"Target column: {TARGET_COLUMN}"
    )

    print("\nABUSE RATE")
    print("-" * 60)

    train_rate = y_train.mean() * 100
    validation_rate = y_validation.mean() * 100
    test_rate = y_test.mean() * 100

    print(
        f"Training:   {train_rate:.2f}%"
    )

    print(
        f"Validation: {validation_rate:.2f}%"
    )

    print(
        f"Test:       {test_rate:.2f}%"
    )

    print("\nTARGET COUNTS")
    print("-" * 60)

    print(
        "Training:"
    )
    print(
        y_train.value_counts()
        .sort_index()
        .to_string()
    )

    print(
        "\nValidation:"
    )
    print(
        y_validation.value_counts()
        .sort_index()
        .to_string()
    )

    print(
        "\nTest:"
    )
    print(
        y_test.value_counts()
        .sort_index()
        .to_string()
    )

    print("\n" + "=" * 60)
    print("ML DATA PREPARATION COMPLETE")
    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """
    Entry point for the ML data preparation pipeline.
    """

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
    ) = load_modeling_data()

    print_dataset_summary(
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
    )


if __name__ == "__main__":
    main()