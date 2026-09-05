"""
RazorGuard AI
V2 Behavioral Logistic Regression

Purpose
-------
Train and evaluate a logistic regression model using
RazorGuard's behavioral intelligence features.

Experiment
----------
V1 = original transaction-level features
V2 = behavioral features

The purpose of this experiment is to measure whether
customer behavioral context provides additional signal
for detecting refund abuse.
"""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# ================================================================
# CONFIGURATION
# ================================================================

TRAIN_FILE = Path(
    "data/processed/splits/train.csv"
)

VALIDATION_FILE = Path(
    "data/processed/splits/validation.csv"
)

TEST_FILE = Path(
    "data/processed/splits/test.csv"
)

BEHAVIORAL_FILE = Path(
    "data/processed/transactions_behavioral_v2.csv"
)

MODEL_FILE = Path(
    "models/behavioral_logistic_v2.joblib"
)

TARGET = "abuse_label"

FEATURES = [
    "historical_orders",
    "historical_refunds_v2",
    "hours_since_previous_order",
    "hours_since_previous_refund",
    "customer_avg_order_amount",
    "amount_vs_customer_avg",
    "customer_avg_refund_amount",
    "refund_vs_customer_avg",
    "orders_last_24H",
    "orders_last_7D",
    "orders_last_30D",
    "refunds_last_24H",
    "refunds_last_7D",
    "refunds_last_30D",
    "refund_rate_7D",
    "refund_rate_30D",
    "refund_acceleration",
]


# ================================================================
# DATA PREPARATION
# ================================================================

def load_split(split_file: Path, behavioral: pd.DataFrame):
    """
    Recover the requested V1 customer split from the original
    split file and attach the V2 behavioral features.

    The split file provides the authoritative transaction IDs.
    """

    split = pd.read_csv(split_file)

    required = {
        "transaction_id",
        TARGET,
    }

    missing = required - set(split.columns)

    if missing:
        raise ValueError(
            f"{split_file} is missing columns: {sorted(missing)}"
        )

    merged = split[
        ["transaction_id", TARGET]
    ].merge(
        behavioral[
            ["transaction_id"] + FEATURES
        ],
        on="transaction_id",
        how="left",
        validate="one_to_one",
    )

    if merged[FEATURES].isna().any().any():
        missing_count = (
            merged[FEATURES].isna().sum().sum()
        )

        raise ValueError(
            f"{missing_count} behavioral feature values "
            f"could not be matched."
        )

    X = merged[FEATURES].copy()
    y = merged[TARGET].copy()

    return X, y


# ================================================================
# MODEL
# ================================================================

def build_model():
    """
    Build a production-style ML pipeline.

    StandardScaler:
        puts numerical features onto comparable scales.

    LogisticRegression:
        provides a strong, interpretable probability baseline.
    """

    return Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


# ================================================================
# EVALUATION
# ================================================================

def evaluate(model, X, y):
    """
    Evaluate probability ranking and classification
    performance on unseen validation data.
    """

    probabilities = model.predict_proba(X)[:, 1]

    predictions = (
        probabilities >= 0.50
    ).astype(int)

    roc_auc = roc_auc_score(
        y,
        probabilities,
    )

    pr_auc = average_precision_score(
        y,
        probabilities,
    )

    precision = precision_score(
        y,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0,
    )

    return {
        "probabilities": probabilities,
        "predictions": predictions,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": precision,
        "recall": recall,
    }


# ================================================================
# MAIN
# ================================================================

def main():

    print("=" * 70)
    print("RAZORGUARD AI — V2 BEHAVIORAL LOGISTIC REGRESSION")
    print("=" * 70)

    # ------------------------------------------------------------
    # Load behavioral dataset
    # ------------------------------------------------------------

    if not BEHAVIORAL_FILE.exists():
        raise FileNotFoundError(
            f"Behavioral dataset not found:\n"
            f"{BEHAVIORAL_FILE}"
        )

    behavioral = pd.read_csv(
        BEHAVIORAL_FILE
    )

    print("\nBEHAVIORAL DATA")
    print("-" * 70)

    print(
        f"Transactions: {len(behavioral):,}"
    )

    print(
        f"Behavioral features: {len(FEATURES)}"
    )

    # ------------------------------------------------------------
    # Recover identical customer-level splits
    # ------------------------------------------------------------

    print("\nLOADING EXISTING CUSTOMER SPLITS")
    print("-" * 70)

    X_train, y_train = load_split(
        TRAIN_FILE,
        behavioral,
    )

    X_validation, y_validation = load_split(
        VALIDATION_FILE,
        behavioral,
    )

    X_test, y_test = load_split(
        TEST_FILE,
        behavioral,
    )

    print(
        f"Training:   {len(X_train):,}"
    )

    print(
        f"Validation: {len(X_validation):,}"
    )

    print(
        f"Test:       {len(X_test):,}"
    )

    # ------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------

    print("\nTRAINING MODEL")
    print("-" * 70)

    model = build_model()

    model.fit(
        X_train,
        y_train,
    )

    print("Training complete.")

    # ------------------------------------------------------------
    # Validation evaluation
    # ------------------------------------------------------------

    results = evaluate(
        model,
        X_validation,
        y_validation,
    )

    print("\n" + "=" * 70)
    print("RAZORGUARD — V2 BEHAVIORAL MODEL")
    print("=" * 70)

    print("\nPROBABILITY-BASED METRICS")
    print("-" * 70)

    print(
        f"ROC-AUC: {results['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC:  {results['pr_auc']:.4f}"
    )

    print("\nTHRESHOLD = 0.50")
    print("-" * 70)

    print(
        f"Precision: "
        f"{results['precision']:.4f}"
    )

    print(
        f"Recall:    "
        f"{results['recall']:.4f}"
    )

    # ------------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------------

    print("\nCONFUSION MATRIX")
    print("-" * 70)

    print(
        confusion_matrix(
            y_validation,
            results["predictions"],
        )
    )

    # ------------------------------------------------------------
    # Classification report
    # ------------------------------------------------------------

    print("\nCLASSIFICATION REPORT")
    print("-" * 70)

    print(
        classification_report(
            y_validation,
            results["predictions"],
            digits=4,
            zero_division=0,
        )
    )

    # ------------------------------------------------------------
    # Test only after validation experiment
    # ------------------------------------------------------------

    print("\nTEST SET")
    print("-" * 70)

    test_results = evaluate(
        model,
        X_test,
        y_test,
    )

    print(
        f"ROC-AUC: {test_results['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC:  {test_results['pr_auc']:.4f}"
    )

    # ------------------------------------------------------------
    # Save model
    # ------------------------------------------------------------

    MODEL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_FILE,
    )

    print("\nMODEL SAVED")
    print("-" * 70)

    print(
        f"Path: {MODEL_FILE}"
    )

    print("\n" + "=" * 70)
    print("V2 BEHAVIORAL MODEL COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()