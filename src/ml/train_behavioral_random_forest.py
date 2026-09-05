import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# -----------------------------
# Configuration
# -----------------------------

BEHAVIORAL_DATA = "data/processed/transactions_behavioral_v2.csv"

TRAIN_SPLIT = "data/processed/splits/train.csv"
VALIDATION_SPLIT = "data/processed/splits/validation.csv"
TEST_SPLIT = "data/processed/splits/test.csv"

MODEL_PATH = "models/behavioral_random_forest_v2.joblib"

RANDOM_STATE = 42
THRESHOLD = 0.50

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

TARGET = "abuse_label"
ID_COLUMN = "transaction_id"


# -----------------------------
# Data loading
# -----------------------------

def load_split(path, behavioral_data):
    split = pd.read_csv(path)

    return pd.merge(
        split[[ID_COLUMN]],
        behavioral_data[
            [ID_COLUMN] + FEATURES + [TARGET]
        ],
        on=ID_COLUMN,
        how="left",
        validate="one_to_one",
    )


def validate_data(data):
    required_columns = [ID_COLUMN, TARGET] + FEATURES

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    if data[ID_COLUMN].duplicated().any():
        raise ValueError(
            "Duplicate transaction IDs found."
        )

    if data[FEATURES].isna().any().any():
        raise ValueError(
            "Missing values found in behavioral features."
        )

    if np.isinf(data[FEATURES].to_numpy()).any():
        raise ValueError(
            "Infinite values found in behavioral features."
        )


# -----------------------------
# Model evaluation
# -----------------------------

def evaluate_model(model, X, y, name):
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= THRESHOLD).astype(int)

    roc_auc = roc_auc_score(y, probabilities)
    pr_auc = average_precision_score(y, probabilities)

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

    f1 = f1_score(
        y,
        predictions,
        zero_division=0,
    )

    print(f"\n{name}")
    print("-" * 60)
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")

    print("\nConfusion Matrix")
    print(confusion_matrix(y, predictions))

    print("\nClassification Report")
    print(
        classification_report(
            y,
            predictions,
            zero_division=0,
        )
    )

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# -----------------------------
# Main experiment
# -----------------------------

def main():
    print("=" * 60)
    print("RAZORGUARD AI — V2 BEHAVIORAL RANDOM FOREST")
    print("=" * 60)

    # Load behavioral dataset
    behavioral_data = pd.read_csv(BEHAVIORAL_DATA)

    print(
        f"\nBehavioral dataset: "
        f"{len(behavioral_data):,} transactions"
    )

    validate_data(behavioral_data)

    # Reuse the existing customer-consistent splits
    train = load_split(TRAIN_SPLIT, behavioral_data)
    validation = load_split(
        VALIDATION_SPLIT,
        behavioral_data,
    )
    test = load_split(
        TEST_SPLIT,
        behavioral_data,
    )

    print("\nExisting splits")
    print("-" * 60)
    print(f"Training:   {len(train):,}")
    print(f"Validation: {len(validation):,}")
    print(f"Test:       {len(test):,}")

    # Make sure the merge didn't lose anything
    for name, data in [
        ("Training", train),
        ("Validation", validation),
        ("Test", test),
    ]:
        if data[FEATURES].isna().any().any():
            raise ValueError(
                f"{name} split contains missing features after merge."
            )

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_validation = validation[FEATURES]
    y_validation = validation[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]

    print("\nTraining model...")

    model = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    print("Training complete.")

    # Validation is used for model development
    validation_metrics = evaluate_model(
        model,
        X_validation,
        y_validation,
        "VALIDATION RESULTS",
    )

    # Feature importance
    importance = (
        pd.Series(
            model.feature_importances_,
            index=FEATURES,
        )
        .sort_values(ascending=False)
    )

    print("\nFeature Importance")
    print("-" * 60)

    for feature, score in importance.items():
        print(f"{feature:<35} {score:.6f}")

    # Test set remains untouched for final evaluation
    test_probabilities = model.predict_proba(X_test)[:, 1]

    test_roc_auc = roc_auc_score(
        y_test,
        test_probabilities,
    )

    test_pr_auc = average_precision_score(
        y_test,
        test_probabilities,
    )

    print("\nTEST RESULTS")
    print("-" * 60)
    print(f"ROC-AUC: {test_roc_auc:.4f}")
    print(f"PR-AUC:  {test_pr_auc:.4f}")

    # Save model and experiment metadata
    os.makedirs(
        os.path.dirname(MODEL_PATH),
        exist_ok=True,
    )

    model_package = {
        "model": model,
        "features": FEATURES,
        "target": TARGET,
        "threshold": THRESHOLD,
        "random_state": RANDOM_STATE,
        "validation_metrics": validation_metrics,
        "test_roc_auc": test_roc_auc,
        "test_pr_auc": test_pr_auc,
    }

    joblib.dump(
        model_package,
        MODEL_PATH,
    )

    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()