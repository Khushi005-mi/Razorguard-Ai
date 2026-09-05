import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


DATA_PATH = Path("data/processed/transactions_behavioral_v2.csv")
SPLIT_DIR = Path("data/processed/splits")

TARGET = "abuse_label"

V1_FEATURES = [
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

HISTORY_FEATURES = [
    "historical_orders",
    "historical_refunds_v2",
    "hours_since_previous_order",
    "hours_since_previous_refund",
]

VELOCITY_FEATURES = [
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

CUSTOMER_RELATIVE_FEATURES = [
    "customer_avg_order_amount",
    "amount_vs_customer_avg",
    "customer_avg_refund_amount",
    "refund_vs_customer_avg",
]

EXPERIMENTS = {
    "V1 baseline": V1_FEATURES,

    "History only": V1_FEATURES + HISTORY_FEATURES,

    "Velocity only": V1_FEATURES + VELOCITY_FEATURES,

    "Customer-relative only": V1_FEATURES + CUSTOMER_RELATIVE_FEATURES,

    "V1 + strongest behavioral": V1_FEATURES + [
        "historical_refunds_v2",
        "refund_rate_7D",
        "refund_rate_30D",
        "refunds_last_7D",
        "refunds_last_30D",
        "refund_acceleration",
    ],
}


def load_split(name, data):
    split = pd.read_csv(SPLIT_DIR / f"{name}.csv")

    if "transaction_id" not in split.columns:
        raise ValueError(f"{name} split does not contain transaction_id")

    split_ids = set(split["transaction_id"])

    result = data[data["transaction_id"].isin(split_ids)].copy()

    if len(result) != len(split):
        raise ValueError(
            f"{name} split mismatch: expected {len(split)}, got {len(result)}"
        )

    if result["transaction_id"].duplicated().any():
        raise ValueError(f"Duplicate transaction IDs found in {name}")

    return result


def validate_data(data):
    required = set(V1_FEATURES + HISTORY_FEATURES + VELOCITY_FEATURES +
                   CUSTOMER_RELATIVE_FEATURES + [TARGET, "transaction_id"])

    missing = required - set(data.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if data["transaction_id"].duplicated().any():
        raise ValueError("Duplicate transaction IDs detected")

    if data[TARGET].isna().any():
        raise ValueError("Missing target values detected")


def train_and_evaluate(train, validation, features):
    X_train = train[features]
    y_train = train[TARGET]

    X_val = validation[features]
    y_val = validation[TARGET]

    model = Pipeline([
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=42,
            ),
        ),
    ])

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_val)[:, 1]

    roc_auc = roc_auc_score(y_val, probabilities)
    pr_auc = average_precision_score(y_val, probabilities)

    return roc_auc, pr_auc


def main():
    print("=" * 60)
    print("RAZORGUARD AI — FEATURE ABLATION EXPERIMENT")
    print("=" * 60)

    data = pd.read_csv(DATA_PATH)

    print(f"\nBehavioral dataset: {len(data):,} transactions")

    validate_data(data)

    train = load_split("train", data)
    validation = load_split("validation", data)
    test = load_split("test", data)

    print("\nExisting splits")
    print("-" * 60)
    print(f"Training:   {len(train):,}")
    print(f"Validation: {len(validation):,}")
    print(f"Test:       {len(test):,}")

    results = []

    print("\nRunning experiments...")

    for name, features in EXPERIMENTS.items():
        roc_auc, pr_auc = train_and_evaluate(
            train,
            validation,
            features,
        )

        results.append({
            "experiment": name,
            "features": len(features),
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
        })

        print(
            f"{name:<30} "
            f"ROC-AUC: {roc_auc:.4f}   "
            f"PR-AUC: {pr_auc:.4f}"
        )

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        "pr_auc",
        ascending=False,
    )

    print("\n" + "=" * 60)
    print("RESULTS — RANKED BY VALIDATION PR-AUC")
    print("=" * 60)

    print(
        results_df[
            ["experiment", "features", "roc_auc", "pr_auc"]
        ].to_string(index=False)
    )

    output_path = Path("data/processed/feature_ablation_results.csv")
    results_df.to_csv(output_path, index=False)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()