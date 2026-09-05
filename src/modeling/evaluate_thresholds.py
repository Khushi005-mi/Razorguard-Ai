"""
RazorGuard AI
Threshold Analysis for Logistic Regression

Purpose
-------
Evaluate how RazorGuard's predictions behave at different
risk thresholds.

We are not changing the model.

We are changing the point at which a probability becomes
an "abuse" prediction.
"""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)


MODEL_PATH = Path(
    "models/logistic_regression_baseline.joblib"
)

THRESHOLDS = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
]


def load_validation_data():

    validation = pd.read_csv(
        "data/processed/splits/validation.csv"
    )

    features = [
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

    X = validation[features]
    y = validation["abuse_label"]

    return X, y


def main():

    print("=" * 80)
    print("RAZORGUARD AI — THRESHOLD ANALYSIS")
    print("=" * 80)

    model = joblib.load(MODEL_PATH)

    X_validation, y_validation = (
        load_validation_data()
    )

    probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    results = []

    for threshold in THRESHOLDS:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_validation,
            predictions
        ).ravel()

        precision = precision_score(
            y_validation,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_validation,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_validation,
            predictions,
            zero_division=0
        )

        results.append({
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "flagged_transactions": tp + fp,
        })

    results_df = pd.DataFrame(results)

    print("\n")
    print(
        results_df.to_string(
            index=False,
            formatters={
                "threshold": "{:.2f}".format,
                "precision": "{:.3f}".format,
                "recall": "{:.3f}".format,
                "f1_score": "{:.3f}".format,
            }
        )
    )

    print("\n" + "=" * 80)
    print("THRESHOLD ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()