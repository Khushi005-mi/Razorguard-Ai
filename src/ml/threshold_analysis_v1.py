from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


MODEL_PATH = Path("models/logistic_regression_baseline.joblib")
SPLIT_PATH = Path("data/processed/splits/validation.csv")

FEATURES = [
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

TARGET = "abuse_label"

THRESHOLDS = np.arange(0.10, 0.91, 0.05)

# Initial business assumptions.
# These are deliberately explicit so they can later be replaced
# with merchant-specific economics.

FALSE_POSITIVE_COST = 100
FALSE_NEGATIVE_COST = 1000


def load_validation_data():
    data = pd.read_csv(SPLIT_PATH)

    required = FEATURES + [TARGET]

    missing = [column for column in required if column not in data.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    return data


def calculate_metrics(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0,
    )

    cost = (
        fp * FALSE_POSITIVE_COST
        + fn * FALSE_NEGATIVE_COST
    )

    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "total_cost": cost,
    }


def main():
    print("=" * 70)
    print("RAZORGUARD AI — THRESHOLD ECONOMICS V1")
    print("=" * 70)

    validation = load_validation_data()

    model = joblib.load(MODEL_PATH)

    X = validation[FEATURES]
    y = validation[TARGET]

    probabilities = model.predict_proba(X)[:, 1]

    results = []

    for threshold in THRESHOLDS:
        metrics = calculate_metrics(
            y,
            probabilities,
            threshold,
        )

        results.append(metrics)

    results_df = pd.DataFrame(results)

    best = results_df.loc[
        results_df["total_cost"].idxmin()
    ]

    print("\nBusiness assumptions")
    print("-" * 70)
    print(f"False-positive cost: ₹{FALSE_POSITIVE_COST:,}")
    print(f"False-negative cost: ₹{FALSE_NEGATIVE_COST:,}")

    print("\nThreshold analysis")
    print("-" * 70)

    print(
        results_df[
            [
                "threshold",
                "precision",
                "recall",
                "f1",
                "true_positives",
                "false_positives",
                "false_negatives",
                "total_cost",
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("LOWEST ESTIMATED COST")
    print("=" * 70)

    print(f"Threshold:       {best['threshold']:.2f}")
    print(f"Precision:       {best['precision']:.4f}")
    print(f"Recall:          {best['recall']:.4f}")
    print(f"F1:              {best['f1']:.4f}")
    print(f"False positives: {int(best['false_positives'])}")
    print(f"False negatives: {int(best['false_negatives'])}")
    print(f"Estimated cost:  ₹{best['total_cost']:,.0f}")

    output = Path("data/processed/threshold_analysis_v1.csv")
    results_df.to_csv(output, index=False)

    print(f"\nResults saved to: {output}")


if __name__ == "__main__":
    main()