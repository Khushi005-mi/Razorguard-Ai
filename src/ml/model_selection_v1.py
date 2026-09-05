from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SPLIT_DIR = Path("data/processed/splits")

TARGET = "abuse_label"

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


def load_split(name):
    return pd.read_csv(SPLIT_DIR / f"{name}.csv")


def evaluate(model, train, validation):
    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_val = validation[FEATURES]
    y_val = validation[TARGET]

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_val)[:, 1]
    predictions = (probabilities >= 0.50).astype(int)

    return {
        "roc_auc": roc_auc_score(y_val, probabilities),
        "pr_auc": average_precision_score(y_val, probabilities),
        "precision": precision_score(y_val, predictions, zero_division=0),
        "recall": recall_score(y_val, predictions, zero_division=0),
        "f1": f1_score(y_val, predictions, zero_division=0),
    }


def main():
    print("=" * 60)
    print("RAZORGUARD AI — MODEL SELECTION V1")
    print("=" * 60)

    train = load_split("train")
    validation = load_split("validation")
    test = load_split("test")

    print(f"\nTraining:   {len(train):,}")
    print(f"Validation: {len(validation):,}")
    print(f"Test:       {len(test):,}")

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),

        "Hist Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=42,
        ),
    }

    results = []

    print("\nValidation experiments")
    print("-" * 60)

    for name, model in models.items():
        metrics = evaluate(model, train, validation)

        results.append({
            "model": name,
            **metrics,
        })

        print(
            f"{name:<25}"
            f" PR-AUC: {metrics['pr_auc']:.4f}"
            f" | ROC-AUC: {metrics['roc_auc']:.4f}"
            f" | Precision: {metrics['precision']:.4f}"
            f" | Recall: {metrics['recall']:.4f}"
            f" | F1: {metrics['f1']:.4f}"
        )

    results_df = pd.DataFrame(results).sort_values(
        "pr_auc",
        ascending=False,
    )

    print("\n" + "=" * 60)
    print("MODEL RANKING")
    print("=" * 60)

    print(results_df.to_string(index=False))

    output = Path("data/processed/model_selection_v1.csv")
    results_df.to_csv(output, index=False)

    print(f"\nResults saved to: {output}")


if __name__ == "__main__":
    main()