import joblib
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


MODEL_PATH = "models/logistic_regression_baseline.joblib"
TEST_DATA_PATH = "data/processed/splits/test.csv"
THRESHOLD = 0.65

FP_COST = 100
FN_COST = 1000


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


def evaluate_model():
    model = joblib.load(MODEL_PATH)

    test_df = pd.read_csv(TEST_DATA_PATH)

    X_test = test_df[FEATURES]
    test_df = pd.read_csv(TEST_DATA_PATH)

    X_test = test_df[FEATURES]
    y_test = test_df["abuse_label"]

    probabilities = model.predict_proba(X_test)[:, 1]

    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities,
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
    ).ravel()

    total_cost = (
        fp * FP_COST
        + fn * FN_COST
    )

    print("\n========== RAZORGUARD MODEL EVALUATION ==========\n")

    print(f"Test transactions : {len(test_df)}")
    print(f"Abusive cases     : {int(y_test.sum())}")
    print(f"Threshold         : {THRESHOLD}")

    print("\n--- Model Performance ---")

    print(f"PR-AUC            : {pr_auc:.4f}")
    print(f"ROC-AUC           : {roc_auc:.4f}")
    print(f"Precision          : {precision:.4f}")
    print(f"Recall             : {recall:.4f}")
    print(f"F1 Score           : {f1:.4f}")

    print("\n--- Confusion Matrix ---")

    print(f"True Negatives     : {tn}")
    print(f"False Positives    : {fp}")
    print(f"False Negatives    : {fn}")
    print(f"True Positives     : {tp}")

    print("\n--- Economic Impact ---")

    print(f"False-positive cost: ₹{fp * FP_COST:,}")
    print(f"False-negative cost: ₹{fn * FN_COST:,}")
    print(f"Total estimated cost: ₹{total_cost:,}")

    print("\n===================================================\n")


if __name__ == "__main__":
    evaluate_model()