"""
RazorGuard AI
Random Forest Baseline

Purpose
-------
Train a nonlinear tree-based model to detect refund abuse.

Why Random Forest?
------------------
Logistic Regression assumes a relatively simple relationship
between features and abuse risk.

Refund abuse can involve behavioral interactions such as:

    high refund frequency
    + previous refunds
    + rapid refund
    + high refund amount ratio

Random Forest can learn these nonlinear relationships.

Important rule
--------------
The test set is NOT used during model development.

We use:

    Training data   -> model learning
    Validation data -> model evaluation
    Test data       -> final untouched evaluation
"""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------

TRAIN_FILE = Path(
    "data/processed/splits/train.csv"
)

VALIDATION_FILE = Path(
    "data/processed/splits/validation.csv"
)

MODEL_FILE = Path(
    "models/random_forest_baseline.joblib"
)


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


# -------------------------------------------------------------------
# DATA LOADING
# -------------------------------------------------------------------

def load_data():
    """
    Load training and validation datasets.

    Returns
    -------
    X_train
    y_train
    X_validation
    y_validation
    """

    train = pd.read_csv(TRAIN_FILE)
    validation = pd.read_csv(VALIDATION_FILE)

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_validation = validation[FEATURES]
    y_validation = validation[TARGET]

    return (
        X_train,
        y_train,
        X_validation,
        y_validation,
    )


# -------------------------------------------------------------------
# MODEL CREATION
# -------------------------------------------------------------------

def create_model():
    """
    Create the Random Forest classifier.

    class_weight='balanced'
    -----------------------
    Abuse is a minority class.

    Without class weighting, the model can become overly
    focused on legitimate transactions.

    random_state=42
    ----------------
    Makes the experiment reproducible.
    """

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    return model


# -------------------------------------------------------------------
# EVALUATION
# -------------------------------------------------------------------

def evaluate_model(
    model,
    X_validation,
    y_validation,
):
    """
    Evaluate the model on validation data.

    We evaluate probability-based metrics first because
    RazorGuard ultimately needs a risk score, not merely
    a binary prediction.
    """

    probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    predictions = (
        probabilities >= 0.50
    ).astype(int)

    roc_auc = roc_auc_score(
        y_validation,
        probabilities,
    )

    pr_auc = average_precision_score(
        y_validation,
        probabilities,
    )

    precision = precision_score(
        y_validation,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_validation,
        predictions,
        zero_division=0,
    )

    confusion = confusion_matrix(
        y_validation,
        predictions,
    )

    print("\n")
    print("=" * 70)
    print("RAZORGUARD — RANDOM FOREST BASELINE")
    print("=" * 70)

    print("\nPROBABILITY-BASED METRICS")
    print("-" * 70)

    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC:  {pr_auc:.4f}")

    print("\nTHRESHOLD = 0.50")
    print("-" * 70)

    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")

    print("\nCONFUSION MATRIX")
    print("-" * 70)

    print(confusion)

    print("\nCLASSIFICATION REPORT")
    print("-" * 70)

    print(
        classification_report(
            y_validation,
            predictions,
            digits=4,
            zero_division=0,
        )
    )

    return probabilities


# -------------------------------------------------------------------
# FEATURE IMPORTANCE
# -------------------------------------------------------------------

def show_feature_importance(model):
    """
    Show which features the Random Forest used most heavily.

    This helps us understand which behavioral signals
    contribute most strongly to the model.
    """

    importance = pd.DataFrame(
        {
            "feature": FEATURES,
            "importance": model.feature_importances_,
        }
    )

    importance = importance.sort_values(
        by="importance",
        ascending=False,
    )

    print("\nFEATURE IMPORTANCE")
    print("-" * 70)

    for _, row in importance.iterrows():

        print(
            f"{row['feature']:30s}"
            f"{row['importance']:.6f}"
        )


# -------------------------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------------------------

def main():

    print("=" * 70)
    print("RAZORGUARD AI — RANDOM FOREST TRAINING")
    print("=" * 70)

    print("\nLoading data...")

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
    ) = load_data()

    print("\nDATA")
    print("-" * 70)

    print(
        f"Training samples:   {len(X_train):,}"
    )

    print(
        f"Validation samples: {len(X_validation):,}"
    )

    print(
        f"Number of features: {len(FEATURES)}"
    )

    print("\nTraining Random Forest...")

    model = create_model()

    model.fit(
        X_train,
        y_train,
    )

    print("Training complete.")

    # Evaluate validation performance
    probabilities = evaluate_model(
        model,
        X_validation,
        y_validation,
    )

    # Explain model behavior
    show_feature_importance(model)

    # Save trained model
    MODEL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_FILE,
    )

    print("\n")
    print("=" * 70)
    print("MODEL SAVED")
    print("=" * 70)

    print(f"Path: {MODEL_FILE}")

    print("\nRANDOM FOREST BASELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()