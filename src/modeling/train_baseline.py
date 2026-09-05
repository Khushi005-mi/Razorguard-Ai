"""
RazorGuard AI - Logistic Regression Baseline

Purpose
-------
Train the first interpretable classification model for
detecting refund abuse.

This model establishes the performance baseline against
which more complex models will be compared.

Pipeline
--------
Prepared ML data
      ↓
Feature scaling
      ↓
Logistic Regression
      ↓
Abuse probability
      ↓
Validation metrics
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from prepare_data import load_modeling_data


# CONFIGURATION

MODEL_DIR = Path("models")

MODEL_PATH = MODEL_DIR / "logistic_regression_baseline.joblib"


RANDOM_STATE = 42


# MODEL CREATION

def build_model() -> Pipeline:
    """
    Create the Logistic Regression pipeline.

    StandardScaler:
        Puts numerical features on comparable scales.

    LogisticRegression:
        Estimates the probability that a transaction
        belongs to the abuse class.
    """

    model = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    max_iter=1000,
                ),
            ),
        ]
    )

    return model


# ============================================================
# MODEL TRAINING
# ============================================================

def train_model(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """
    Train the model using only the training dataset.
    """

    print("\nTraining Logistic Regression...")

    model.fit(
        X_train,
        y_train,
    )

    print("Training complete.")

    return model


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(
    model: Pipeline,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> None:
    """
    Evaluate the model on unseen validation data.

    We deliberately focus on metrics suitable for an
    imbalanced abuse-detection problem.
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

    print("\n" + "=" * 60)
    print("RAZORGUARD — LOGISTIC REGRESSION BASELINE")
    print("=" * 60)

    print("\nPROBABILITY-BASED METRICS")
    print("-" * 60)

    print(
        f"ROC-AUC: {roc_auc:.4f}"
    )

    print(
        f"PR-AUC:  {pr_auc:.4f}"
    )

    print("\nTHRESHOLD = 0.50")
    print("-" * 60)

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall:    {recall:.4f}"
    )

    print("\nCONFUSION MATRIX")
    print("-" * 60)

    matrix = confusion_matrix(
        y_validation,
        predictions,
    )

    print(matrix)

    print("\nCLASSIFICATION REPORT")
    print("-" * 60)

    print(
        classification_report(
            y_validation,
            predictions,
            digits=4,
            zero_division=0,
        )
    )


# ============================================================
# MODEL PERSISTENCE
# ============================================================

def save_model(
    model: Pipeline,
) -> None:
    """
    Save the trained model so that it can later be loaded
    by the RazorGuard inference service.
    """

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    print("\nMODEL SAVED")
    print("-" * 60)
    print(
        f"Path: {MODEL_PATH}"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """
    Complete baseline training pipeline.
    """

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        _X_test,
        _y_test,
    ) = load_modeling_data()

    model = build_model()

    model = train_model(
        model,
        X_train,
        y_train,
    )

    evaluate_model(
        model,
        X_validation,
        y_validation,
    )

    save_model(model)


if __name__ == "__main__":
    main()