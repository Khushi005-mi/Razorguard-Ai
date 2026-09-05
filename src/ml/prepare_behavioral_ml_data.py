"""
RazorGuard AI
V2 Behavioral ML Data Preparation

Purpose:
    Load the time-safe behavioral dataset and prepare
    the feature matrix (X) and target vector (y).

This module does NOT:
    - train a model
    - scale features
    - tune hyperparameters
    - perform train/test splitting

Those responsibilities belong to later stages of the ML pipeline.
"""

from pathlib import Path

import pandas as pd


# ================================================================
# CONFIGURATION
# ================================================================

DATA_FILE = Path(
    "data/processed/transactions_behavioral_v2.csv"
)

TARGET = "abuse_label"

BEHAVIORAL_FEATURES = [
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
# DATA LOADING
# ================================================================

def load_behavioral_data():
    """
    Load the behavioral dataset and return:

        X -> behavioral feature matrix
        y -> abuse target

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
    """

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"\nBehavioral dataset not found:\n"
            f"{DATA_FILE}\n"
        )

    df = pd.read_csv(DATA_FILE)

    required_columns = (
        ["transaction_id"]
        + BEHAVIORAL_FEATURES
        + [TARGET]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "\nMissing required columns:\n"
            + "\n".join(
                f"- {column}"
                for column in missing_columns
            )
        )

    X = df[BEHAVIORAL_FEATURES].copy()
    y = df[TARGET].copy()

    return df, X, y


# ================================================================
# MAIN
# ================================================================

def main():

    print("=" * 70)
    print("RAZORGUARD AI — V2 BEHAVIORAL ML DATA")
    print("=" * 70)

    # ------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------

    df, X, y = load_behavioral_data()

    # ------------------------------------------------------------
    # Dataset information
    # ------------------------------------------------------------

    print("\nDATASET")
    print("-" * 70)

    print(f"Samples:           {len(df):,}")
    print(f"Behavioral features: {len(BEHAVIORAL_FEATURES)}")

    # ------------------------------------------------------------
    # Feature list
    # ------------------------------------------------------------

    print("\nFEATURES")
    print("-" * 70)

    for number, feature in enumerate(
        BEHAVIORAL_FEATURES,
        start=1,
    ):
        print(f"{number:2}. {feature}")

    # ------------------------------------------------------------
    # Target
    # ------------------------------------------------------------

    print("\nTARGET")
    print("-" * 70)

    print(f"Target column: {TARGET}")

    # ------------------------------------------------------------
    # Target distribution
    # ------------------------------------------------------------

    print("\nTARGET DISTRIBUTION")
    print("-" * 70)

    target_counts = (
        y.value_counts()
        .sort_index()
    )

    print(target_counts.to_string())

    # ------------------------------------------------------------
    # Abuse rate
    # ------------------------------------------------------------

    abuse_rate = y.mean() * 100

    print("\nABUSE RATE")
    print("-" * 70)

    print(f"{abuse_rate:.2f}%")

    # ------------------------------------------------------------
    # Data quality
    # ------------------------------------------------------------

    print("\nDATA QUALITY")
    print("-" * 70)

    missing_values = X.isna().sum().sum()

    duplicate_features = X.duplicated().sum()

    duplicate_transactions = (
        df["transaction_id"]
        .duplicated()
        .sum()
    )

    infinite_values = (
        X.select_dtypes(include="number")
        .isin([float("inf"), float("-inf")])
        .sum()
        .sum()
    )

    print(
        f"Missing feature values:      "
        f"{missing_values:,}"
    )

    print(
        f"Duplicate feature vectors:   "
        f"{duplicate_features:,}"
    )

    print(
        f"Duplicate transaction IDs:   "
        f"{duplicate_transactions:,}"
    )

    print(
        f"Infinite feature values:     "
        f"{infinite_values:,}"
    )

    # ------------------------------------------------------------
    # Final validation
    # ------------------------------------------------------------

    if missing_values > 0:
        raise ValueError(
            "Dataset contains missing feature values."
        )

    if infinite_values > 0:
        raise ValueError(
            "Dataset contains infinite feature values."
        )

    if duplicate_transactions > 0:
        raise ValueError(
            "Duplicate transaction IDs detected."
        )

    # ------------------------------------------------------------
    # Completion
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("V2 ML DATA PREPARATION COMPLETE")
    print("=" * 70)


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":
    main()