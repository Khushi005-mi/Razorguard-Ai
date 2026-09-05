"""
RazorGuard AI
Behavioral Feature Engine V2

Purpose
-------
Transform raw transaction history into time-safe customer
behavioral features.

Core principle
--------------
For transaction T:

    Features(T) may use information from transactions BEFORE T.

    Features(T) must NEVER use information from transactions AFTER T.

This prevents temporal data leakage.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ================================================================
# CONFIGURATION
# ================================================================

INPUT_FILE = Path(
    "data/processed/transactions_ml.csv"
)

OUTPUT_FILE = Path(
    "data/processed/transactions_behavioral_v2.csv"
)


# ================================================================
# HELPER FUNCTIONS
# ================================================================

def safe_divide(numerator, denominator):
    """
    Divide two values safely.

    If denominator is zero, return 0 instead of producing
    infinity or NaN.
    """

    return np.where(
        denominator > 0,
        numerator / denominator,
        0.0,
    )


# ================================================================
# LOAD AND PREPARE DATA
# ================================================================

def load_data():
    """
    Load transactions and prepare them for chronological
    behavioral feature generation.
    """

    df = pd.read_csv(INPUT_FILE)

    # Convert timestamp from string to datetime.
    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    # Chronological ordering is critical.
    #
    # Customer behavior must be calculated in time order.
    df = df.sort_values(
        ["customer_id", "timestamp"]
    ).reset_index(drop=True)

    return df


# ================================================================
# BUILD BEHAVIORAL FEATURES
# ================================================================

def build_features(df):
    """
    Build customer behavioral features using only historical
    information available before each transaction.
    """

    result = df.copy()

    # ------------------------------------------------------------
    # BASIC CUSTOMER HISTORY
    # ------------------------------------------------------------

    # Number of transactions BEFORE current transaction.
    result["historical_orders"] = (
        result.groupby("customer_id")
        .cumcount()
    )

    # Historical refunds BEFORE current transaction.
    #
    # refund_requested is the current transaction's refund status.
    # shift(1) ensures today's transaction is NOT included.
    historical_refunds = (
        result.groupby("customer_id")[
            "refund_requested"
        ]
        .transform(
            lambda x: x.shift(1).fillna(0).cumsum()
        )
    )

    result["historical_refunds_v2"] = (
        historical_refunds
    )

    # ------------------------------------------------------------
    # TIME SINCE PREVIOUS TRANSACTION
    # ------------------------------------------------------------

    previous_timestamp = (
        result.groupby("customer_id")[
            "timestamp"
        ]
        .shift(1)
    )

    result["hours_since_previous_order"] = (
        result["timestamp"] - previous_timestamp
    ).dt.total_seconds() / 3600

    result["hours_since_previous_order"] = (
        result["hours_since_previous_order"]
        .fillna(0)
    )

    # ------------------------------------------------------------
    # PREVIOUS REFUND TIMESTAMP
    # ------------------------------------------------------------

    refund_timestamp = result["timestamp"].where(
        result["refund_requested"] == 1
    )

    previous_refund_timestamp = (
        refund_timestamp
        .groupby(result["customer_id"])
        .ffill()
        .groupby(result["customer_id"])
        .shift(1)
    )

    result["hours_since_previous_refund"] = (
        result["timestamp"]
        - previous_refund_timestamp
    ).dt.total_seconds() / 3600

    result["hours_since_previous_refund"] = (
        result["hours_since_previous_refund"]
        .fillna(0)
    )

    # ------------------------------------------------------------
    # CUSTOMER HISTORICAL AVERAGE ORDER AMOUNT
    # ------------------------------------------------------------

    cumulative_amount = (
        result.groupby("customer_id")["amount"]
        .transform(
            lambda x: x.shift(1).expanding().mean()
        )
    )

    result["customer_avg_order_amount"] = (
        cumulative_amount.fillna(0)
    )

    # Current amount relative to customer's historical average.
    result["amount_vs_customer_avg"] = safe_divide(
        result["amount"],
        result["customer_avg_order_amount"],
    )

    # ------------------------------------------------------------
    # CUSTOMER HISTORICAL AVERAGE REFUND
    # ------------------------------------------------------------

    historical_refund_amount = (
        result["refund_amount"]
        .where(
            result["refund_requested"] == 1,
            0.0,
        )
    )

    customer_avg_refund = (
        historical_refund_amount
        .groupby(result["customer_id"])
        .transform(
            lambda x: x.shift(1).expanding().mean()
        )
    )

    result["customer_avg_refund_amount"] = (
        customer_avg_refund.fillna(0)
    )

    result["refund_vs_customer_avg"] = safe_divide(
        result["refund_amount"],
        result["customer_avg_refund_amount"],
    )

    # ------------------------------------------------------------
    # RECENT ACTIVITY WINDOWS
    # ------------------------------------------------------------

    # We calculate rolling windows independently for each customer.
    #
    # shift(1) is crucial:
    #
    # current transaction
    #        ↓
    # exclude current transaction
    #        ↓
    # calculate historical activity
    #

    def calculate_recent_activity(group):

        timestamps = group["timestamp"]

        order_indicator = pd.Series(
            1,
            index=group.index,
        )

        refund_indicator = group[
            "refund_requested"
        ]

        order_series = pd.Series(
            order_indicator.values,
            index=timestamps,
        )

        refund_series = pd.Series(
            refund_indicator.values,
            index=timestamps,
        )

        historical_orders = (
           order_series
           .shift(1)
           .rolling("24h")
           .sum()
)

        historical_orders_7d = (
         order_series
          .shift(1)
          .rolling("7D")
          .sum()
)

        historical_orders_30d = (
         order_series
          .shift(1)
          .rolling("30D")
          .sum()
)

        historical_refunds = (
           refund_series
           .shift(1)
           .rolling("24h")
           .sum()
)

        historical_refunds_7d = (
        refund_series
         .shift(1)
         .rolling("7D")
          .sum()
)

        historical_refunds_30d = (
          refund_series
          .shift(1)
          .rolling("30D")
          .sum()
)

        return pd.DataFrame(
            {
                "orders_last_24H": historical_orders.values,
                "orders_last_7D": historical_orders_7d.values,
                "orders_last_30D": historical_orders_30d.values,
                "refunds_last_24H": historical_refunds.values,
                "refunds_last_7D": historical_refunds_7d.values,
                "refunds_last_30D": historical_refunds_30d.values,
            },
            index=group.index,
        )

    recent_features = (
        result
        .groupby("customer_id", group_keys=False)
        .apply(
            calculate_recent_activity,
            include_groups=False,
        )
    )

    # Align results with original dataframe.
    recent_features = recent_features.reset_index(
        level=0,
        drop=True,
    )

    for column in recent_features.columns:
        result[column] = (
            recent_features[column]
            .reindex(result.index)
            .fillna(0)
        )

    # ------------------------------------------------------------
    # RECENT REFUND RATES
    # ------------------------------------------------------------

    result["refund_rate_7D"] = safe_divide(
        result["refunds_last_7D"],
        result["orders_last_7D"],
    )

    result["refund_rate_30D"] = safe_divide(
        result["refunds_last_30D"],
        result["orders_last_30D"],
    )

    # ------------------------------------------------------------
    # REFUND ACCELERATION
    # ------------------------------------------------------------

    result["refund_acceleration"] = safe_divide(
        result["refund_rate_7D"],
        result["refund_rate_30D"],
    )

    # ------------------------------------------------------------
    # CLEAN NUMERIC OUTPUT
    # ------------------------------------------------------------

    numeric_columns = result.select_dtypes(
        include=np.number
    ).columns

    result[numeric_columns] = (
        result[numeric_columns]
        .replace(
            [np.inf, -np.inf],
            0,
        )
        .fillna(0)
    )

    return result


# ================================================================
# VALIDATION
# ================================================================

def validate_features(df):
    """
    Perform basic validation on the generated feature set.
    """

    print("\n")
    print("=" * 70)
    print("RAZORGUARD — BEHAVIORAL FEATURE VALIDATION")
    print("=" * 70)

    print("\nDataset shape:")
    print(df.shape)

    print("\nNew behavioral features:")
    print("-" * 70)

    new_features = [
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

    for feature in new_features:
        print(
            f"{feature:35s}"
            f"{df[feature].dtype}"
        )

    print("\nMissing values:")
    print(
        df[new_features]
        .isna()
        .sum()
        .sum()
    )

    print("\nInfinite values:")

    infinite_count = np.isinf(
        df[new_features]
        .select_dtypes(include=np.number)
    ).sum().sum()

    print(infinite_count)

    print("\nBehavioral feature statistics:")
    print(
        df[new_features]
        .describe()
        .T[
            [
                "mean",
                "std",
                "min",
                "max",
            ]
        ]
    )


# ================================================================
# MAIN
# ================================================================

def main():

    print("=" * 70)
    print("RAZORGUARD AI — BEHAVIORAL FEATURE ENGINE V2")
    print("=" * 70)

    print("\nLoading transaction data...")

    df = load_data()

    print(
        f"Loaded {len(df):,} transactions."
    )

    print("\nBuilding time-safe behavioral features...")

    df = build_features(df)

    print("Feature engineering complete.")

    validate_features(df)

    # ------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\n")
    print("=" * 70)
    print("BEHAVIORAL FEATURE DATASET SAVED")
    print("=" * 70)

    print(
        f"Path: {OUTPUT_FILE}"
    )

    print("\n")
    print("RAZORGUARD FEATURE ENGINE V2 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()