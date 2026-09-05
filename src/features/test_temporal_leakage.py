"""
RazorGuard AI
Temporal Leakage Test

Purpose:
Verify that behavioral features for transaction T
do not contain information from transactions occurring
after T.
"""

from pathlib import Path
import pytest
import pandas as pd


INPUT_FILE = Path(
    "data/processed/transactions_behavioral_v2.csv"
)


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
]


def load_data():
    """Load and chronologically sort the behavioral dataset."""

    df = pd.read_csv(INPUT_FILE)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    return df.sort_values(
        ["customer_id", "timestamp"]
    ).reset_index(drop=True)
@pytest.fixture
def df():
    return load_data()

def test_historical_orders(df):
    """
    Historical order count must equal the number of
    transactions occurring before the current transaction.
    """

    expected = (
        df.groupby("customer_id")
        .cumcount()
    )

    mismatches = (
        df["historical_orders"]
        != expected
    ).sum()

    assert mismatches == 0, (
    f"Found {mismatches} historical order count mismatches"
)


def test_first_transaction_history(df):
    """
    The first transaction of every customer must have
    zero historical activity.
    """

    first_transactions = (
        df.groupby("customer_id")
        .head(1)
    )

    columns = [
    "historical_orders",
    "historical_refunds_v2",
    "orders_last_24H",
    "orders_last_7D",
    "orders_last_30D",
    "refunds_last_24H",
    "refunds_last_7D",
    "refunds_last_30D",
]
    violations = (
        first_transactions[columns]
        != 0
    ).any(axis=1).sum()

    assert violations == 0, (
    f"Found {violations} first-transaction history violations"
)


def test_monotonic_history(df):
    """
    Historical order count must never decrease for a customer.
    """

    violations = 0

    for _, group in df.groupby("customer_id"):

        history = group[
            "historical_orders"
        ]

        if not history.is_monotonic_increasing:
            violations += 1

    assert violations == 0, (
    f"Found {violations} customers with non-monotonic history"
)


def test_future_information(df):
    """
    For each transaction, behavioral features must not
    depend on future transaction labels.

    This test checks whether the behavioral feature columns
    accidentally contain the target itself.
    """

    leakage_columns = [
        feature
        for feature in BEHAVIORAL_FEATURES
        if feature == "abuse_label"
    ]

    assert len(leakage_columns) == 0, (
    f"Target leakage columns detected: {leakage_columns}"
)


def test_missing_and_infinite(df):
    """Check for NaN and infinite behavioral values."""

    behavioral = df[
        BEHAVIORAL_FEATURES
    ]

    missing = behavioral.isna().sum().sum()

    numeric = behavioral.select_dtypes(
        include="number"
    )

    infinite = (
        numeric
        .isin([float("inf"), float("-inf")])
        .sum()
        .sum()
    )

    assert missing == 0, (
    f"Found {missing} missing behavioral feature values"
)

    assert infinite == 0, (
    f"Found {infinite} infinite behavioral feature values"
)


def main():

    print("=" * 70)
    print("RAZORGUARD AI — TEMPORAL LEAKAGE TEST")
    print("=" * 70)

    df = load_data()

    print(
        f"\nTransactions tested: {len(df):,}"
    )

    print(
        f"Customers tested: "
        f"{df['customer_id'].nunique():,}"
    )

    # ------------------------------------------------------------
    # TEST 1
    # ------------------------------------------------------------

    historical_mismatches = (
        test_historical_orders(df)
    )

    print("\n1. HISTORICAL ORDER COUNT")
    print("-" * 70)

    print(
        f"Mismatches: {historical_mismatches}"
    )

    # ------------------------------------------------------------
    # TEST 2
    # ------------------------------------------------------------

    first_transaction_violations = (
        test_first_transaction_history(df)
    )

    print("\n2. FIRST TRANSACTION HISTORY")
    print("-" * 70)

    print(
        f"Violations: "
        f"{first_transaction_violations}"
    )

    # ------------------------------------------------------------
    # TEST 3
    # ------------------------------------------------------------

    monotonic_violations = (
        test_monotonic_history(df)
    )

    print("\n3. HISTORICAL ORDER MONOTONICITY")
    print("-" * 70)

    print(
        f"Customers with violations: "
        f"{monotonic_violations}"
    )

    # ------------------------------------------------------------
    # TEST 4
    # ------------------------------------------------------------

    leakage_columns = (
        test_future_information(df)
    )

    print("\n4. TARGET LEAKAGE CHECK")
    print("-" * 70)

    print(
        f"Target-containing behavioral features: "
        f"{leakage_columns}"
    )

    # ------------------------------------------------------------
    # TEST 5
    # ------------------------------------------------------------

    missing, infinite = (
        test_missing_and_infinite(df)
    )

    print("\n5. DATA QUALITY")
    print("-" * 70)

    print(f"Missing values:   {missing}")
    print(f"Infinite values:  {infinite}")

    # ------------------------------------------------------------
    # FINAL RESULT
    # ------------------------------------------------------------

    passed = (
        historical_mismatches == 0
        and first_transaction_violations == 0
        and monotonic_violations == 0
        and leakage_columns == 0
        and missing == 0
        and infinite == 0
    )

    print("\n" + "=" * 70)

    if passed:
        print("TEMPORAL LEAKAGE TEST: PASSED")
        print("=" * 70)
        print(
            "\nBehavioral features passed all automated checks."
        )
    else:
        print("TEMPORAL LEAKAGE TEST: FAILED")
        print("=" * 70)
        print(
            "\nDo NOT train the next model."
        )


if __name__ == "__main__":
    main()