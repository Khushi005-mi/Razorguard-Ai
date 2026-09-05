import pandas as pd
import numpy as np


FILE = "data/processed/transactions_ml.csv"


def audit():

    print("========================================")
    print("RAZORGUARD DATASET DEEP AUDIT")
    print("========================================")

    df = pd.read_csv(FILE)

    print("\n1. BASIC INFORMATION")
    print("----------------------------------------")

    print("Shape:", df.shape)

    print("\nColumns:")
    for column in df.columns:
        print("-", column)

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    print("\n2. MISSING VALUES")
    print("----------------------------------------")

    missing = df.isnull().sum()

    print(missing)

    assert missing.sum() == 0, \
        "Missing values detected."

    # --------------------------------------------------------
    # Duplicate IDs
    # --------------------------------------------------------

    print("\n3. DUPLICATE TRANSACTIONS")
    print("----------------------------------------")

    duplicates = df["transaction_id"].duplicated().sum()

    print("Duplicate transaction IDs:", duplicates)

    assert duplicates == 0, \
        "Duplicate transactions detected."

    # --------------------------------------------------------
    # Amount validation
    # --------------------------------------------------------

    print("\n4. AMOUNT VALIDATION")
    print("----------------------------------------")

    negative_amounts = (
        df["amount"] < 0
    ).sum()

    print(
        "Negative amounts:",
        negative_amounts
    )

    assert negative_amounts == 0

    print(
        "\nAmount statistics:"
    )

    print(
        df["amount"].describe()
    )

    # --------------------------------------------------------
    # Refund validation
    # --------------------------------------------------------

    print("\n5. REFUND VALIDATION")
    print("----------------------------------------")

    invalid_refunds = (
        (
            df["refund_requested"] == 0
        )
        &
        (
            df["refund_amount"] > 0
        )
    ).sum()

    print(
        "Invalid refund records:",
        invalid_refunds
    )

    assert invalid_refunds == 0

    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    print("\n6. TARGET DISTRIBUTION")
    print("----------------------------------------")

    counts = df["abuse_label"].value_counts()

    print(counts)

    abuse_rate = (
        df["abuse_label"].mean()
        * 100
    )

    print(
        f"\nAbuse rate: {abuse_rate:.2f}%"
    )

    # --------------------------------------------------------
    # High-value feature
    # --------------------------------------------------------

    print("\n7. HIGH-VALUE TRANSACTION AUDIT")
    print("----------------------------------------")

    high_value_rate = (
        df["high_value_transaction"].mean()
        * 100
    )

    print(
        f"High-value transactions: "
        f"{high_value_rate:.2f}%"
    )

    assert high_value_rate > 0, \
        "High-value feature is inactive."

    # --------------------------------------------------------
    # Feature separation
    # --------------------------------------------------------

    print("\n8. FEATURE SEPARATION")
    print("----------------------------------------")

    numeric_features = [
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
        "refund_frequency_signal"
    ]

    print(
        df.groupby(
            "abuse_label"
        )[numeric_features]
        .mean()
        .T
    )

    # --------------------------------------------------------
    # Customer-level distribution
    # --------------------------------------------------------

    print("\n9. CUSTOMER ACTIVITY")
    print("----------------------------------------")

    customer_counts = (
        df.groupby("customer_id")
        .size()
    )

    print(
        customer_counts.describe()
    )

    print(
        "\nCustomers represented:",
        df["customer_id"].nunique()
    )

    # --------------------------------------------------------
    # Abuse by customer
    # --------------------------------------------------------

    customer_abuse = (
        df.groupby("customer_id")["abuse_label"]
        .mean()
    )

    print(
        "\nCustomer abuse-rate statistics:"
    )

    print(
        customer_abuse.describe()
    )

    # --------------------------------------------------------
    # Temporal coverage
    # --------------------------------------------------------

    print("\n10. TEMPORAL COVERAGE")
    print("----------------------------------------")

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    print(
        "Start:",
        df["timestamp"].min()
    )

    print(
        "End:",
        df["timestamp"].max()
    )

    print(
        "Unique days:",
        df["timestamp"].dt.date.nunique()
    )

    # --------------------------------------------------------
    # Future leakage sanity check
    # --------------------------------------------------------

    print("\n11. LEAKAGE SANITY CHECK")
    print("----------------------------------------")

    forbidden_columns = [
        "customer_segment",
        "abuse_probability"
    ]

    leakage_columns = [
        col
        for col in forbidden_columns
        if col in df.columns
    ]

    print(
        "Hidden target-related columns:",
        leakage_columns
    )

    assert len(leakage_columns) == 0, \
        "Potential target leakage detected."

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print("\n========================================")
    print("DEEP AUDIT COMPLETE")
    print("========================================")

    print(
        "Dataset is structurally ready "
        "for train/test split."
    )


if __name__ == "__main__":
    audit()