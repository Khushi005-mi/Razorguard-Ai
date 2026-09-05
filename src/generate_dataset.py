import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

SEED = 42

N_CUSTOMERS = 10_000
N_TRANSACTIONS = 50_000

START_DATE = datetime(2026, 1, 1)

np.random.seed(SEED)


# ============================================================
# DIRECTORIES
# ============================================================

Path("data/raw").mkdir(parents=True, exist_ok=True)
Path("data/processed").mkdir(parents=True, exist_ok=True)


# ============================================================
# CUSTOMER PROFILES
# ============================================================

def create_customer_profiles(n_customers):

    segments = np.random.choice(
        [
            "normal",
            "occasional_refunder",
            "suspicious",
            "aggressive_abuser"
        ],
        size=n_customers,
        p=[0.855, 0.075, 0.052, 0.018]
    )

    customers = []

    for i, segment in enumerate(segments):

        customer_id = f"CUST_{i + 1:05d}"

        # Behavioral tendency.
        # This is hidden from the ML model.
        if segment == "normal":

            base_refund_rate = np.random.beta(2, 35)

        elif segment == "occasional_refunder":

            base_refund_rate = np.random.beta(3, 15)

        elif segment == "suspicious":

            base_refund_rate = np.random.beta(5, 8)

        else:

            base_refund_rate = np.random.beta(8, 5)

        customers.append({
            "customer_id": customer_id,
            "segment": segment,
            "base_refund_rate": base_refund_rate
        })

    return pd.DataFrame(customers)


# ============================================================
# TRANSACTION GENERATION
# ============================================================

def generate_transactions(customers, n_transactions):

    transactions = []

    customer_ids = customers["customer_id"].values

    for i in range(n_transactions):

        customer_id = np.random.choice(customer_ids)

        customer = customers[
            customers["customer_id"] == customer_id
        ].iloc[0]

        segment = customer["segment"]
        base_refund_rate = customer["base_refund_rate"]

        transaction_id = f"TXN_{i + 1:07d}"

        timestamp = (
            START_DATE
            + timedelta(
                days=int(np.random.randint(0, 180)),
                hours=int(np.random.randint(0, 24)),
                minutes=int(np.random.randint(0, 60))
            )
        )

        # ----------------------------------------------------
        # Transaction amount
        # ----------------------------------------------------

        amount = np.random.lognormal(
            mean=7.4,
            sigma=1.15
        )

        # Explicit high-value tail.
        if np.random.random() < 0.03:

            amount *= np.random.uniform(
                5,
                15
            )

        amount = round(
            np.clip(
                amount,
                100,
                500_000
            ),
            2
        )

        # ----------------------------------------------------
        # Payment method
        # ----------------------------------------------------

        payment_method = np.random.choice(
            [
                "upi",
                "card",
                "netbanking",
                "wallet"
            ],
            p=[
                0.45,
                0.35,
                0.15,
                0.05
            ]
        )

        # ----------------------------------------------------
        # Refund probability
        # ----------------------------------------------------

        # Behavioral tendency + randomness.
        refund_probability = np.clip(
            base_refund_rate
            + np.random.normal(0, 0.03),
            0.005,
            0.90
        )

        refund_requested = int(
            np.random.random()
            < refund_probability
        )

        # ----------------------------------------------------
        # Refund timing
        # ----------------------------------------------------

        if refund_requested:

            if segment == "normal":

                delay = np.random.gamma(
                    shape=3.5,
                    scale=24
                )

            elif segment == "occasional_refunder":

                delay = np.random.gamma(
                    shape=3,
                    scale=20
                )

            elif segment == "suspicious":

                delay = np.random.gamma(
                    shape=2,
                    scale=12
                )

            else:

                delay = np.random.gamma(
                    shape=1.8,
                    scale=8
                )

            # Add behavioral overlap.
            if np.random.random() < 0.15:

                delay = np.random.uniform(
                    1,
                    120
                )

            refund_delay_hours = round(
                np.clip(delay, 0.5, 720),
                2
            )

            # Most refunds are partial/full.
            refund_ratio = np.random.uniform(
                0.50,
                1.00
            )

            refund_amount = round(
                amount * refund_ratio,
                2
            )

        else:

            refund_amount = 0
            refund_delay_hours = 0

        transactions.append({
            "transaction_id": transaction_id,
            "customer_id": customer_id,
            "timestamp": timestamp,
            "amount": amount,
            "payment_method": payment_method,
            "refund_requested": refund_requested,
            "refund_amount": refund_amount,
            "refund_delay_hours": refund_delay_hours,
            "customer_segment": segment
        })

    return pd.DataFrame(transactions)


# ============================================================
# TEMPORAL BEHAVIORAL FEATURES
# ============================================================

def create_behavioral_features(df):

    df = df.sort_values(
        ["customer_id", "timestamp"]
    ).copy()

    # --------------------------------------------------------
    # Historical transaction count
    # --------------------------------------------------------

    df["previous_orders"] = (
        df.groupby("customer_id")
        .cumcount()
    )

    # --------------------------------------------------------
    # Historical refunds
    # --------------------------------------------------------

    df["previous_refunds"] = (
        df.groupby("customer_id")["refund_requested"]
        .cumsum()
        - df["refund_requested"]
    )

    # --------------------------------------------------------
    # Historical refund rate
    # --------------------------------------------------------

    df["historical_refund_rate"] = np.where(
        df["previous_orders"] > 0,
        df["previous_refunds"]
        / df["previous_orders"],
        0
    )

    # --------------------------------------------------------
    # Transaction value
    # --------------------------------------------------------

    df["high_value_transaction"] = (
        df["amount"] >= 25_000
    ).astype(int)

    # --------------------------------------------------------
    # Rapid refund
    # --------------------------------------------------------

    df["rapid_refund"] = (
        (df["refund_requested"] == 1)
        &
        (df["refund_delay_hours"] <= 6)
    ).astype(int)

    # --------------------------------------------------------
    # Refund ratio
    # --------------------------------------------------------

    df["refund_amount_ratio"] = np.where(
        df["amount"] > 0,
        df["refund_amount"] / df["amount"],
        0
    )

    # --------------------------------------------------------
    # Behavioral interaction
    # --------------------------------------------------------

    df["refund_frequency_signal"] = (
        df["previous_refunds"]
        *
        df["historical_refund_rate"]
    )

    return df


# ============================================================
# GROUND TRUTH
# ============================================================

def create_ground_truth(df):

    df = df.copy()

    score = np.zeros(len(df))

    # Historical behavior
    score += (
        df["historical_refund_rate"]
        * 3.0
    )

    # Previous refunds
    score += np.minimum(
        df["previous_refunds"],
        8
    ) * 0.25

    # Rapid refund
    score += (
        df["rapid_refund"]
        * 0.9
    )

    # High refund ratio
    score += np.where(
        df["refund_amount_ratio"] > 0.80,
        0.6,
        0
    )

    # High-value transaction contributes modestly.
    score += (
        df["high_value_transaction"]
        * 0.25
    )

    # --------------------------------------------------------
    # Convert behavioral score into probability.
    # --------------------------------------------------------

    probability = 1 / (
        1 + np.exp(
            -(score - 2.2)
        )
    )

    # Make abuse uncommon.
    probability *= 0.35

    # Add unobserved factors.
    probability += np.random.normal(
        0,
        0.025,
        len(df)
    )

    probability = np.clip(
        probability,
        0,
        1
    )

    df["abuse_probability"] = probability

    df["abuse_label"] = (
        np.random.random(len(df))
        < probability
    ).astype(int)

    return df


# ============================================================
# MAIN
# ============================================================

def main():

    print("Creating customer profiles...")

    customers = create_customer_profiles(
        N_CUSTOMERS
    )

    print("Generating transactions...")

    transactions = generate_transactions(
        customers,
        N_TRANSACTIONS
    )

    print("Creating behavioral features...")

    transactions = create_behavioral_features(
        transactions
    )

    print("Creating ground-truth labels...")

    transactions = create_ground_truth(
        transactions
    )

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    transactions = (
        transactions
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Save raw customer data
    # --------------------------------------------------------

    customers.to_csv(
        "data/raw/customers.csv",
        index=False
    )

    # --------------------------------------------------------
    # Raw transaction data
    # --------------------------------------------------------

    raw_columns = [
        "transaction_id",
        "customer_id",
        "timestamp",
        "amount",
        "payment_method",
        "refund_requested",
        "refund_amount",
        "refund_delay_hours",
        "customer_segment"
    ]

    transactions[
        raw_columns
    ].to_csv(
        "data/raw/transactions_raw.csv",
        index=False
    )

    # --------------------------------------------------------
    # Full ground truth dataset
    # --------------------------------------------------------

    transactions.to_csv(
        "data/raw/transactions_ground_truth.csv",
        index=False
    )

    # --------------------------------------------------------
    # ML dataset
    #
    # IMPORTANT:
    # customer_segment and abuse_probability are hidden
    # variables and therefore removed.
    # --------------------------------------------------------

    ml_df = transactions.drop(
        columns=[
            "customer_segment",
            "abuse_probability"
        ]
    )

    ml_df.to_csv(
        "data/processed/transactions_ml.csv",
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n================================")
    print("DATASET GENERATION COMPLETE")
    print("================================")

    print(
        f"Customers: {len(customers):,}"
    )

    print(
        f"Transactions: {len(transactions):,}"
    )

    print("\nCustomer segments:")

    print(
        customers["segment"]
        .value_counts(
            normalize=True
        )
        .mul(100)
        .round(2)
    )

    print("\nAbuse labels:")

    print(
        ml_df["abuse_label"]
        .value_counts()
    )

    print("\nAbuse percentage:")

    print(
        round(
            ml_df["abuse_label"].mean() * 100,
            2
        ),
        "%"
    )

    print("\nHigh-value transaction percentage:")

    print(
        round(
            ml_df["high_value_transaction"].mean() * 100,
            2
        ),
        "%"
    )


if __name__ == "__main__":
    main()