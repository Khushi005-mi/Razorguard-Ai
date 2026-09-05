import pandas as pd


FILE = "data/processed/transactions_ml.csv"


def validate():

    df = pd.read_csv(FILE)

    print("================================")
    print("RAZORGUARD DATASET VALIDATION")
    print("================================")

    # Shape
    print("\nDataset shape:")
    print(df.shape)

    # Missing values
    print("\nMissing values:")
    print(
        df.isnull().sum()
        .sum()
    )

    # Duplicate IDs
    print("\nDuplicate transactions:")
    print(
        df["transaction_id"]
        .duplicated()
        .sum()
    )

    # Negative amounts
    print("\nNegative transaction amounts:")
    print(
        (df["amount"] < 0).sum()
    )

    # Labels
    print("\nAbuse distribution:")
    print(
        df["abuse_label"]
        .value_counts()
    )

    print("\nAbuse percentage:")
    print(
        df["abuse_label"]
        .mean() * 100
    )

    # Refund validation
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
        "\nInvalid refund records:"
    )

    print(
        invalid_refunds
    )


if __name__ == "__main__":
    validate()