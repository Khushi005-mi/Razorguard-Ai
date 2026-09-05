import pandas as pd
from pathlib import Path


INPUT = "data/processed/transactions_ml.csv"

OUTPUT_DIR = Path(
    "data/processed/splits"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def create_splits():

    df = pd.read_csv(INPUT)

    # Unique customers
    customers = (
        df["customer_id"]
        .drop_duplicates()
        .sample(
            frac=1,
            random_state=42
        )
        .reset_index(drop=True)
    )

    n = len(customers)

    train_end = int(
        n * 0.70
    )

    validation_end = int(
        n * 0.85
    )

    train_customers = set(
        customers.iloc[:train_end]
    )

    validation_customers = set(
        customers.iloc[
            train_end:validation_end
        ]
    )

    test_customers = set(
        customers.iloc[
            validation_end:
        ]
    )

    train = df[
        df["customer_id"]
        .isin(train_customers)
    ]

    validation = df[
        df["customer_id"]
        .isin(validation_customers)
    ]

    test = df[
        df["customer_id"]
        .isin(test_customers)
    ]

    train.to_csv(
        OUTPUT_DIR / "train.csv",
        index=False
    )

    validation.to_csv(
        OUTPUT_DIR / "validation.csv",
        index=False
    )

    test.to_csv(
        OUTPUT_DIR / "test.csv",
        index=False
    )

    print("================================")
    print("DATASET SPLIT COMPLETE")
    print("================================")

    print(
        f"Train customers: {len(train_customers):,}"
    )

    print(
        f"Validation customers: "
        f"{len(validation_customers):,}"
    )

    print(
        f"Test customers: {len(test_customers):,}"
    )

    print("\nTransactions:")

    print(
        "Train:",
        len(train)
    )

    print(
        "Validation:",
        len(validation)
    )

    print(
        "Test:",
        len(test)
    )

    # Verify no customer overlap
    train_ids = set(train["customer_id"])
    validation_ids = set(validation["customer_id"])
    test_ids = set(test["customer_id"])

    assert train_ids.isdisjoint(validation_ids), \
    "Customer leakage detected between train and validation."

    assert train_ids.isdisjoint(test_ids), \
    "Customer leakage detected between train and test."

    assert validation_ids.isdisjoint(test_ids), \
    "Customer leakage detected between validation and test."

print("\nCustomer leakage check: PASS")


if __name__ == "__main__":
    create_splits()