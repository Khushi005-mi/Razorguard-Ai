"""
RazorGuard AI
Behavioral Feature Statistical Audit

Purpose
-------
Determine whether RazorGuard's new behavioral features contain
meaningful statistical signal for distinguishing abusive and
legitimate transactions.

This is NOT model training.

We are answering:

    "Do these engineered behavioral features actually behave
     differently for abuse vs legitimate transactions?"

If a feature has no meaningful separation, we should question
whether it belongs in the production feature set.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ================================================================
# CONFIGURATION
# ================================================================

INPUT_FILE = Path(
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
# LOAD DATA
# ================================================================

def load_data():
    """Load the V2 behavioral dataset."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    return df


# ================================================================
# VALIDATE INPUT
# ================================================================

def validate_input(df):
    """Verify that the expected columns exist."""

    required_columns = (
        BEHAVIORAL_FEATURES + [TARGET]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(
                f"- {column}"
                for column in missing_columns
            )
        )

    invalid_targets = set(
        df[TARGET].dropna().unique()
    ) - {0, 1}

    if invalid_targets:
        raise ValueError(
            f"Unexpected target values: "
            f"{invalid_targets}"
        )


# ================================================================
# BASIC DATA QUALITY
# ================================================================

def audit_data_quality(df):

    print("\n1. DATA QUALITY")
    print("-" * 70)

    missing = (
        df[BEHAVIORAL_FEATURES]
        .isna()
        .sum()
        .sum()
    )

    numeric_features = (
        df[BEHAVIORAL_FEATURES]
        .select_dtypes(include=np.number)
    )

    infinite = (
        np.isinf(numeric_features)
        .sum()
        .sum()
    )

    print(
        f"Missing behavioral values:   {missing}"
    )

    print(
        f"Infinite behavioral values:  {infinite}"
    )

    print(
        f"Rows:                         {len(df):,}"
    )

    print(
        f"Behavioral features:          "
        f"{len(BEHAVIORAL_FEATURES)}"
    )

    return missing, infinite


# ================================================================
# TARGET DISTRIBUTION
# ================================================================

def audit_target(df):

    print("\n2. TARGET DISTRIBUTION")
    print("-" * 70)

    counts = df[TARGET].value_counts().sort_index()

    legitimate = counts.get(0, 0)
    abuse = counts.get(1, 0)

    total = legitimate + abuse

    abuse_rate = (
        abuse / total * 100
        if total > 0
        else 0
    )

    print(
        f"Legitimate transactions: {legitimate:,}"
    )

    print(
        f"Abusive transactions:     {abuse:,}"
    )

    print(
        f"Abuse rate:               {abuse_rate:.2f}%"
    )


# ================================================================
# FEATURE STATISTICS
# ================================================================

def calculate_statistics(df):

    legitimate = df[
        df[TARGET] == 0
    ]

    abusive = df[
        df[TARGET] == 1
    ]

    rows = []

    for feature in BEHAVIORAL_FEATURES:

        normal_values = (
            legitimate[feature]
        )

        abuse_values = (
            abusive[feature]
        )

        normal_mean = normal_values.mean()
        abuse_mean = abuse_values.mean()

        normal_median = normal_values.median()
        abuse_median = abuse_values.median()

        normal_std = normal_values.std()
        abuse_std = abuse_values.std()

        mean_difference = (
            abuse_mean - normal_mean
        )

        absolute_difference = abs(
            mean_difference
        )

        # --------------------------------------------------------
        # Standardized mean difference
        #
        # This gives us a rough scale-independent measure
        # of separation between the two populations.
        # --------------------------------------------------------

        pooled_std = np.sqrt(
            (
                normal_std ** 2
                +
                abuse_std ** 2
            ) / 2
        )

        if pooled_std > 0:
            standardized_difference = (
                mean_difference / pooled_std
            )
        else:
            standardized_difference = 0.0

        # --------------------------------------------------------
        # Relative shift
        #
        # Useful when values are positive and the normal mean
        # isn't zero.
        # --------------------------------------------------------

        if normal_mean != 0:
            relative_shift = (
                abuse_mean - normal_mean
            ) / abs(normal_mean)
        else:
            relative_shift = np.nan

        rows.append(
            {
                "feature": feature,
                "normal_mean": normal_mean,
                "abuse_mean": abuse_mean,
                "normal_median": normal_median,
                "abuse_median": abuse_median,
                "normal_std": normal_std,
                "abuse_std": abuse_std,
                "mean_difference": mean_difference,
                "absolute_difference": absolute_difference,
                "standardized_difference": standardized_difference,
                "relative_shift": relative_shift,
            }
        )

    return pd.DataFrame(rows)


# ================================================================
# PRINT STATISTICAL COMPARISON
# ================================================================

def print_statistics(statistics):

    print("\n3. BEHAVIORAL FEATURE SEPARATION")
    print("-" * 70)

    display_columns = [
        "feature",
        "normal_mean",
        "abuse_mean",
        "normal_median",
        "abuse_median",
        "standardized_difference",
    ]

    display = statistics[
        display_columns
    ].copy()

    numeric_columns = [
        column
        for column in display.columns
        if column != "feature"
    ]

    display[numeric_columns] = (
        display[numeric_columns]
        .round(4)
    )

    print(
        display.to_string(
            index=False
        )
    )


# ================================================================
# RANK FEATURES
# ================================================================

def rank_features(statistics):

    ranked = statistics.copy()

    ranked["signal_strength"] = (
        ranked["standardized_difference"]
        .abs()
    )

    ranked = ranked.sort_values(
        "signal_strength",
        ascending=False,
    )

    return ranked


def print_rankings(ranked):

    print("\n4. FEATURE SIGNAL RANKING")
    print("-" * 70)

    print(
        f"{'Rank':<6}"
        f"{'Feature':<35}"
        f"{'Signal Strength':>16}"
    )

    print("-" * 70)

    for rank, (_, row) in enumerate(
        ranked.iterrows(),
        start=1,
    ):

        print(
            f"{rank:<6}"
            f"{row['feature']:<35}"
            f"{row['signal_strength']:>16.4f}"
        )


# ================================================================
# INTERPRETATION
# ================================================================

def print_interpretation(ranked):

    print("\n5. PRELIMINARY INTERPRETATION")
    print("-" * 70)

    top_features = ranked.head(5)

    print(
        "\nTop behavioral signals:"
    )

    for _, row in top_features.iterrows():

        direction = (
            "higher in abuse"
            if row["standardized_difference"] > 0
            else "lower in abuse"
        )

        print(
            f"- {row['feature']}: "
            f"{direction}, "
            f"signal={row['signal_strength']:.4f}"
        )

    print(
        "\nImportant:"
    )

    print(
        "Statistical separation does NOT prove causation "
        "and does NOT guarantee predictive performance."
    )

    print(
        "The next step is model-based validation on "
        "unseen validation data."
    )


# ================================================================
# SAVE AUDIT
# ================================================================

def save_audit(statistics, ranked):

    output_file = Path(
        "data/processed/"
        "behavioral_feature_audit.csv"
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ranked.to_csv(
        output_file,
        index=False,
    )

    print(
        f"\nAudit saved to: {output_file}"
    )


# ================================================================
# MAIN
# ================================================================

def main():

    print("=" * 70)
    print(
        "RAZORGUARD AI — "
        "BEHAVIORAL FEATURE STATISTICAL AUDIT"
    )
    print("=" * 70)

    print(
        "\nLoading behavioral dataset..."
    )

    df = load_data()

    print(
        f"Loaded {len(df):,} transactions."
    )

    validate_input(df)

    missing, infinite = (
        audit_data_quality(df)
    )

    audit_target(df)

    if missing > 0 or infinite > 0:

        raise ValueError(
            "\nData quality gate failed. "
            "Fix missing/infinite values before continuing."
        )

    print(
        "\nCalculating feature statistics..."
    )

    statistics = calculate_statistics(df)

    print_statistics(statistics)

    ranked = rank_features(statistics)

    print_rankings(ranked)

    print_interpretation(ranked)

    save_audit(
        statistics,
        ranked,
    )

    print("\n")
    print("=" * 70)
    print(
        "BEHAVIORAL FEATURE AUDIT COMPLETE"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()