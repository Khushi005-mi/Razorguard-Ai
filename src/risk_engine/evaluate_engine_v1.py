import pandas as pd

from risk_engine.engine import RiskEngine
from audit.logger import AuditLogger

VALIDATION_PATH = "data/processed/splits/validation.csv"


def main():

    df = pd.read_csv(VALIDATION_PATH)

    engine = RiskEngine()
    audit_logger = AuditLogger()

    results = []

    for _, row in df.iterrows():

        transaction = row.to_dict()

        decision = engine.evaluate(transaction)
        audit_logger.record(decision)
        results.append({
            "transaction_id": decision.transaction_id,
            "risk_score": decision.risk_score,
            "risk_level": decision.risk_level,
            "action": decision.action,
            "abuse_label": row["abuse_label"],
        })

    results_df = pd.DataFrame(results)

    print("\nRAZORGUARD AI — RISK ENGINE V1")
    print("=" * 70)

    print("\nRisk-level distribution")
    print("-" * 70)

    print(
        results_df["risk_level"]
        .value_counts()
        .sort_index()
    )

    print("\nAction distribution")
    print("-" * 70)

    print(
        results_df["action"]
        .value_counts()
    )

    print("\nActual abuse rate by risk level")
    print("-" * 70)

    print(
        results_df
        .groupby("risk_level")["abuse_label"]
        .agg(["count", "mean"])
    )

    print("\nActual abuse rate by action")
    print("-" * 70)

    print(
        results_df
        .groupby("action")["abuse_label"]
        .agg(["count", "mean"])
    )

    output_path = (
        "data/processed/"
        "risk_engine_validation_v1.csv"
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()