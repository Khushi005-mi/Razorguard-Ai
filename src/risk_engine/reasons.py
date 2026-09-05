def generate_reasons(transaction: dict) -> list[str]:
    reasons = []

    if transaction.get("historical_refund_rate", 0) >= 0.30:
        reasons.append("High historical refund rate")

    if transaction.get("previous_refunds", 0) >= 3:
        reasons.append("Frequent refund activity")

    if transaction.get("refund_amount_ratio", 0) >= 0.80:
        reasons.append("Refund amount is unusually high relative to transaction value")

    if transaction.get("refund_frequency_signal", 0) >= 2:
        reasons.append("Elevated refund frequency signal")

    if transaction.get("rapid_refund", 0) == 1:
        reasons.append("Rapid refund request")

    if transaction.get("high_value_transaction", 0) == 1:
        reasons.append("High-value transaction")

    return reasons