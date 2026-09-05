import hashlib
import hmac
import os


def verify_razorpay_signature(
    raw_body: bytes,
    signature: str,
) -> bool:
    secret = os.getenv("RAZORPAY_WEBHOOK_SECRET")

    if not secret:
        raise RuntimeError(
            "RAZORPAY_WEBHOOK_SECRET is not configured."
        )

    expected_signature = hmac.new(
        secret.encode("utf-8"),
        raw_body,
        hashlib.sha256,
    ).hexdigest()

    return hmac.compare_digest(
        expected_signature,
        signature,
    )

