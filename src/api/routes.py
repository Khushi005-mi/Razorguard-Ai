from fastapi import APIRouter, Header, HTTPException, Request, status

from src.api.schemas import (
    RazorpayWebhookPayload,
    RiskEvaluationResponse,
    RiskReviewUpdate,
    TransactionCreate,
    TransactionResponse,
)
from src.api.webhook_security import verify_razorpay_signature

from src.db.connection import get_connection
from src.db.repositories.audit_events import create_audit_event
from src.db.repositories.dashboard import get_dashboard_summary
from src.db.repositories.risk_decisions import create_risk_decision
from src.db.repositories.risk_reviews import (
    create_risk_review,
    get_pending_reviews,
    update_risk_review,
)
from src.db.repositories.transactions import (
    create_transaction,
    get_transaction,
)
from src.db.repositories.webhook_events import (
    create_webhook_event,
    mark_webhook_failed,
    mark_webhook_processed,
)
from src.db.repositories.webhook_normalization import (
    normalize_refund_webhook,
)

from src.risk_engine.engine import RiskEngine


router = APIRouter()

risk_engine = RiskEngine()


# ============================================================
# DASHBOARD
# ============================================================

@router.get("/dashboard/summary")
def dashboard_summary():
    return get_dashboard_summary()


# ============================================================
# PENDING REVIEWS
# ============================================================

@router.get("/reviews/pending")
def pending_reviews():
    try:
        return get_pending_reviews()

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve pending reviews.",
        ) from exc


# ============================================================
# HEALTH CHECK
# ============================================================

@router.get("/health")
def health_check():
    conn = None

    try:
        conn = get_connection()

        with conn.cursor() as cursor:
            cursor.execute("SELECT 1")
            cursor.fetchone()

        return {
            "status": "healthy",
            "database": "connected",
        }

    except Exception:
        return {
            "status": "unhealthy",
            "database": "disconnected",
        }

    finally:
        if conn is not None:
            conn.close()


# ============================================================
# TRANSACTION INGESTION
# ============================================================

@router.post(
    "/transactions",
    response_model=TransactionResponse,
    status_code=status.HTTP_201_CREATED,
)
def ingest_transaction(transaction: TransactionCreate):
    try:
        return create_transaction(transaction)

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to persist transaction.",
        ) from exc


# ============================================================
# RISK EVALUATION
# ============================================================

@router.post(
    "/transactions/{transaction_id}/risk",
    response_model=RiskEvaluationResponse,
)
def evaluate_transaction_risk(transaction_id: str):
    transaction = get_transaction(transaction_id)

    if transaction is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Transaction not found.",
        )

    try:
        decision = risk_engine.evaluate(transaction)

        created_decision = create_risk_decision(decision)

        review_created = False

        if (
            decision.risk_level == "HIGH"
            and decision.action == "FLAG_FOR_REVIEW"
        ):
            create_risk_review(
                transaction_id=decision.transaction_id,
                risk_decision_id=created_decision["decision_id"],
            )

            review_created = True

        create_audit_event(
            event_type="RISK_DECISION_CREATED",
            transaction_id=decision.transaction_id,
            event_data={
                "decision_id": int(created_decision["decision_id"]),
                "risk_score": float(decision.risk_score),
                "risk_level": str(decision.risk_level),
                "action": str(decision.action),
                "risk_signals": list(decision.risk_signals),
                "decision_reason": str(decision.decision_reason),
                "model_version": str(decision.model_version),
                "policy_version": str(decision.policy_version),
                "review_created": bool(review_created),
            },
        )

        return {
            "transaction_id": decision.transaction_id,
            "risk_score": decision.risk_score,
            "risk_level": decision.risk_level,
            "action": decision.action,
            "risk_signals": decision.risk_signals,
            "decision_reason": decision.decision_reason,
            "model_version": decision.model_version,
            "policy_version": decision.policy_version,
            "review_created": review_created,
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to evaluate transaction risk.",
        ) from exc


# ============================================================
# REVIEW UPDATE
# ============================================================

@router.patch("/reviews/{review_id}")
def update_review(
    review_id: int,
    review: RiskReviewUpdate,
):
    try:
        updated_review = update_risk_review(
            review_id=review_id,
            status=review.status,
            reviewer=review.reviewer,
            review_reason=review.review_reason,
        )

        if updated_review is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Risk review not found.",
            )

        create_audit_event(
            event_type="RISK_REVIEW_UPDATED",
            transaction_id=updated_review["transaction_id"],
            event_data={
                "review_id": int(review_id),
                "status": str(review.status),
                "reviewer": (
                    str(review.reviewer)
                    if review.reviewer is not None
                    else None
                ),
                "review_reason": (
                    str(review.review_reason)
                    if review.review_reason is not None
                    else None
                ),
            },
        )

        return updated_review

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update risk review.",
        ) from exc


# ============================================================
# RAZORPAY WEBHOOK
# ============================================================

@router.post("/webhooks/razorpay")
async def razorpay_webhook(
    request: Request,
    payload: RazorpayWebhookPayload,
    x_razorpay_event_id: str | None = Header(default=None),
    x_razorpay_signature: str | None = Header(default=None),
):
    # --------------------------------------------------------
    # 1. Require event ID
    # --------------------------------------------------------

    if not x_razorpay_event_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing X-Razorpay-Event-Id header.",
        )

    # --------------------------------------------------------
    # 2. Require and verify signature
    # --------------------------------------------------------

    if not x_razorpay_signature:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing X-Razorpay-Signature header.",
        )

    raw_body = await request.body()

    try:
        signature_valid = verify_razorpay_signature(
            raw_body,
            x_razorpay_signature,
        )

    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    if not signature_valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid webhook signature.",
        )

    # --------------------------------------------------------
    # 3. Persist webhook event
    # --------------------------------------------------------

    event = create_webhook_event(
        event_id=x_razorpay_event_id,
        event_type=payload.event,
        transaction_id=payload.payload.payment.id,
        payload=payload.model_dump(),
    )

    # --------------------------------------------------------
    # 4. Ignore duplicate deliveries
    # --------------------------------------------------------

    if event is None:
        return {
            "status": "duplicate",
            "event_id": x_razorpay_event_id,
        }

    try:
        # ----------------------------------------------------
        # 5. Extract Razorpay entities
        # ----------------------------------------------------

        refund = payload.payload.refund
        payment = payload.payload.payment

        payment_id = payment.id
        refund_id = refund.id

        # Razorpay sends INR amounts in paise.
        refund_amount = refund.amount / 100

        # ----------------------------------------------------
        # 6. Normalize webhook
        # ----------------------------------------------------

        normalized_transaction = normalize_refund_webhook(
            payment_id=payment_id,
            refund_amount=refund_amount,
            refund_id=refund_id,
            refund_created_at=refund.created_at,
        )

        # ----------------------------------------------------
        # 7. No matching internal transaction
        # ----------------------------------------------------

        if normalized_transaction is None:
            mark_webhook_processed(
                event["webhook_event_id"]
            )

            return {
                "status": "unmapped",
                "event_id": x_razorpay_event_id,
                "webhook_event_id": event["webhook_event_id"],
                "payment_id": payment_id,
                "message": (
                    "Webhook received, but no matching internal "
                    "transaction was found."
                ),
            }

        # ----------------------------------------------------
        # 8. Reload complete transaction
        # ----------------------------------------------------

        transaction = get_transaction(payment_id)

        if transaction is None:
            raise ValueError(
                "Transaction disappeared after normalization."
            )

        # ----------------------------------------------------
        # 9. Run RiskEngine
        # ----------------------------------------------------

        decision = risk_engine.evaluate(transaction)

        # ----------------------------------------------------
        # 10. Persist risk decision
        # ----------------------------------------------------

        created_decision = create_risk_decision(decision)

        # ----------------------------------------------------
        # 11. Automatically create review for HIGH risk
        # ----------------------------------------------------

        review_created = False

        if (
            decision.risk_level == "HIGH"
            and decision.action == "FLAG_FOR_REVIEW"
        ):
            create_risk_review(
                transaction_id=decision.transaction_id,
                risk_decision_id=created_decision["decision_id"],
            )

            review_created = True

        # ----------------------------------------------------
        # 12. Create audit event
        # ----------------------------------------------------

        create_audit_event(
            event_type="WEBHOOK_RISK_DECISION_CREATED",
            transaction_id=decision.transaction_id,
            event_data={
                "webhook_event_id": int(
                    event["webhook_event_id"]
                ),
                "event_id": str(x_razorpay_event_id),
                "event_type": str(payload.event),
                "payment_id": str(payment_id),
                "refund_id": str(refund_id),
                "refund_amount": float(refund_amount),
                "refund_delay_hours": float(
                    transaction["refund_delay_hours"]
                ),
                "refund_amount_ratio": float(
                    transaction["refund_amount_ratio"]
                ),
                "risk_score": float(decision.risk_score),
                "risk_level": str(decision.risk_level),
                "action": str(decision.action),
                "risk_signals": list(decision.risk_signals),
                "model_version": str(
                    decision.model_version
                ),
                "policy_version": str(
                    decision.policy_version
                ),
                "decision_reason": str(
                    decision.decision_reason
                ),
            },
        )

        # ----------------------------------------------------
        # 13. Mark webhook processed
        # ----------------------------------------------------

        mark_webhook_processed(
            event["webhook_event_id"]
        )

        # ----------------------------------------------------
        # 14. Return result
        # ----------------------------------------------------

        return {
            "status": "processed",
            "event_id": x_razorpay_event_id,
            "webhook_event_id": event["webhook_event_id"],
            "payment_id": payment_id,
            "transaction_id": decision.transaction_id,
            "refund_id": refund_id,
            "refund_amount": refund_amount,
            "refund_delay_hours": transaction[
                "refund_delay_hours"
            ],
            "refund_amount_ratio": transaction[
                "refund_amount_ratio"
            ],
            "risk_score": decision.risk_score,
            "risk_level": decision.risk_level,
            "action": decision.action,
            "risk_signals": decision.risk_signals,
            "decision_reason": decision.decision_reason,
            "model_version": decision.model_version,
            "policy_version": decision.policy_version,
            "review_created": review_created,
        }

    except Exception as exc:
        mark_webhook_failed(
            event["webhook_event_id"],
            str(exc),
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process webhook.",
        ) from exc