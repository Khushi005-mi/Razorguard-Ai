from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TransactionCreate(BaseModel):
    """
    Request schema for creating a transaction.
    """

    model_config = ConfigDict(extra="forbid")

    transaction_id: str = Field(min_length=1)
    customer_id: str = Field(min_length=1)
    timestamp: datetime

    amount: float = Field(ge=0)
    payment_method: str = Field(min_length=1)

    refund_requested: bool = False
    refund_amount: float = Field(default=0, ge=0)
    refund_delay_hours: float | None = Field(default=None, ge=0)

    previous_orders: int = Field(default=0, ge=0)
    previous_refunds: int = Field(default=0, ge=0)
    historical_refund_rate: float = Field(default=0, ge=0, le=1)

    high_value_transaction: bool = False
    rapid_refund: bool = False
    refund_amount_ratio: float = Field(default=0, ge=0, le=1)
    refund_frequency_signal: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_refund_data(self):
        if self.refund_amount > self.amount:
            raise ValueError(
                "refund_amount cannot exceed transaction amount"
            )

        if not self.refund_requested and self.refund_amount != 0:
            raise ValueError(
                "refund_amount must be 0 when refund_requested is false"
            )

        return self


class TransactionResponse(BaseModel):
    """
    Response schema returned after a transaction is stored.
    """

    transaction_id: str
    customer_id: str
    timestamp: datetime
    amount: float
    payment_method: str
    refund_requested: bool
    refund_amount: float
    refund_delay_hours: float | None
    previous_orders: int
    previous_refunds: int
    historical_refund_rate: float
    high_value_transaction: bool
    rapid_refund: bool
    refund_amount_ratio: float
    refund_frequency_signal: int
    created_at: datetime


class RiskEvaluationResponse(BaseModel):
    """
    Response returned after evaluating transaction risk.
    """

    transaction_id: str
    risk_score: float = Field(ge=0, le=1)
    risk_level: str
    action: str
    risk_signals: list[str]
    decision_reason: str
    model_version: str
    policy_version: str


class RiskReviewUpdate(BaseModel):
    """
    Request schema for completing a risk review.
    """

    model_config = ConfigDict(extra="forbid")

    status: str
    reviewer: str = Field(min_length=1)
    review_reason: str = Field(min_length=1)

class RazorpayWebhookRefund(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    amount: int
    payment_id: str
    created_at: int


class RazorpayWebhookPayment(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    amount: int
    method: str
    created_at: int


class RazorpayWebhookPayloadData(BaseModel):
    model_config = ConfigDict(extra="allow")

    refund: RazorpayWebhookRefund
    payment: RazorpayWebhookPayment


class RazorpayWebhookPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    event: str
    payload: RazorpayWebhookPayloadData
    