CREATE TABLE IF NOT EXISTS risk_reviews (
    review_id BIGSERIAL PRIMARY KEY,

    transaction_id VARCHAR(255) NOT NULL
        REFERENCES transactions(transaction_id),

    risk_decision_id BIGINT NOT NULL
        REFERENCES risk_decisions(decision_id),

    status VARCHAR(32) NOT NULL DEFAULT 'PENDING'
        CHECK (status IN ('PENDING', 'CONFIRMED_ABUSE', 'FALSE_POSITIVE')),

    reviewer VARCHAR(255),

    review_reason TEXT,

    reviewed_at TIMESTAMPTZ,

    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_risk_reviews_status
    ON risk_reviews(status);

CREATE INDEX IF NOT EXISTS idx_risk_reviews_transaction
    ON risk_reviews(transaction_id);

CREATE INDEX IF NOT EXISTS idx_risk_reviews_created_at
    ON risk_reviews(created_at DESC);
