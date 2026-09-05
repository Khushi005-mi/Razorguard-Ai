BEGIN;

CREATE TABLE IF NOT EXISTS transactions (
    transaction_id VARCHAR(255) PRIMARY KEY,
    customer_id VARCHAR(255) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    amount NUMERIC(18,2) NOT NULL CHECK (amount >= 0),
    payment_method VARCHAR(100) NOT NULL,

    refund_requested BOOLEAN NOT NULL DEFAULT FALSE,
    refund_amount NUMERIC(18,2) NOT NULL DEFAULT 0
        CHECK (refund_amount >= 0 AND refund_amount <= amount),
    refund_delay_hours NUMERIC(18,2),

    previous_orders INTEGER NOT NULL DEFAULT 0,
    previous_refunds INTEGER NOT NULL DEFAULT 0,

    historical_refund_rate NUMERIC(8,6) NOT NULL DEFAULT 0
        CHECK (historical_refund_rate >= 0 AND historical_refund_rate <= 1),

    high_value_transaction BOOLEAN NOT NULL DEFAULT FALSE,
    rapid_refund BOOLEAN NOT NULL DEFAULT FALSE,

    refund_amount_ratio NUMERIC(8,6) NOT NULL DEFAULT 0
        CHECK (refund_amount_ratio >= 0 AND refund_amount_ratio <= 1),

    refund_frequency_signal INTEGER NOT NULL DEFAULT 0,

    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CHECK (refund_requested = TRUE OR refund_amount = 0)
);

CREATE INDEX IF NOT EXISTS idx_transactions_customer_id
    ON transactions(customer_id);


CREATE TABLE IF NOT EXISTS risk_decisions (
    decision_id BIGSERIAL PRIMARY KEY,

    transaction_id VARCHAR(255) NOT NULL UNIQUE
        REFERENCES transactions(transaction_id),

    risk_score NUMERIC(10,8) NOT NULL
        CHECK (risk_score >= 0 AND risk_score <= 1),

    risk_level VARCHAR(32) NOT NULL
        CHECK (risk_level IN ('LOW', 'MEDIUM', 'HIGH')),

    action VARCHAR(64) NOT NULL
        CHECK (action IN ('ALLOW', 'MONITOR', 'FLAG_FOR_REVIEW')),

    risk_signals JSONB NOT NULL DEFAULT '[]'::jsonb,

    decision_reason TEXT NOT NULL,

    model_version VARCHAR(100) NOT NULL,
    policy_version VARCHAR(100) NOT NULL,

    decision_timestamp TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_risk_decisions_transaction
    ON risk_decisions(transaction_id);

CREATE INDEX IF NOT EXISTS idx_risk_decisions_risk_level
    ON risk_decisions(risk_level);

CREATE INDEX IF NOT EXISTS idx_risk_decisions_timestamp
    ON risk_decisions(decision_timestamp DESC);


CREATE TABLE IF NOT EXISTS audit_events (
    audit_event_id BIGSERIAL PRIMARY KEY,

    transaction_id VARCHAR(255)
        REFERENCES transactions(transaction_id),

    event_type VARCHAR(100) NOT NULL,

    event_data JSONB NOT NULL DEFAULT '{}'::jsonb,

    event_timestamp TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_audit_events_transaction
    ON audit_events(transaction_id);

CREATE INDEX IF NOT EXISTS idx_audit_events_type
    ON audit_events(event_type);

CREATE INDEX IF NOT EXISTS idx_audit_events_timestamp
    ON audit_events(event_timestamp DESC);


CREATE TABLE IF NOT EXISTS schema_migrations (
    version VARCHAR(50) PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO schema_migrations (version)
VALUES ('001_initial_schema')
ON CONFLICT (version) DO NOTHING;

COMMIT;
