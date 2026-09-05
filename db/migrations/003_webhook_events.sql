CREATE TABLE IF NOT EXISTS webhook_events (
    webhook_event_id BIGSERIAL PRIMARY KEY,

    event_id VARCHAR(255) NOT NULL UNIQUE,

    event_type VARCHAR(100) NOT NULL,

    transaction_id VARCHAR(255),

    payload JSONB NOT NULL,

    status VARCHAR(32) NOT NULL DEFAULT 'RECEIVED'
        CHECK (status IN ('RECEIVED', 'PROCESSED', 'FAILED')),

    received_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    processed_at TIMESTAMPTZ,

    error_message TEXT
);

CREATE INDEX IF NOT EXISTS idx_webhook_events_event_type
    ON webhook_events(event_type);

CREATE INDEX IF NOT EXISTS idx_webhook_events_transaction
    ON webhook_events(transaction_id);

CREATE INDEX IF NOT EXISTS idx_webhook_events_received_at
    ON webhook_events(received_at DESC);

CREATE INDEX IF NOT EXISTS idx_webhook_events_status
    ON webhook_events(status);
