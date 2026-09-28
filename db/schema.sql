-- RazorRisk PostgreSQL-first schema.
-- SQLite remains supported only as an explicit test/local fallback.

CREATE TABLE IF NOT EXISTS users (
    user_id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(100), email VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    account_status VARCHAR(20) DEFAULT 'ACTIVE'
);
CREATE TABLE IF NOT EXISTS devices (
    device_id VARCHAR(50) PRIMARY KEY,
    device_type VARCHAR(50), os VARCHAR(50),
    is_vpn_proxy BOOLEAN DEFAULT FALSE,
    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS ip_addresses (
    ip_address VARCHAR(45) PRIMARY KEY,
    country VARCHAR(50), city VARCHAR(50), isp VARCHAR(100),
    is_suspicious_proxy BOOLEAN DEFAULT FALSE
);
CREATE TABLE IF NOT EXISTS merchants (
    merchant_id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(100), category VARCHAR(50),
    fraud_rate DOUBLE PRECISION DEFAULT 0.01,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS transactions (
    transaction_id VARCHAR(50) PRIMARY KEY,
    user_id VARCHAR(50) REFERENCES users(user_id),
    device_id VARCHAR(50) REFERENCES devices(device_id),
    ip_address VARCHAR(45) REFERENCES ip_addresses(ip_address),
    merchant_id VARCHAR(50) REFERENCES merchants(merchant_id),
    amount DOUBLE PRECISION NOT NULL,
    currency VARCHAR(10) DEFAULT 'INR',
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(20) DEFAULT 'COMPLETED',
    velocity_1h INTEGER DEFAULT 1,
    velocity_enabled BOOLEAN DEFAULT FALSE,
    velocity_source TEXT DEFAULT 'BACKEND',
    amount_zscore_prior DOUBLE PRECISION DEFAULT 0.0,
    is_fraud_ground_truth BOOLEAN DEFAULT FALSE
);
CREATE TABLE IF NOT EXISTS risk_scores (
    scoring_id VARCHAR(50) PRIMARY KEY,
    transaction_id VARCHAR(50) REFERENCES transactions(transaction_id),
    risk_score DOUBLE PRECISION NOT NULL,
    tabular_score DOUBLE PRECISION NOT NULL,
    gnn_score DOUBLE PRECISION NOT NULL,
    stacker_calibrated_score DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    velocity_multiplier DOUBLE PRECISION DEFAULT 1.0,
    evidence_multiplier DOUBLE PRECISION DEFAULT 1.0,
    risk_tier VARCHAR(20) NOT NULL,
    decision VARCHAR(30) NOT NULL,
    scored_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS investigation_reports (
    investigation_id VARCHAR(50) PRIMARY KEY,
    transaction_id VARCHAR(50) REFERENCES transactions(transaction_id),
    risk_score DOUBLE PRECISION NOT NULL,
    evidence_json TEXT NOT NULL,
    fraud_hypothesis TEXT NOT NULL,
    recommended_action VARCHAR(50) NOT NULL,
    summary_report TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS system_logs (
    id BIGSERIAL PRIMARY KEY,
    module VARCHAR(50), level VARCHAR(20), message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS human_reviews (
    review_id VARCHAR(60) PRIMARY KEY,
    transaction_id VARCHAR(50) REFERENCES transactions(transaction_id),
    status VARCHAR(20) NOT NULL DEFAULT 'PENDING',
    risk_score DOUBLE PRECISION NOT NULL,
    reasons_json TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    reviewer VARCHAR(80), reviewer_decision VARCHAR(20), reviewer_rationale TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    reviewed_at TIMESTAMP
);
CREATE TABLE IF NOT EXISTS user_watchlist (
    user_id VARCHAR(50) PRIMARY KEY REFERENCES users(user_id),
    reason VARCHAR(50) NOT NULL,
    source_transaction_id VARCHAR(50),
    flagged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tx_user ON transactions(user_id);
CREATE INDEX IF NOT EXISTS idx_tx_device ON transactions(device_id);
CREATE INDEX IF NOT EXISTS idx_tx_ip ON transactions(ip_address);
CREATE INDEX IF NOT EXISTS idx_tx_timestamp ON transactions(timestamp);
CREATE INDEX IF NOT EXISTS idx_human_reviews_status ON human_reviews(status);
CREATE INDEX IF NOT EXISTS idx_human_reviews_created ON human_reviews(created_at);
CREATE INDEX IF NOT EXISTS idx_watchlist_expires ON user_watchlist(expires_at);
