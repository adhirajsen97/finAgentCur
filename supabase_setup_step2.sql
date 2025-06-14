-- Step 2: Create portfolio analytics table
CREATE TABLE IF NOT EXISTS portfolio_analytics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    analysis_date TIMESTAMPTZ DEFAULT NOW(),
    total_value DECIMAL(15,2) NOT NULL,
    allocation JSONB,
    drift_analysis JSONB,
    risk_assessment JSONB,
    rebalance_recommendation JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
); 