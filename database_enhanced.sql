-- Enhanced FinAgent Database Schema
-- Simplified to avoid user_id issues while adding enhanced features

-- Portfolio Analytics Table (enhanced)
CREATE TABLE IF NOT EXISTS portfolio_analytics (
    id BIGSERIAL PRIMARY KEY,
    analysis_date TIMESTAMPTZ DEFAULT NOW(),
    total_value DECIMAL(15,2) NOT NULL,
    allocation JSONB NOT NULL,
    drift_analysis JSONB,
    risk_assessment JSONB,
    rebalance_recommendation JSONB,
    portfolio_metrics JSONB,
    compliance_status JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Market Data Cache Table (enhanced)
CREATE TABLE IF NOT EXISTS market_data_cache (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    price DECIMAL(10,4),
    change_amount DECIMAL(10,4),
    change_percent VARCHAR(10),
    volume BIGINT,
    high DECIMAL(10,4),
    low DECIMAL(10,4),
    technical_analysis JSONB,
    data_source VARCHAR(20) DEFAULT 'alpha_vantage',
    last_updated TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- AI Analysis Log Table (new)
CREATE TABLE IF NOT EXISTS ai_analysis_log (
    id BIGSERIAL PRIMARY KEY,
    agent_type VARCHAR(50) NOT NULL,
    query_text TEXT,
    analysis_result JSONB,
    confidence_score DECIMAL(3,2),
    symbols TEXT[],
    processing_time_ms INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Market Sentiment Table (new)
CREATE TABLE IF NOT EXISTS market_sentiment (
    id BIGSERIAL PRIMARY KEY,
    sentiment_date DATE DEFAULT CURRENT_DATE,
    overall_sentiment VARCHAR(20),
    fear_greed_index INTEGER,
    market_trend VARCHAR(20),
    volatility_index INTEGER,
    data_source VARCHAR(20) DEFAULT 'mock',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Strategy Performance Table (new)
CREATE TABLE IF NOT EXISTS strategy_performance (
    id BIGSERIAL PRIMARY KEY,
    strategy_name VARCHAR(50) DEFAULT 'Straight Arrow',
    performance_date DATE DEFAULT CURRENT_DATE,
    expected_return DECIMAL(5,4),
    volatility DECIMAL(5,4),
    sharpe_ratio DECIMAL(5,2),
    max_drawdown DECIMAL(5,4),
    performance_metrics JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Compliance Audit Table (new)
CREATE TABLE IF NOT EXISTS compliance_audit (
    id BIGSERIAL PRIMARY KEY,
    audit_date TIMESTAMPTZ DEFAULT NOW(),
    compliance_status VARCHAR(20),
    violations JSONB,
    warnings JSONB,
    portfolio_data JSONB,
    audit_type VARCHAR(50) DEFAULT 'portfolio_analysis',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_portfolio_analytics_date ON portfolio_analytics(analysis_date DESC);
CREATE INDEX IF NOT EXISTS idx_market_data_symbol ON market_data_cache(symbol);
CREATE INDEX IF NOT EXISTS idx_market_data_updated ON market_data_cache(last_updated DESC);
CREATE INDEX IF NOT EXISTS idx_ai_analysis_agent ON ai_analysis_log(agent_type);
CREATE INDEX IF NOT EXISTS idx_ai_analysis_date ON ai_analysis_log(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_market_sentiment_date ON market_sentiment(sentiment_date DESC);
CREATE INDEX IF NOT EXISTS idx_strategy_performance_date ON strategy_performance(performance_date DESC);
CREATE INDEX IF NOT EXISTS idx_compliance_audit_date ON compliance_audit(audit_date DESC);

-- Create a function to clean old data (optional)
CREATE OR REPLACE FUNCTION cleanup_old_data()
RETURNS void AS $$
BEGIN
    -- Keep only last 100 portfolio analyses
    DELETE FROM portfolio_analytics 
    WHERE id NOT IN (
        SELECT id FROM portfolio_analytics 
        ORDER BY analysis_date DESC 
        LIMIT 100
    );
    
    -- Keep market data for last 30 days
    DELETE FROM market_data_cache 
    WHERE last_updated < NOW() - INTERVAL '30 days';
    
    -- Keep AI analysis log for last 7 days
    DELETE FROM ai_analysis_log 
    WHERE created_at < NOW() - INTERVAL '7 days';
    
    -- Keep market sentiment for last 90 days
    DELETE FROM market_sentiment 
    WHERE sentiment_date < CURRENT_DATE - INTERVAL '90 days';
    
    -- Keep strategy performance for last 365 days
    DELETE FROM strategy_performance 
    WHERE performance_date < CURRENT_DATE - INTERVAL '365 days';
    
    -- Keep compliance audit for last 30 days
    DELETE FROM compliance_audit 
    WHERE audit_date < NOW() - INTERVAL '30 days';
END;
$$ LANGUAGE plpgsql;

-- Enable Row Level Security (optional)
ALTER TABLE portfolio_analytics ENABLE ROW LEVEL SECURITY;
ALTER TABLE market_data_cache ENABLE ROW LEVEL SECURITY;
ALTER TABLE ai_analysis_log ENABLE ROW LEVEL SECURITY;
ALTER TABLE market_sentiment ENABLE ROW LEVEL SECURITY;
ALTER TABLE strategy_performance ENABLE ROW LEVEL SECURITY;
ALTER TABLE compliance_audit ENABLE ROW LEVEL SECURITY;

-- Create policies for public access (since we don't have user authentication)
CREATE POLICY "Allow all operations" ON portfolio_analytics FOR ALL USING (true);
CREATE POLICY "Allow all operations" ON market_data_cache FOR ALL USING (true);
CREATE POLICY "Allow all operations" ON ai_analysis_log FOR ALL USING (true);
CREATE POLICY "Allow all operations" ON market_sentiment FOR ALL USING (true);
CREATE POLICY "Allow all operations" ON strategy_performance FOR ALL USING (true);
CREATE POLICY "Allow all operations" ON compliance_audit FOR ALL USING (true); 