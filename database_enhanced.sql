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
    performance_date DATE DEFAULT CURRENT_DATE,
    strategy_type VARCHAR(50) DEFAULT 'Straight Arrow',
    expected_return DECIMAL(5,4),
    actual_return DECIMAL(5,4),
    volatility DECIMAL(5,4),
    sharpe_ratio DECIMAL(5,2),
    max_drawdown DECIMAL(5,4),
    benchmark_return DECIMAL(5,4),
    outperformance DECIMAL(5,4),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Compliance Audit Table (new)
CREATE TABLE IF NOT EXISTS compliance_audit (
    id BIGSERIAL PRIMARY KEY,
    audit_date TIMESTAMPTZ DEFAULT NOW(),
    audit_type VARCHAR(50),
    compliance_check JSONB,
    violations JSONB,
    risk_level VARCHAR(20),
    recommendations JSONB,
    portfolio_data JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- NEW: UNIFIED STRATEGY SYSTEM TABLES
-- ============================================================================

-- Investment Strategies Table (NEW)
CREATE TABLE IF NOT EXISTS investment_strategies (
    id BIGSERIAL PRIMARY KEY,
    strategy_id VARCHAR(50) UNIQUE NOT NULL,
    user_id VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    strategy_type VARCHAR(50) NOT NULL DEFAULT 'Straight Arrow Enhanced',
    time_horizon VARCHAR(50),
    risk_tolerance VARCHAR(20),
    investment_goals JSONB,
    portfolio_snapshot JSONB NOT NULL,
    market_context JSONB,
    ai_insights JSONB,
    trade_orders JSONB NOT NULL,
    execution_guidelines JSONB,
    strategy_summary JSONB,
    risk_warnings JSONB,
    status VARCHAR(20) DEFAULT 'created',
    executed_at TIMESTAMPTZ,
    performance_metrics JSONB,
    next_review_date TIMESTAMPTZ,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Trade Executions Table (NEW)
CREATE TABLE IF NOT EXISTS trade_executions (
    id BIGSERIAL PRIMARY KEY,
    execution_id VARCHAR(50) UNIQUE NOT NULL,
    strategy_id VARCHAR(50) REFERENCES investment_strategies(strategy_id),
    user_id VARCHAR(50),
    symbol VARCHAR(10) NOT NULL,
    action VARCHAR(10) NOT NULL,
    order_type VARCHAR(10) NOT NULL,
    requested_quantity DECIMAL(15,4),
    requested_price DECIMAL(10,4),
    executed_quantity DECIMAL(15,4),
    executed_price DECIMAL(10,4),
    commission DECIMAL(10,4),
    execution_method VARCHAR(20) DEFAULT 'SIMULATED',
    status VARCHAR(20) NOT NULL,
    priority VARCHAR(10),
    reason TEXT,
    expected_impact TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    executed_at TIMESTAMPTZ,
    error_message TEXT,
    slippage DECIMAL(10,4),
    portfolio_impact JSONB,
    notes TEXT
);

-- Portfolio Positions Table (NEW)
CREATE TABLE IF NOT EXISTS portfolio_positions (
    id BIGSERIAL PRIMARY KEY,
    user_id VARCHAR(50) NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    shares DECIMAL(15,4) NOT NULL DEFAULT 0,
    average_cost DECIMAL(10,4) NOT NULL DEFAULT 0,
    market_value DECIMAL(15,2),
    unrealized_pnl DECIMAL(15,2),
    cost_basis DECIMAL(15,2),
    last_updated TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(user_id, symbol)
);

-- Portfolio History Table (NEW)
CREATE TABLE IF NOT EXISTS portfolio_history (
    id BIGSERIAL PRIMARY KEY,
    user_id VARCHAR(50) NOT NULL,
    snapshot_date DATE NOT NULL,
    total_value DECIMAL(15,2),
    cash_balance DECIMAL(15,2),
    positions JSONB,
    allocation JSONB,
    performance_metrics JSONB,
    daily_return DECIMAL(8,4),
    cumulative_return DECIMAL(8,4),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(user_id, snapshot_date)
);

-- Strategy Execution Sessions Table (NEW)
CREATE TABLE IF NOT EXISTS strategy_execution_sessions (
    id BIGSERIAL PRIMARY KEY,
    session_id VARCHAR(50) UNIQUE NOT NULL,
    strategy_id VARCHAR(50) REFERENCES investment_strategies(strategy_id),
    user_id VARCHAR(50),
    execution_method VARCHAR(20) NOT NULL,
    status VARCHAR(20) NOT NULL,
    total_orders INTEGER,
    executed_orders INTEGER,
    failed_orders INTEGER,
    total_invested DECIMAL(15,2),
    total_divested DECIMAL(15,2),
    execution_time_ms INTEGER,
    started_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    error_message TEXT,
    execution_options JSONB,
    final_portfolio JSONB
);

-- Indexes for better performance
CREATE INDEX IF NOT EXISTS idx_portfolio_analytics_date ON portfolio_analytics(analysis_date);
CREATE INDEX IF NOT EXISTS idx_market_data_symbol ON market_data_cache(symbol);
CREATE INDEX IF NOT EXISTS idx_market_data_updated ON market_data_cache(last_updated);
CREATE INDEX IF NOT EXISTS idx_ai_analysis_agent ON ai_analysis_log(agent_type);
CREATE INDEX IF NOT EXISTS idx_ai_analysis_date ON ai_analysis_log(created_at);
CREATE INDEX IF NOT EXISTS idx_sentiment_date ON market_sentiment(sentiment_date);
CREATE INDEX IF NOT EXISTS idx_strategy_performance_date ON strategy_performance(performance_date);
CREATE INDEX IF NOT EXISTS idx_compliance_audit_date ON compliance_audit(audit_date);

-- New indexes for unified strategy system
CREATE INDEX IF NOT EXISTS idx_investment_strategies_user ON investment_strategies(user_id);
CREATE INDEX IF NOT EXISTS idx_investment_strategies_status ON investment_strategies(status);
CREATE INDEX IF NOT EXISTS idx_investment_strategies_created ON investment_strategies(created_at);
CREATE INDEX IF NOT EXISTS idx_trade_executions_strategy ON trade_executions(strategy_id);
CREATE INDEX IF NOT EXISTS idx_trade_executions_user ON trade_executions(user_id);
CREATE INDEX IF NOT EXISTS idx_trade_executions_symbol ON trade_executions(symbol);
CREATE INDEX IF NOT EXISTS idx_trade_executions_status ON trade_executions(status);
CREATE INDEX IF NOT EXISTS idx_portfolio_positions_user ON portfolio_positions(user_id);
CREATE INDEX IF NOT EXISTS idx_portfolio_history_user_date ON portfolio_history(user_id, snapshot_date);
CREATE INDEX IF NOT EXISTS idx_execution_sessions_strategy ON strategy_execution_sessions(strategy_id);

-- Function to clean up old data
CREATE OR REPLACE FUNCTION cleanup_old_data()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER := 0;
BEGIN
    -- Keep only last 100 portfolio analyses
    DELETE FROM portfolio_analytics 
    WHERE id NOT IN (
        SELECT id FROM portfolio_analytics 
        ORDER BY analysis_date DESC 
        LIMIT 100
    );
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    
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
    
    -- Keep investment strategies for last 90 days
    DELETE FROM investment_strategies 
    WHERE created_at < NOW() - INTERVAL '90 days';
    
    -- Keep trade executions for last 90 days
    DELETE FROM trade_executions 
    WHERE created_at < NOW() - INTERVAL '90 days';
    
    -- Keep portfolio history for last 2 years
    DELETE FROM portfolio_history 
    WHERE snapshot_date < CURRENT_DATE - INTERVAL '2 years';
    
    -- Keep execution sessions for last 30 days
    DELETE FROM strategy_execution_sessions 
    WHERE started_at < NOW() - INTERVAL '30 days';
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Function to update portfolio positions after trade execution
CREATE OR REPLACE FUNCTION update_portfolio_position()
RETURNS TRIGGER AS $$
BEGIN
    -- Only process successful executions
    IF NEW.status = 'success' AND NEW.executed_quantity > 0 THEN
        -- Update or insert portfolio position
        INSERT INTO portfolio_positions (user_id, symbol, shares, average_cost, last_updated)
        VALUES (
            NEW.user_id, 
            NEW.symbol, 
            CASE 
                WHEN NEW.action = 'BUY' THEN NEW.executed_quantity
                WHEN NEW.action = 'SELL' THEN -NEW.executed_quantity
                ELSE 0
            END,
            NEW.executed_price,
            NEW.executed_at
        )
        ON CONFLICT (user_id, symbol) 
        DO UPDATE SET
            shares = CASE 
                WHEN NEW.action = 'BUY' THEN portfolio_positions.shares + NEW.executed_quantity
                WHEN NEW.action = 'SELL' THEN portfolio_positions.shares - NEW.executed_quantity
                ELSE portfolio_positions.shares
            END,
            average_cost = CASE 
                WHEN NEW.action = 'BUY' THEN 
                    ((portfolio_positions.shares * portfolio_positions.average_cost) + (NEW.executed_quantity * NEW.executed_price)) / 
                    (portfolio_positions.shares + NEW.executed_quantity)
                ELSE portfolio_positions.average_cost
            END,
            last_updated = NEW.executed_at;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for automatic portfolio position updates
DROP TRIGGER IF EXISTS trigger_update_portfolio_position ON trade_executions;
CREATE TRIGGER trigger_update_portfolio_position
    AFTER UPDATE ON trade_executions
    FOR EACH ROW
    WHEN (NEW.status = 'success' AND OLD.status != 'success')
    EXECUTE FUNCTION update_portfolio_position();

-- Enable Row Level Security (optional)
ALTER TABLE portfolio_analytics ENABLE ROW LEVEL SECURITY;
ALTER TABLE market_data_cache ENABLE ROW LEVEL SECURITY;
ALTER TABLE ai_analysis_log ENABLE ROW LEVEL SECURITY;
ALTER TABLE market_sentiment ENABLE ROW LEVEL SECURITY;
ALTER TABLE strategy_performance ENABLE ROW LEVEL SECURITY;
ALTER TABLE compliance_audit ENABLE ROW LEVEL SECURITY;
ALTER TABLE investment_strategies ENABLE ROW LEVEL SECURITY;
ALTER TABLE trade_executions ENABLE ROW LEVEL SECURITY;
ALTER TABLE portfolio_positions ENABLE ROW LEVEL SECURITY;
ALTER TABLE portfolio_history ENABLE ROW LEVEL SECURITY;
ALTER TABLE strategy_execution_sessions ENABLE ROW LEVEL SECURITY;

-- Create policies for public access (since we don't have user authentication)
CREATE POLICY "Allow all operations" ON portfolio_analytics FOR ALL USING (true);
CREATE POLICY "Allow all operations" ON market_data_cache FOR ALL USING (true);
CREATE POLICY "Allow all operations" ON ai_analysis_log FOR ALL USING (true);
CREATE POLICY "Allow all operations" ON market_sentiment FOR ALL USING (true);
CREATE POLICY "Allow all operations" ON strategy_performance FOR ALL USING (true);
CREATE POLICY "Allow all operations" ON compliance_audit FOR ALL USING (true);
CREATE POLICY "Allow all operations" ON investment_strategies FOR ALL USING (true);
CREATE POLICY "Allow all operations" ON trade_executions FOR ALL USING (true);
CREATE POLICY "Allow all operations" ON portfolio_positions FOR ALL USING (true);
CREATE POLICY "Allow all operations" ON portfolio_history FOR ALL USING (true);
CREATE POLICY "Allow all operations" ON strategy_execution_sessions FOR ALL USING (true);

-- Insert sample data for testing
INSERT INTO portfolio_analytics (total_value, allocation, drift_analysis, risk_assessment, rebalance_recommendation) 
VALUES (
    100000.00, 
    '{"VTI": 0.60, "BNDX": 0.30, "GSG": 0.10}',
    '{"VTI": {"drift": 0.05}, "BNDX": {"drift": -0.02}, "GSG": {"drift": 0.01}}',
    '{"overall_risk": "MODERATE", "sharpe_ratio": 0.85}',
    '[{"symbol": "VTI", "action": "REDUCE", "priority": "MEDIUM"}]'
) ON CONFLICT DO NOTHING;

INSERT INTO market_sentiment (overall_sentiment, fear_greed_index, market_trend, volatility_index)
VALUES ('NEUTRAL', 65, 'SIDEWAYS', 18) ON CONFLICT DO NOTHING;

INSERT INTO strategy_performance (expected_return, actual_return, volatility, sharpe_ratio)
VALUES (0.08, 0.075, 0.12, 0.85) ON CONFLICT DO NOTHING; 