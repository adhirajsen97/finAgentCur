-- Initialize FinAgent Database
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Portfolio positions table
CREATE TABLE IF NOT EXISTS portfolio_positions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(10) NOT NULL,
    quantity DECIMAL(15,6) NOT NULL,
    average_cost DECIMAL(15,2) NOT NULL,
    current_value DECIMAL(15,2),
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Trade history table
CREATE TABLE IF NOT EXISTS trade_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(10) NOT NULL,
    trade_type VARCHAR(10) NOT NULL CHECK (trade_type IN ('BUY', 'SELL')),
    quantity DECIMAL(15,6) NOT NULL,
    price DECIMAL(15,2) NOT NULL,
    total_amount DECIMAL(15,2) NOT NULL,
    fees DECIMAL(15,2) DEFAULT 0,
    trade_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    strategy VARCHAR(50) DEFAULT 'Straight Arrow',
    notes TEXT
);

-- Portfolio analytics table
CREATE TABLE IF NOT EXISTS portfolio_analytics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    analysis_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    total_value DECIMAL(15,2) NOT NULL,
    total_return DECIMAL(10,4),
    sharpe_ratio DECIMAL(10,4),
    volatility DECIMAL(10,4),
    max_drawdown DECIMAL(10,4),
    allocation JSONB,
    drift_analysis JSONB,
    risk_assessment JSONB,
    rebalance_recommendation JSONB
);

-- Market data cache table
CREATE TABLE IF NOT EXISTS market_data_cache (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(10) NOT NULL,
    data_type VARCHAR(50) NOT NULL,
    data JSONB NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, data_type)
);

-- Agent execution logs
CREATE TABLE IF NOT EXISTS agent_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_type VARCHAR(50) NOT NULL,
    task_id VARCHAR(100),
    status VARCHAR(20) NOT NULL CHECK (status IN ('STARTED', 'COMPLETED', 'FAILED')),
    input_data JSONB,
    output_data JSONB,
    error_message TEXT,
    execution_time_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_portfolio_positions_symbol ON portfolio_positions(symbol);
CREATE INDEX IF NOT EXISTS idx_trade_history_symbol ON trade_history(symbol);
CREATE INDEX IF NOT EXISTS idx_trade_history_date ON trade_history(trade_date);
CREATE INDEX IF NOT EXISTS idx_portfolio_analytics_date ON portfolio_analytics(analysis_date);
CREATE INDEX IF NOT EXISTS idx_market_data_cache_symbol ON market_data_cache(symbol);
CREATE INDEX IF NOT EXISTS idx_market_data_cache_expires ON market_data_cache(expires_at);
CREATE INDEX IF NOT EXISTS idx_agent_logs_type ON agent_logs(agent_type);
CREATE INDEX IF NOT EXISTS idx_agent_logs_created ON agent_logs(created_at);

-- Insert sample data for testing
INSERT INTO portfolio_positions (symbol, quantity, average_cost, current_value) VALUES
    ('VTI', 100.0, 220.50, 22500.00),
    ('BNDX', 200.0, 52.75, 10800.00),
    ('GSG', 50.0, 18.20, 950.00)
ON CONFLICT DO NOTHING;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO finagent;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO finagent; 