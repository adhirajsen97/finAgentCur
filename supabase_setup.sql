-- FinAgent Supabase Database Setup
-- Run this in your Supabase SQL Editor

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Portfolio analytics table
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

-- Agent logs table
CREATE TABLE IF NOT EXISTS agent_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_type VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'COMPLETED',
    input_data JSONB,
    output_data JSONB,
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Portfolio positions table (optional - for tracking actual positions)
CREATE TABLE IF NOT EXISTS portfolio_positions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(10) NOT NULL,
    quantity DECIMAL(15,6) NOT NULL,
    average_cost DECIMAL(15,2) NOT NULL,
    current_value DECIMAL(15,2),
    last_updated TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Market data cache table (optional - for caching market data)
CREATE TABLE IF NOT EXISTS market_data_cache (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(10) NOT NULL,
    data_type VARCHAR(50) NOT NULL,
    data JSONB NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(symbol, data_type)
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_portfolio_analytics_date ON portfolio_analytics(analysis_date);
CREATE INDEX IF NOT EXISTS idx_agent_logs_type ON agent_logs(agent_type);
CREATE INDEX IF NOT EXISTS idx_agent_logs_created ON agent_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_portfolio_positions_symbol ON portfolio_positions(symbol);
CREATE INDEX IF NOT EXISTS idx_market_data_cache_symbol ON market_data_cache(symbol);
CREATE INDEX IF NOT EXISTS idx_market_data_cache_expires ON market_data_cache(expires_at);

-- Insert sample data for testing (optional)
INSERT INTO portfolio_positions (symbol, quantity, average_cost, current_value) VALUES
    ('VTI', 100.0, 220.50, 22500.00),
    ('BNDX', 200.0, 52.75, 10800.00),
    ('GSG', 50.0, 18.20, 950.00)
ON CONFLICT DO NOTHING;

-- Row Level Security (RLS) policies
-- Enable RLS on tables
ALTER TABLE portfolio_analytics ENABLE ROW LEVEL SECURITY;
ALTER TABLE agent_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE portfolio_positions ENABLE ROW LEVEL SECURITY;
ALTER TABLE market_data_cache ENABLE ROW LEVEL SECURITY;

-- Create policies (allow all operations for now - customize as needed)
CREATE POLICY "Allow all operations on portfolio_analytics" ON portfolio_analytics FOR ALL USING (true);
CREATE POLICY "Allow all operations on agent_logs" ON agent_logs FOR ALL USING (true);
CREATE POLICY "Allow all operations on portfolio_positions" ON portfolio_positions FOR ALL USING (true);
CREATE POLICY "Allow all operations on market_data_cache" ON market_data_cache FOR ALL USING (true); 