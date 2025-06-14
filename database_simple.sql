-- Simplified FinAgent Database Schema
-- Supabase PostgreSQL Schema

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Portfolio Analytics Table (simplified)
CREATE TABLE IF NOT EXISTS portfolio_analytics (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    analysis_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    total_value DECIMAL(15,2) NOT NULL,
    allocation JSONB NOT NULL,
    drift_analysis JSONB NOT NULL,
    risk_assessment JSONB NOT NULL,
    rebalance_recommendation JSONB DEFAULT '[]',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Market Data Cache Table (optional)
CREATE TABLE IF NOT EXISTS market_data_cache (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    data_type VARCHAR(50) NOT NULL,
    data JSONB NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(symbol, data_type)
);

-- Indexes for better performance
CREATE INDEX IF NOT EXISTS idx_portfolio_analytics_date ON portfolio_analytics(analysis_date);
CREATE INDEX IF NOT EXISTS idx_market_data_symbol ON market_data_cache(symbol);
CREATE INDEX IF NOT EXISTS idx_market_data_expires ON market_data_cache(expires_at);

-- Function to clean up expired market data
CREATE OR REPLACE FUNCTION cleanup_expired_market_data()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM market_data_cache WHERE expires_at < NOW();
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE 'plpgsql';

-- Insert sample data for testing (optional)
INSERT INTO portfolio_analytics (total_value, allocation, drift_analysis, risk_assessment, rebalance_recommendation) 
VALUES 
    (50000, 
     '{"VTI": 0.5, "BNDX": 0.3, "GSG": 0.2}', 
     '{"VTI": {"current_weight": 0.5, "target_weight": 0.6, "drift": -0.1}, "BNDX": {"current_weight": 0.3, "target_weight": 0.3, "drift": 0.0}, "GSG": {"current_weight": 0.2, "target_weight": 0.1, "drift": 0.1}}',
     '{"overall_risk": "MODERATE", "max_drift": 0.1, "needs_rebalancing": true}',
     '[{"symbol": "VTI", "action": "INCREASE", "priority": "HIGH"}, {"symbol": "GSG", "action": "REDUCE", "priority": "HIGH"}]')
ON CONFLICT DO NOTHING; 