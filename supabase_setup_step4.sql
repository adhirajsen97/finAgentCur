-- Step 4: Enable RLS and create policies
ALTER TABLE portfolio_analytics ENABLE ROW LEVEL SECURITY;
ALTER TABLE agent_logs ENABLE ROW LEVEL SECURITY;

-- Allow all operations for anonymous users (you can restrict this later)
CREATE POLICY "Allow all operations on portfolio_analytics" 
ON portfolio_analytics FOR ALL 
USING (true);

CREATE POLICY "Allow all operations on agent_logs" 
ON agent_logs FOR ALL 
USING (true); 