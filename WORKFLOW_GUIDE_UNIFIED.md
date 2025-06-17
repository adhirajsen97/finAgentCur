# 🚀 FinAgent Unified Strategy Workflow Guide

## Overview

This guide demonstrates the complete workflow for using the FinAgent unified strategy system to create and execute investment strategies. The unified API orchestrates all services to provide actionable trade orders.

---

## 🎯 Complete Investment Strategy Workflow

### 1. Health Check & Feature Verification

First, verify all system components are operational:

```bash
curl -X GET "https://finagentcur.onrender.com/health"
```

**Expected Response:**
```json
{
  "status": "healthy",
  "features": {
    "database": "supabase",
    "market_data": "alpha_vantage",
    "ai_service": "openai",
    "agents": ["data_analyst", "risk_analyst", "trading_analyst"],
    "compliance": "enabled",
    "risk_metrics": "enabled"
  }
}
```

### 2. 🚀 Create Unified Investment Strategy

**The main endpoint that orchestrates everything:**

```bash
curl -X POST "https://finagentcur.onrender.com/api/unified-strategy" \
  -H "Content-Type: application/json" \
  -d '{
    "portfolio": {"VTI": 50000.0, "BNDX": 30000.0, "GSG": 20000.0},
    "total_value": 100000.0,
    "available_cash": 10000.0,
    "time_horizon": "3 weeks",
    "risk_tolerance": "moderate",
    "investment_goals": ["rebalancing", "growth"]
  }'
```

**This single call:**
- ✅ Analyzes your current portfolio
- ✅ Gets real-time market data
- ✅ Runs AI analysis for market insights
- ✅ Assesses risk and compliance
- ✅ Generates specific trade orders
- ✅ Provides execution guidelines

**Response includes actionable trade orders:**
```json
{
  "status": "success",
  "strategy": {
    "strategy_id": "strategy_20241201_143025",
    "trade_orders": [
      {
        "symbol": "VTI",
        "action": "BUY",
        "quantity": 25.5,
        "dollar_amount": 6000.0,
        "current_price": 235.29,
        "priority": "HIGH",
        "reason": "Rebalance VTI: 50.0% → 60.0%",
        "timing_suggestion": "Execute soon - favorable conditions"
      }
    ],
    "execution_guidelines": {
      "execution_order": "Execute HIGH priority first",
      "timing": "Spread trades over 1-3 days",
      "suggested_sequence": ["BUY VTI: $6,000"]
    }
  }
}
```

### 3. Individual Service Analysis (Optional)

You can still call individual services for deeper analysis:

#### Portfolio Analysis
```bash
curl -X POST "https://finagentcur.onrender.com/api/analyze-portfolio" \
  -H "Content-Type: application/json" \
  -d '{"portfolio": {"VTI": 50000.0, "BNDX": 30000.0, "GSG": 20000.0}, "total_value": 100000.0}'
```

#### Market Data with Technical Analysis
```bash
curl -X POST "https://finagentcur.onrender.com/api/market-data" \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["VTI", "BNDX", "GSG"]}'
```

#### AI Market Analysis
```bash
curl -X POST "https://finagentcur.onrender.com/api/agents/data-analyst" \
  -H "Content-Type: application/json" \
  -d '{"query": "Analyze market conditions for 3-week investment horizon", "symbols": ["VTI", "BNDX", "GSG"]}'
```

---

## 💰 Trade Execution Workflow

### 4. Execute Individual Trade Order

```bash
curl -X POST "https://finagentcur.onrender.com/api/execute-trade" \
  -H "Content-Type: application/json" \
  -d '{
    "strategy_id": "strategy_20241201_143025",
    "trade_order": {
      "symbol": "VTI",
      "action": "BUY",
      "quantity": 25.5,
      "dollar_amount": 6000.0,
      "current_price": 235.29,
      "order_type": "MARKET",
      "priority": "HIGH"
    },
    "execution_method": "SIMULATED",
    "notes": "Rebalancing to target allocation"
  }'
```

### 5. Execute Complete Strategy

```bash
curl -X POST "https://finagentcur.onrender.com/api/execute-strategy" \
  -H "Content-Type: application/json" \
  -d '{
    "strategy_id": "strategy_20241201_143025",
    "execution_method": "SIMULATED",
    "execution_options": {
      "max_parallel_orders": 1,
      "delay_between_orders": 30,
      "stop_on_error": true
    }
  }'
```

---

## 📊 Portfolio Tracking Workflow

### 6. Check Current Portfolio

```bash
curl -X GET "https://finagentcur.onrender.com/api/portfolio/user123/current"
```

### 7. View Portfolio History

```bash
curl -X GET "https://finagentcur.onrender.com/api/portfolio-history"
```

---

## 🧪 Python Integration Example

```python
import httpx
import asyncio
from datetime import datetime

class FinAgentClient:
    def __init__(self, base_url="https://finagentcur.onrender.com"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=60.0)
    
    async def create_investment_strategy(self, portfolio, total_value, available_cash=0, 
                                       time_horizon="3 weeks", risk_tolerance="moderate"):
        """Create a complete investment strategy"""
        response = await self.client.post(
            f"{self.base_url}/api/unified-strategy",
            json={
                "portfolio": portfolio,
                "total_value": total_value,
                "available_cash": available_cash,
                "time_horizon": time_horizon,
                "risk_tolerance": risk_tolerance,
                "investment_goals": ["rebalancing"]
            }
        )
        return response.json()
    
    async def execute_strategy(self, strategy_id, execution_method="SIMULATED"):
        """Execute all trades from a strategy"""
        response = await self.client.post(
            f"{self.base_url}/api/execute-strategy",
            json={
                "strategy_id": strategy_id,
                "execution_method": execution_method,
                "execution_options": {
                    "stop_on_error": True,
                    "delay_between_orders": 30
                }
            }
        )
        return response.json()
    
    async def complete_workflow(self, portfolio, total_value, available_cash=0):
        """Complete workflow: Strategy creation + execution"""
        print(f"🚀 Creating strategy for ${total_value:,.2f} portfolio...")
        
        # Step 1: Create strategy
        strategy_result = await self.create_investment_strategy(
            portfolio, total_value, available_cash
        )
        
        if strategy_result["status"] != "success":
            raise Exception(f"Strategy creation failed: {strategy_result}")
        
        strategy = strategy_result["strategy"]
        trade_orders = strategy["trade_orders"]
        
        print(f"✅ Strategy created: {strategy['strategy_id']}")
        print(f"📋 Trade orders: {len(trade_orders)}")
        
        # Display trade orders
        for i, order in enumerate(trade_orders, 1):
            print(f"\n{i}. {order['action']} {order['symbol']}")
            print(f"   Amount: ${order['dollar_amount']:,.2f}")
            print(f"   Quantity: {order['quantity']} shares")
            print(f"   Priority: {order['priority']}")
            print(f"   Reason: {order['reason']}")
        
        # Step 2: Execute strategy (simulated)
        print(f"\n💰 Executing strategy...")
        execution_result = await self.execute_strategy(
            strategy["strategy_id"], "SIMULATED"
        )
        
        print(f"✅ Execution completed")
        print(f"📊 Summary: {execution_result['execution_summary']}")
        
        return {
            "strategy": strategy,
            "execution": execution_result
        }

# Usage example
async def main():
    client = FinAgentClient()
    
    # Define portfolio
    portfolio = {
        "VTI": 45000.0,   # 45% (target: 60%)
        "BNDX": 35000.0,  # 35% (target: 30%)
        "GSG": 20000.0    # 20% (target: 10%)
    }
    
    # Run complete workflow
    result = await client.complete_workflow(
        portfolio=portfolio,
        total_value=100000.0,
        available_cash=15000.0
    )
    
    print(f"\n🎉 Workflow completed successfully!")
    print(f"Strategy ID: {result['strategy']['strategy_id']}")

# Run the workflow
# asyncio.run(main())
```

---

## 📈 Frontend Integration Patterns

### React Integration Example

```javascript
import React, { useState } from 'react';

const InvestmentStrategyCreator = () => {
  const [portfolio, setPortfolio] = useState({
    VTI: 50000,
    BNDX: 30000,
    GSG: 20000
  });
  const [strategy, setStrategy] = useState(null);
  const [loading, setLoading] = useState(false);

  const createStrategy = async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/unified-strategy', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          portfolio,
          total_value: Object.values(portfolio).reduce((a, b) => a + b, 0),
          available_cash: 10000,
          time_horizon: "3 weeks",
          risk_tolerance: "moderate"
        })
      });
      
      const data = await response.json();
      setStrategy(data.strategy);
    } catch (error) {
      console.error('Strategy creation failed:', error);
    } finally {
      setLoading(false);
    }
  };

  const executeTradeOrder = async (tradeOrder) => {
    try {
      const response = await fetch('/api/execute-trade', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          strategy_id: strategy.strategy_id,
          trade_order: tradeOrder,
          execution_method: "SIMULATED"
        })
      });
      
      const result = await response.json();
      console.log('Trade executed:', result);
    } catch (error) {
      console.error('Trade execution failed:', error);
    }
  };

  return (
    <div>
      <h2>Investment Strategy Creator</h2>
      
      {/* Portfolio Input */}
      <div>
        <h3>Current Portfolio</h3>
        {Object.entries(portfolio).map(([symbol, value]) => (
          <div key={symbol}>
            <label>{symbol}: $</label>
            <input 
              type="number" 
              value={value}
              onChange={(e) => setPortfolio({
                ...portfolio,
                [symbol]: parseFloat(e.target.value)
              })}
            />
          </div>
        ))}
      </div>

      {/* Create Strategy Button */}
      <button onClick={createStrategy} disabled={loading}>
        {loading ? 'Creating Strategy...' : 'Create Investment Strategy'}
      </button>

      {/* Display Strategy Results */}
      {strategy && (
        <div>
          <h3>Strategy: {strategy.strategy_id}</h3>
          <h4>Trade Orders ({strategy.trade_orders.length})</h4>
          
          {strategy.trade_orders
            .sort((a, b) => {
              const priority = { HIGH: 3, MEDIUM: 2, LOW: 1 };
              return priority[b.priority] - priority[a.priority];
            })
            .map((order, index) => (
              <div key={index} style={{ 
                border: '1px solid #ccc', 
                margin: '10px', 
                padding: '10px',
                backgroundColor: order.priority === 'HIGH' ? '#ffe6e6' : '#f5f5f5'
              }}>
                <h5>{order.action} {order.symbol} - {order.priority} Priority</h5>
                <p>Quantity: {order.quantity} shares</p>
                <p>Amount: ${order.dollar_amount.toLocaleString()}</p>
                <p>Price: ${order.current_price}</p>
                <p>Reason: {order.reason}</p>
                
                <button onClick={() => executeTradeOrder(order)}>
                  Execute Trade
                </button>
              </div>
            ))}
        </div>
      )}
    </div>
  );
};

export default InvestmentStrategyCreator;
```

---

## 🔄 Automated Workflow Scheduling

### Daily Rebalancing Check

```python
import schedule
import time
import asyncio

async def daily_portfolio_check():
    """Run daily portfolio analysis and create strategy if needed"""
    client = FinAgentClient()
    
    # Get current portfolio (from your broker API)
    current_portfolio = {
        "VTI": 52000.0,
        "BNDX": 28000.0,
        "GSG": 20000.0
    }
    
    # Create strategy
    strategy_result = await client.create_investment_strategy(
        portfolio=current_portfolio,
        total_value=100000.0,
        time_horizon="1 week"
    )
    
    if strategy_result["status"] == "success":
        strategy = strategy_result["strategy"]
        
        # Check if rebalancing is needed
        if strategy["portfolio_analysis"]["needs_rebalancing"]:
            print(f"🚨 Rebalancing needed! {len(strategy['trade_orders'])} trades recommended")
            
            # Execute high-priority trades only
            high_priority_orders = [
                order for order in strategy["trade_orders"] 
                if order["priority"] == "HIGH"
            ]
            
            for order in high_priority_orders:
                print(f"Executing: {order['action']} {order['symbol']} - ${order['dollar_amount']:,.2f}")
                # Your trade execution logic here
        else:
            print("✅ Portfolio is well balanced")

# Schedule daily checks
schedule.every().day.at("09:30").do(lambda: asyncio.run(daily_portfolio_check()))

# Keep the scheduler running
while True:
    schedule.run_pending()
    time.sleep(60)
```

---

## 🎯 Best Practices

### 1. Strategy Creation
- ✅ Always verify market hours before creating strategies
- ✅ Include sufficient available cash for rebalancing
- ✅ Set appropriate time horizons for your investment goals
- ✅ Review AI insights and risk warnings before execution

### 2. Trade Execution
- ✅ Execute trades in priority order (HIGH → MEDIUM → LOW)
- ✅ Spread large orders over multiple days to minimize market impact
- ✅ Monitor executions for 24-48 hours after placing orders
- ✅ Use simulated execution for testing and validation

### 3. Portfolio Monitoring
- ✅ Review strategies weekly for short-term horizons
- ✅ Track execution performance vs. expected outcomes
- ✅ Maintain cash reserves for opportunities and rebalancing
- ✅ Document reasons for deviating from recommendations

### 4. Error Handling
- ✅ Always check response status before processing
- ✅ Implement retry logic for network timeouts
- ✅ Log all API calls for debugging and audit trails
- ✅ Have fallback procedures for system outages

---

## 📊 Monitoring and Analytics

### Performance Tracking

```python
async def track_strategy_performance(strategy_id):
    """Track how well a strategy performed"""
    client = FinAgentClient()
    
    # Get execution results
    execution_history = await client.get_execution_history(strategy_id)
    
    # Calculate performance metrics
    total_invested = sum(
        exec['dollar_amount'] for exec in execution_history 
        if exec['action'] == 'BUY'
    )
    
    total_divested = sum(
        exec['dollar_amount'] for exec in execution_history 
        if exec['action'] == 'SELL'
    )
    
    net_cash_flow = total_divested - total_invested
    
    print(f"Strategy Performance for {strategy_id}:")
    print(f"Total Invested: ${total_invested:,.2f}")
    print(f"Total Divested: ${total_divested:,.2f}")
    print(f"Net Cash Flow: ${net_cash_flow:,.2f}")
```

This unified workflow provides everything you need to create, execute, and monitor investment strategies using the FinAgent system! 🚀 