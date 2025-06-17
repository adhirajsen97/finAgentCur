# 🚀 Unified Investment Strategy API - Integration Guide

## Overview

The Unified Investment Strategy API orchestrates all existing FinAgent services to create comprehensive, actionable investment strategies with specific trade orders that your frontend can execute directly.

## Integration Steps

### 1. Add to your main FastAPI app

Add this to your `main_enhanced_complete.py`:

```python
# At the top with other imports:
from unified_strategy_api import (
    UnifiedStrategyRequest, 
    UnifiedStrategyOrchestrator, 
    create_unified_strategy_endpoint
)

# After your service initialization (around line 960):
orchestrator_service = UnifiedStrategyOrchestrator(strategy_service, market_service, ai_service)

# Add this endpoint after your existing endpoints:
@app.post("/api/unified-strategy")
async def unified_strategy(request: UnifiedStrategyRequest):
    """
    🚀 UNIFIED INVESTMENT STRATEGY API
    
    Creates comprehensive investment strategies with actionable trade orders.
    
    Returns:
    - Specific buy/sell orders with quantities and dollar amounts
    - Execution priorities and timing guidelines
    - Risk warnings and compliance information
    - Performance expectations and review dates
    """
    return await create_unified_strategy_endpoint(request, orchestrator_service)
```

### 2. Test the API

**Working curl command:**
```bash
curl -X POST "https://finagentcur.onrender.com/api/unified-strategy" \
  -H "Content-Type: application/json" \
  -d '{
    "portfolio": {"VTI": 50000.0, "BNDX": 30000.0, "GSG": 20000.0},
    "total_value": 100000.0,
    "available_cash": 10000.0,
    "time_horizon": "3 weeks",
    "risk_tolerance": "moderate",
    "investment_goals": ["rebalancing"]
  }'
```

### 3. Expected Response Structure

```json
{
  "status": "success",
  "strategy": {
    "strategy_id": "strategy_20241201_143025",
    "created_at": "2024-12-01T14:30:25.123456",
    "time_horizon": "3 weeks",
    "risk_tolerance": "moderate",
    "strategy_type": "Straight Arrow Enhanced",
    
    "market_context": {
      "current_prices": {
        "VTI": 235.29,
        "BNDX": 52.18,
        "GSG": 58.45
      },
      "market_sentiment": {
        "overall_sentiment": "NEUTRAL",
        "fear_greed_index": 65,
        "volatility_index": 18
      }
    },
    
    "portfolio_analysis": {
      "current_allocation": {
        "VTI": 0.50,
        "BNDX": 0.30,
        "GSG": 0.20
      },
      "target_allocation": {
        "VTI": 0.60,
        "BNDX": 0.30,
        "GSG": 0.10
      },
      "needs_rebalancing": true
    },
    
    "ai_insights": {
      "market_analysis": "Current market conditions suggest moderate volatility with neutral sentiment...",
      "confidence_score": 0.75
    },
    
    "trade_orders": [
      {
        "symbol": "VTI",
        "action": "BUY",
        "order_type": "MARKET",
        "quantity": 25.5,
        "dollar_amount": 6000.0,
        "current_price": 235.29,
        "priority": "HIGH",
        "reason": "Rebalance VTI: 50.0% → 60.0%",
        "expected_impact": "Brings VTI closer to target allocation"
      },
      {
        "symbol": "GSG",
        "action": "SELL",
        "order_type": "MARKET",
        "quantity": 171.4,
        "dollar_amount": 10000.0,
        "current_price": 58.45,
        "priority": "HIGH",
        "reason": "Rebalance GSG: 20.0% → 10.0%",
        "expected_impact": "Brings GSG closer to target allocation"
      }
    ],
    
    "strategy_summary": {
      "overview": "Straight Arrow strategy with 2 trades",
      "total_trades": 2,
      "buy_orders": 1,
      "sell_orders": 1,
      "total_investment": 6000.0,
      "total_divestment": 10000.0,
      "rebalancing_needed": true,
      "market_conditions": "NEUTRAL"
    },
    
    "execution_guidelines": {
      "execution_order": "Execute HIGH priority first, then MEDIUM, then LOW",
      "timing": "Spread trades over 1-3 days",
      "market_hours": "Execute during regular trading hours",
      "monitoring": "Monitor for 24-48 hours after execution",
      "high_priority_count": 2,
      "suggested_sequence": [
        "BUY VTI: $6,000",
        "SELL GSG: $10,000"
      ]
    },
    
    "risk_warnings": [
      "All investments carry risk of loss",
      "Market conditions can change rapidly",
      "This strategy may not suit all investors"
    ],
    
    "next_review_date": "2024-12-08T14:30:25.123456"
  },
  "execution_ready": true,
  "trade_count": 2,
  "disclaimer": "Educational purposes only. Not financial advice."
}
```

## Frontend Integration

### 4. Processing Trade Orders in Your Frontend

The `trade_orders` array contains everything your frontend needs to execute trades:

```javascript
// Example frontend processing
const response = await fetch('/api/unified-strategy', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    portfolio: { VTI: 50000, BNDX: 30000, GSG: 20000 },
    total_value: 100000,
    available_cash: 10000,
    time_horizon: "3 weeks",
    risk_tolerance: "moderate"
  })
});

const data = await response.json();
const tradeOrders = data.strategy.trade_orders;

// Sort by priority for execution
const sortedTrades = tradeOrders.sort((a, b) => {
  const priorityOrder = { HIGH: 3, MEDIUM: 2, LOW: 1 };
  return priorityOrder[b.priority] - priorityOrder[a.priority];
});

// Execute trades
for (const trade of sortedTrades) {
  console.log(`${trade.action} ${trade.quantity} shares of ${trade.symbol} at $${trade.current_price}`);
  console.log(`Total: $${trade.dollar_amount} - Priority: ${trade.priority}`);
  console.log(`Reason: ${trade.reason}`);
  
  // Your trading execution logic here
  // await executeTrade(trade);
}
```

### 5. Trade Execution Flow

1. **Parse trade orders** from the API response
2. **Sort by priority** (HIGH → MEDIUM → LOW)
3. **Validate orders** against current account balances
4. **Execute trades** using your broker's API
5. **Monitor execution** for 24-48 hours as recommended
6. **Review strategy** on the suggested review date

### 6. Error Handling

```javascript
try {
  const response = await fetch('/api/unified-strategy', { /* ... */ });
  
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  
  const data = await response.json();
  
  if (data.status !== 'success') {
    throw new Error('Strategy generation failed');
  }
  
  // Process trade orders
  processTrades(data.strategy.trade_orders);
  
} catch (error) {
  console.error('Strategy API error:', error);
  // Handle error appropriately in your UI
}
```

## Key Features

✅ **Actionable Trade Orders**: Specific buy/sell orders with quantities and dollar amounts
✅ **Priority System**: HIGH/MEDIUM/LOW priorities for execution order
✅ **Market Context**: Real-time prices and market sentiment
✅ **AI Insights**: Market analysis and confidence scores
✅ **Risk Management**: Warnings and compliance information
✅ **Execution Guidelines**: Timing and monitoring recommendations
✅ **Ready for Production**: Comprehensive error handling and validation

## Request Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `portfolio` | Dict[str, float] | Yes | - | Current holdings (symbol: dollar amount) |
| `total_value` | float | Yes | - | Total portfolio value |
| `available_cash` | float | No | 0.0 | Available cash for investing |
| `time_horizon` | str | No | "3 weeks" | Investment time horizon |
| `risk_tolerance` | str | No | "moderate" | Risk tolerance level |
| `investment_goals` | List[str] | No | ["rebalancing"] | Investment objectives |

## Trade Order Fields

Each trade order includes:
- `symbol`: Trading symbol (VTI, BNDX, GSG)
- `action`: BUY or SELL
- `quantity`: Number of shares
- `dollar_amount`: Total dollar amount
- `current_price`: Current market price
- `priority`: Execution priority (HIGH/MEDIUM/LOW)
- `reason`: Human-readable explanation
- `expected_impact`: Portfolio impact description

This unified API gives your frontend everything it needs to execute a complete investment strategy! 🚀 