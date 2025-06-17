# 📋 FinAgent API Contracts

## Overview

This document defines the complete API contracts for the FinAgent investment system, including the unified strategy API, trade execution APIs, and data persistence contracts.

---

## 🚀 Unified Investment Strategy API

### `POST /api/unified-strategy`

**Purpose**: Creates a comprehensive investment strategy with actionable trade orders by orchestrating all system services.

#### Request Contract

```typescript
interface UnifiedStrategyRequest {
  portfolio: Record<string, number>;        // Current holdings: symbol -> dollar value
  total_value: number;                      // Total portfolio value (must be > 0)
  available_cash?: number;                  // Available cash for investing (default: 0.0)
  time_horizon?: string;                    // Investment time horizon (default: "3 weeks")
  risk_tolerance?: "conservative" | "moderate" | "aggressive";  // Risk tolerance (default: "moderate")
  investment_goals?: string[];              // Investment objectives (default: ["rebalancing"])
}
```

**Example Request**:
```json
{
  "portfolio": {
    "VTI": 50000.0,
    "BNDX": 30000.0,
    "GSG": 20000.0
  },
  "total_value": 100000.0,
  "available_cash": 10000.0,
  "time_horizon": "3 weeks",
  "risk_tolerance": "moderate",
  "investment_goals": ["rebalancing", "growth"]
}
```

#### Response Contract

```typescript
interface UnifiedStrategyResponse {
  status: "success" | "error";
  strategy: InvestmentStrategy;
  execution_ready: boolean;
  trade_count: number;
  disclaimer: string;
  error?: string;                           // Present if status is "error"
}

interface InvestmentStrategy {
  strategy_id: string;                      // Unique strategy identifier
  created_at: string;                       // ISO timestamp
  time_horizon: string;
  risk_tolerance: string;
  strategy_type: "Straight Arrow Enhanced";
  
  // Market context for decision making
  market_context: {
    current_prices: Record<string, number>;
    market_sentiment: MarketSentiment;
    technical_signals?: Record<string, string>;
  };
  
  // Portfolio analysis summary
  portfolio_analysis: {
    current_allocation: Record<string, number>;
    target_allocation: Record<string, number>;
    needs_rebalancing: boolean;
    drift_summary?: Record<string, string>;
  };
  
  // AI insights and confidence
  ai_insights: {
    market_analysis: string;
    risk_assessment?: string;
    confidence_score: number;               // 0.0 - 1.0
  };
  
  // 🎯 ACTIONABLE TRADE ORDERS - Core output for execution
  trade_orders: TradeOrder[];
  
  // Strategy summary and metadata
  strategy_summary: StrategySummary;
  execution_guidelines: ExecutionGuidelines;
  risk_warnings: string[];
  next_review_date: string;                 // ISO timestamp
}

interface TradeOrder {
  symbol: string;                           // Trading symbol (VTI, BNDX, GSG)
  action: "BUY" | "SELL" | "HOLD";
  order_type: "MARKET" | "LIMIT" | "STOP";
  quantity: number;                         // Number of shares
  dollar_amount: number;                    // Total dollar amount
  current_price: number;                    // Current market price per share
  target_price?: number;                    // Target price for limit orders
  priority: "HIGH" | "MEDIUM" | "LOW";
  reason: string;                           // Human-readable explanation
  expected_impact: string;                  // Expected portfolio impact
  technical_context?: string;               // Technical analysis context
  timing_suggestion?: string;               // Timing recommendation
}

interface StrategySummary {
  overview: string;
  total_trades: number;
  buy_orders: number;
  sell_orders: number;
  total_investment: number;
  total_divestment: number;
  rebalancing_needed: boolean;
  market_conditions: string;
  strategy_confidence?: "HIGH" | "MEDIUM" | "LOW";
}

interface ExecutionGuidelines {
  execution_order: string;
  timing: string;
  market_hours: string;
  monitoring: string;
  high_priority_count: number;
  suggested_sequence: string[];
}

interface MarketSentiment {
  overall_sentiment: "BULLISH" | "BEARISH" | "NEUTRAL";
  fear_greed_index?: number;                // 0-100
  volatility_index?: number;                // VIX-style index
}
```

**Success Response Example**:
```json
{
  "status": "success",
  "strategy": {
    "strategy_id": "strategy_20241201_143025",
    "created_at": "2024-12-01T14:30:25.123456Z",
    "time_horizon": "3 weeks",
    "risk_tolerance": "moderate",
    "strategy_type": "Straight Arrow Enhanced",
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
        "expected_impact": "Brings VTI closer to target allocation",
        "timing_suggestion": "Execute soon - favorable conditions"
      }
    ],
    "execution_guidelines": {
      "execution_order": "Execute HIGH priority first, then MEDIUM, then LOW",
      "timing": "Spread trades over 1-3 days",
      "high_priority_count": 1,
      "suggested_sequence": ["BUY VTI: $6,000"]
    }
  },
  "execution_ready": true,
  "trade_count": 1,
  "disclaimer": "Educational purposes only. Not financial advice."
}
```

---

## 💰 Trade Execution APIs

### `POST /api/execute-trade`

**Purpose**: Execute a single trade order with tracking and persistence.

#### Request Contract

```typescript
interface TradeExecutionRequest {
  strategy_id: string;                      // Reference to originating strategy
  trade_order: TradeOrder;                  // The trade order to execute
  execution_method: "SIMULATED" | "PAPER" | "LIVE";
  user_id?: string;                         // Optional user identifier
  notes?: string;                           // Optional execution notes
}
```

#### Response Contract

```typescript
interface TradeExecutionResponse {
  status: "success" | "failed" | "pending";
  execution_id: string;                     // Unique execution identifier
  trade_order: TradeOrder;
  execution_details: {
    executed_at: string;                    // ISO timestamp
    executed_price: number;                 // Actual execution price
    executed_quantity: number;              // Actual shares executed
    execution_method: string;
    commission?: number;                    // Commission paid
    slippage?: number;                      // Price slippage
    error_message?: string;                 // If execution failed
  };
  portfolio_impact: {
    new_position: number;                   // New position size
    cash_impact: number;                    // Cash flow impact
    new_allocation: Record<string, number>; // Updated allocation percentages
  };
}
```

### `POST /api/execute-strategy`

**Purpose**: Execute all trade orders from a strategy in priority order.

#### Request Contract

```typescript
interface StrategyExecutionRequest {
  strategy_id: string;
  execution_method: "SIMULATED" | "PAPER" | "LIVE";
  execution_options?: {
    max_parallel_orders?: number;           // Max simultaneous executions (default: 1)
    delay_between_orders?: number;          // Seconds between orders (default: 0)
    stop_on_error?: boolean;                // Stop if any order fails (default: true)
    partial_execution?: boolean;            // Allow partial fills (default: true)
  };
  user_id?: string;
  notes?: string;
}
```

#### Response Contract

```typescript
interface StrategyExecutionResponse {
  status: "success" | "partial" | "failed";
  strategy_execution_id: string;
  strategy_id: string;
  execution_summary: {
    total_orders: number;
    executed_orders: number;
    failed_orders: number;
    total_invested: number;
    total_divested: number;
    execution_time_ms: number;
  };
  order_results: TradeExecutionResponse[];
  final_portfolio: {
    positions: Record<string, number>;
    cash_balance: number;
    total_value: number;
    allocation: Record<string, number>;
  };
}
```

---

## 📊 Portfolio Tracking APIs

### `GET /api/portfolio/{user_id}/current`

**Purpose**: Get current portfolio state with real-time valuations.

#### Response Contract

```typescript
interface CurrentPortfolioResponse {
  user_id: string;
  portfolio: {
    positions: Record<string, PositionDetails>;
    cash_balance: number;
    total_value: number;
    last_updated: string;
  };
  performance: {
    today_change: number;
    today_change_percent: number;
    total_return: number;
    total_return_percent: number;
  };
  allocation: Record<string, number>;
}

interface PositionDetails {
  shares: number;
  current_price: number;
  market_value: number;
  cost_basis: number;
  unrealized_pnl: number;
  unrealized_pnl_percent: number;
}
```

### `POST /api/portfolio/{user_id}/update`

**Purpose**: Update portfolio after trade executions.

#### Request Contract

```typescript
interface PortfolioUpdateRequest {
  execution_id: string;
  symbol: string;
  action: "BUY" | "SELL";
  shares: number;
  price: number;
  commission?: number;
  timestamp: string;
}
```

---

## 📈 Market Data APIs

### `GET /api/market/quotes`

**Purpose**: Get real-time market quotes with technical analysis.

#### Response Contract

```typescript
interface MarketQuotesResponse {
  quotes: Record<string, QuoteData>;
  timestamp: string;
  data_source: "alpha_vantage" | "mock";
}

interface QuoteData {
  symbol: string;
  price: number;
  change: number;
  change_percent: string;
  volume?: number;
  high?: number;
  low?: number;
  technical_analysis?: {
    trend: "BULLISH" | "BEARISH" | "NEUTRAL";
    recommendation: "BUY" | "SELL" | "HOLD";
    support?: number;
    resistance?: number;
  };
}
```

---

## 💾 Data Persistence Contracts

### Investment Strategies Table

```sql
CREATE TABLE investment_strategies (
    id BIGSERIAL PRIMARY KEY,
    strategy_id VARCHAR(50) UNIQUE NOT NULL,
    user_id VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    strategy_type VARCHAR(50) NOT NULL,
    time_horizon VARCHAR(50),
    risk_tolerance VARCHAR(20),
    investment_goals JSONB,
    portfolio_snapshot JSONB NOT NULL,
    market_context JSONB,
    ai_insights JSONB,
    trade_orders JSONB NOT NULL,
    execution_guidelines JSONB,
    strategy_summary JSONB,
    status VARCHAR(20) DEFAULT 'created',
    executed_at TIMESTAMPTZ,
    performance_metrics JSONB
);
```

### Trade Executions Table

```sql
CREATE TABLE trade_executions (
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
    execution_method VARCHAR(20),
    status VARCHAR(20) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    executed_at TIMESTAMPTZ,
    error_message TEXT,
    slippage DECIMAL(10,4),
    portfolio_impact JSONB
);
```

### Portfolio Positions Table

```sql
CREATE TABLE portfolio_positions (
    id BIGSERIAL PRIMARY KEY,
    user_id VARCHAR(50) NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    shares DECIMAL(15,4) NOT NULL DEFAULT 0,
    average_cost DECIMAL(10,4) NOT NULL DEFAULT 0,
    last_updated TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(user_id, symbol)
);
```

### Portfolio History Table

```sql
CREATE TABLE portfolio_history (
    id BIGSERIAL PRIMARY KEY,
    user_id VARCHAR(50) NOT NULL,
    snapshot_date DATE NOT NULL,
    total_value DECIMAL(15,2),
    cash_balance DECIMAL(15,2),
    positions JSONB,
    allocation JSONB,
    performance_metrics JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(user_id, snapshot_date)
);
```

---

## 🔐 Authentication & Authorization

### Headers Required

```typescript
interface APIHeaders {
  'Content-Type': 'application/json';
  'Authorization'?: 'Bearer <jwt_token>';  // For user-specific operations
  'X-API-Key'?: string;                    // For service-to-service calls
}
```

---

## ⚠️ Error Response Contract

All APIs return standardized error responses:

```typescript
interface ErrorResponse {
  status: "error";
  error: {
    code: string;                          // Machine-readable error code
    message: string;                       // Human-readable error message
    details?: any;                         // Additional error context
    timestamp: string;                     // ISO timestamp
    request_id?: string;                   // For tracking/debugging
  };
}
```

**Common Error Codes**:
- `INVALID_REQUEST` - Request validation failed
- `INSUFFICIENT_FUNDS` - Not enough cash for trade
- `MARKET_CLOSED` - Market is closed for trading
- `SYMBOL_NOT_SUPPORTED` - Symbol not supported
- `STRATEGY_NOT_FOUND` - Strategy ID not found
- `EXECUTION_FAILED` - Trade execution failed
- `RATE_LIMITED` - Too many requests

---

## 🧪 Testing Contracts

### Test Data Sets

```typescript
// Standard test portfolio
const TEST_PORTFOLIO = {
  "VTI": 50000.0,
  "BNDX": 30000.0,
  "GSG": 20000.0
};

// Test strategy request
const TEST_STRATEGY_REQUEST = {
  portfolio: TEST_PORTFOLIO,
  total_value: 100000.0,
  available_cash: 10000.0,
  time_horizon: "3 weeks",
  risk_tolerance: "moderate",
  investment_goals: ["rebalancing"]
};
```

This comprehensive API contract documentation ensures consistent integration and implementation across all system components. All APIs follow REST principles with JSON payloads and standardized error handling. 