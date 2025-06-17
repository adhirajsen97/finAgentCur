"""
Unified Investment Strategy API
==============================

This module provides a unified API endpoint that orchestrates all existing FinAgent services
to create comprehensive, actionable investment strategies with specific trade orders.

Usage:
    POST /api/unified-strategy
    
Example Request:
{
    "portfolio": {"VTI": 50000.0, "BNDX": 30000.0, "GSG": 20000.0},
    "total_value": 100000.0,
    "available_cash": 10000.0,
    "time_horizon": "3 weeks",
    "risk_tolerance": "moderate",
    "investment_goals": ["rebalancing", "growth"]
}

Example Response:
{
    "status": "success",
    "strategy": {
        "strategy_id": "strategy_20241201_143025",
        "trade_orders": [
            {
                "symbol": "VTI",
                "action": "BUY",
                "dollar_amount": 5000.0,
                "quantity": 20.5,
                "priority": "HIGH",
                "reason": "Rebalance to target allocation: 50.0% → 60.0%"
            }
        ],
        "execution_guidelines": {...},
        "risk_warnings": [...],
        "performance_expectations": {...}
    }
}
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import asyncio
import logging
from fastapi import HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ============================================================================
# PYDANTIC MODELS FOR UNIFIED STRATEGY
# ============================================================================

class UnifiedStrategyRequest(BaseModel):
    """Unified investment strategy request"""
    portfolio: Dict[str, float] = Field(..., description="Current portfolio holdings")
    total_value: float = Field(..., gt=0, description="Total portfolio value")
    available_cash: Optional[float] = Field(default=0.0, description="Available cash for investing")
    time_horizon: Optional[str] = Field(default="3 weeks", description="Investment time horizon")
    risk_tolerance: Optional[str] = Field(default="moderate", description="Risk tolerance")
    investment_goals: Optional[List[str]] = Field(default=["rebalancing"], description="Investment goals")

class TradeOrder(BaseModel):
    """Individual trade order"""
    symbol: str = Field(..., description="Trading symbol")
    action: str = Field(..., description="BUY, SELL, or HOLD")
    order_type: str = Field(default="MARKET", description="MARKET, LIMIT, etc.")
    quantity: Optional[float] = Field(default=None, description="Number of shares")
    dollar_amount: Optional[float] = Field(default=None, description="Dollar amount to invest")
    current_price: Optional[float] = Field(default=None, description="Current market price")
    target_price: Optional[float] = Field(default=None, description="Target price for limit orders")
    priority: str = Field(default="MEDIUM", description="HIGH, MEDIUM, LOW")
    reason: str = Field(..., description="Reason for this trade")
    expected_impact: str = Field(..., description="Expected impact on portfolio")
    technical_context: Optional[str] = Field(default="", description="Technical analysis context")
    timing_suggestion: Optional[str] = Field(default="", description="Timing suggestion")

# ============================================================================
# UNIFIED STRATEGY ORCHESTRATOR
# ============================================================================

class UnifiedStrategyOrchestrator:
    """Orchestrates all services to create comprehensive investment strategies"""
    
    def __init__(self, strategy_service, market_service, ai_service):
        self.strategy_service = strategy_service
        self.market_service = market_service
        self.ai_service = ai_service
        self.straight_arrow_symbols = ["VTI", "BNDX", "GSG"]
    
    async def create_investment_strategy(self, request: UnifiedStrategyRequest) -> Dict[str, Any]:
        """Create a comprehensive investment strategy with actionable trade orders"""
        try:
            logger.info(f"Creating strategy for ${request.total_value:,.2f} portfolio")
            
            # Get market data and portfolio analysis
            market_data = await self.market_service.get_quotes(self.straight_arrow_symbols)
            portfolio_analysis = self.strategy_service.analyze_portfolio(request.portfolio, request.total_value)
            market_sentiment = await self.market_service.get_market_sentiment()
            
            # Get AI insights
            ai_analysis = await self.ai_service.data_analyst(
                f"Analyze market for {request.time_horizon} with {request.risk_tolerance} risk tolerance", 
                self.straight_arrow_symbols
            )
            
            # Generate trade orders
            trade_orders = self._generate_trade_orders(
                portfolio_analysis, market_data, request.available_cash, request.total_value, market_sentiment
            )
            
            strategy = {
                "strategy_id": f"strategy_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "created_at": datetime.now().isoformat(),
                "time_horizon": request.time_horizon,
                "risk_tolerance": request.risk_tolerance,
                "strategy_type": "Straight Arrow Enhanced",
                "market_context": {
                    "current_prices": {symbol: data.get("price", 0) for symbol, data in market_data.items()},
                    "market_sentiment": market_sentiment.get("sentiment", {})
                },
                "portfolio_analysis": {
                    "current_allocation": portfolio_analysis["current_weights"],
                    "target_allocation": portfolio_analysis["target_allocation"],
                    "needs_rebalancing": portfolio_analysis["risk_assessment"]["needs_rebalancing"]
                },
                "ai_insights": {
                    "market_analysis": ai_analysis.get("analysis", "")[:500],
                    "confidence_score": ai_analysis.get("confidence", 0.5)
                },
                "trade_orders": trade_orders,
                "strategy_summary": self._create_strategy_summary(portfolio_analysis, trade_orders, market_sentiment),
                "execution_guidelines": self._create_execution_guidelines(trade_orders),
                "risk_warnings": self._create_risk_warnings(portfolio_analysis, market_sentiment),
                "next_review_date": self._calculate_next_review_date(request.time_horizon)
            }
            
            return strategy
            
        except Exception as e:
            logger.error(f"Strategy creation error: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to create strategy: {str(e)}")
    
    def _generate_trade_orders(self, portfolio_analysis, market_data, available_cash, total_value, market_sentiment):
        """Generate specific trade orders"""
        trade_orders = []
        current_prices = {symbol: data.get("price", 0) for symbol, data in market_data.items()}
        
        # Process rebalancing recommendations
        recommendations = portfolio_analysis.get("recommendations", [])
        for rec in recommendations:
            symbol = rec["symbol"]
            current_percent = rec["current_percent"]
            target_percent = rec["target_percent"]
            
            if symbol not in current_prices or current_prices[symbol] <= 0:
                continue
                
            current_price = current_prices[symbol]
            current_value = (current_percent / 100) * total_value
            target_value = (target_percent / 100) * total_value
            difference_value = target_value - current_value
            
            if abs(difference_value) > 1000:  # $1000 minimum
                action = "BUY" if difference_value > 0 else "SELL"
                dollar_amount = min(abs(difference_value), available_cash) if action == "BUY" else abs(difference_value)
                quantity = dollar_amount / current_price
                
                trade_order = {
                    "symbol": symbol,
                    "action": action,
                    "order_type": "MARKET",
                    "quantity": round(quantity, 2),
                    "dollar_amount": round(dollar_amount, 2),
                    "current_price": current_price,
                    "priority": rec["priority"],
                    "reason": f"Rebalance {symbol}: {current_percent:.1f}% → {target_percent:.1f}%",
                    "expected_impact": f"Brings {symbol} closer to target allocation"
                }
                trade_orders.append(trade_order)
        
        # Add cash investment orders if available cash > $5000
        if available_cash > 5000:
            target_allocation = {"VTI": 0.60, "BNDX": 0.30, "GSG": 0.10}
            for symbol, weight in target_allocation.items():
                if symbol in current_prices and current_prices[symbol] > 0:
                    investment = available_cash * weight
                    if investment >= 500:
                        trade_orders.append({
                            "symbol": symbol,
                            "action": "BUY",
                            "order_type": "MARKET",
                            "quantity": round(investment / current_prices[symbol], 2),
                            "dollar_amount": round(investment, 2),
                            "current_price": current_prices[symbol],
                            "priority": "MEDIUM",
                            "reason": f"Deploy cash: {weight:.0%} allocation",
                            "expected_impact": f"Increases {symbol} position"
                        })
        
        return trade_orders
    
    def _create_strategy_summary(self, portfolio_analysis, trade_orders, market_sentiment):
        """Create strategy summary"""
        buy_orders = [o for o in trade_orders if o["action"] == "BUY"]
        sell_orders = [o for o in trade_orders if o["action"] == "SELL"]
        
        return {
            "overview": f"Straight Arrow strategy with {len(trade_orders)} trades",
            "total_trades": len(trade_orders),
            "buy_orders": len(buy_orders),
            "sell_orders": len(sell_orders),
            "total_investment": sum(o["dollar_amount"] for o in buy_orders),
            "total_divestment": sum(o["dollar_amount"] for o in sell_orders),
            "rebalancing_needed": portfolio_analysis["risk_assessment"]["needs_rebalancing"],
            "market_conditions": market_sentiment.get("sentiment", {}).get("overall_sentiment", "NEUTRAL")
        }
    
    def _create_execution_guidelines(self, trade_orders):
        """Create execution guidelines"""
        high_priority = [o for o in trade_orders if o["priority"] == "HIGH"]
        
        return {
            "execution_order": "Execute HIGH priority first, then MEDIUM, then LOW",
            "timing": "Spread trades over 1-3 days",
            "market_hours": "Execute during regular trading hours",
            "monitoring": "Monitor for 24-48 hours after execution",
            "high_priority_count": len(high_priority),
            "suggested_sequence": [
                f"{o['action']} {o['symbol']}: ${o['dollar_amount']:,.0f}" 
                for o in sorted(trade_orders, key=lambda x: {"HIGH": 3, "MEDIUM": 2, "LOW": 1}[x["priority"]], reverse=True)[:3]
            ]
        }
    
    def _create_risk_warnings(self, portfolio_analysis, market_sentiment):
        """Create risk warnings"""
        warnings = [
            "All investments carry risk of loss",
            "Market conditions can change rapidly", 
            "This strategy may not suit all investors"
        ]
        
        if portfolio_analysis.get("risk_assessment", {}).get("overall_risk") == "HIGH":
            warnings.append("Portfolio shows HIGH risk levels")
        
        if market_sentiment.get("sentiment", {}).get("overall_sentiment") == "BEARISH":
            warnings.append("Current market sentiment is bearish")
        
        return warnings
    
    def _calculate_next_review_date(self, time_horizon):
        """Calculate next review date"""
        days = 7 if "week" in time_horizon.lower() else 14
        return (datetime.now() + timedelta(days=days)).isoformat()

# ============================================================================
# FASTAPI ENDPOINT FUNCTION
# ============================================================================

async def create_unified_strategy_endpoint(request: UnifiedStrategyRequest, orchestrator_service):
    """Unified strategy API endpoint"""
    try:
        strategy = await orchestrator_service.create_investment_strategy(request)
        
        return {
            "status": "success",
            "strategy": strategy,
            "execution_ready": True,
            "trade_count": len(strategy.get("trade_orders", [])),
            "disclaimer": "Educational purposes only. Not financial advice."
        }
        
    except Exception as e:
        logger.error(f"Unified strategy error: {e}")
        raise HTTPException(status_code=500, detail=f"Strategy creation failed: {str(e)}")

# ============================================================================
# EXAMPLE USAGE AND INTEGRATION
# ============================================================================

"""
To integrate this into your main FastAPI app, add this to main_enhanced_complete.py:

# At the top with other imports:
from unified_strategy_api import (
    UnifiedStrategyRequest, 
    UnifiedStrategyOrchestrator, 
    create_unified_strategy_endpoint
)

# After your service initialization:
orchestrator_service = UnifiedStrategyOrchestrator(strategy_service, market_service, ai_service)

# Add this endpoint:
@app.post("/api/unified-strategy")
async def unified_strategy(request: UnifiedStrategyRequest):
    return await create_unified_strategy_endpoint(request, orchestrator_service)

# Example curl command:
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

The response will include a "trade_orders" array that your frontend can use directly to execute trades:
{
  "status": "success",
  "strategy": {
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
      "execution_order": "Execute HIGH priority trades first...",
      "timing": "Spread trades over 1-3 days...",
      "suggested_sequence": ["BUY VTI: $6,000", "SELL BNDX: $3,000"]
    }
  }
}
""" 