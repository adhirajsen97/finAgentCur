"""
Simplified FinAgent Main Application for Render Deployment
This version works without external PostgreSQL/Redis dependencies
"""
import os
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="FinAgent - AI Investment System",
    description="AI-powered investment analysis with Straight Arrow strategy",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class PortfolioRequest(BaseModel):
    portfolio: Dict[str, float]
    total_value: float

class AgentRequest(BaseModel):
    query: str
    symbols: Optional[list] = None
    portfolio: Optional[Dict[str, float]] = None
    portfolio_value: Optional[float] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    environment: str

# In-memory storage for demo (replace with database in production)
portfolio_cache = {}
analysis_cache = {}

# Mock services for demonstration
class MockMarketDataService:
    """Mock market data service for demonstration"""
    
    async def get_current_price(self, symbol: str) -> Dict[str, Any]:
        # Mock data for demo
        mock_prices = {
            "VTI": {"price": 225.50, "change": 0.025, "volume": 1000000},
            "BNDX": {"price": 54.20, "change": -0.01, "volume": 500000},
            "GSG": {"price": 19.10, "change": 0.03, "volume": 200000}
        }
        return mock_prices.get(symbol, {"price": 100.0, "change": 0.0, "volume": 100000})
    
    async def get_market_sentiment(self) -> Dict[str, Any]:
        return {
            "overall_sentiment": "NEUTRAL",
            "confidence": 0.65,
            "factors": ["Mixed economic indicators", "Moderate volatility"],
            "timestamp": datetime.now().isoformat()
        }

class MockAIAgent:
    """Mock AI agent for demonstration"""
    
    def __init__(self, agent_type: str):
        self.agent_type = agent_type
    
    async def analyze(self, query: str, **kwargs) -> Dict[str, Any]:
        if self.agent_type == "data_analyst":
            return {
                "analysis": f"Market analysis for query: {query}",
                "recommendations": [
                    "Current market conditions show moderate volatility",
                    "Three-fund portfolio allocation remains appropriate",
                    "Monitor bond yields for potential rebalancing opportunities"
                ],
                "confidence": 0.75,
                "timestamp": datetime.now().isoformat()
            }
        elif self.agent_type == "trading_analyst":
            return {
                "signals": [
                    {"symbol": "VTI", "action": "HOLD", "confidence": 0.8, "reason": "Strong fundamentals"},
                    {"symbol": "BNDX", "action": "HOLD", "confidence": 0.7, "reason": "Stable bond allocation"},
                    {"symbol": "GSG", "action": "MONITOR", "confidence": 0.6, "reason": "Commodity volatility"}
                ],
                "market_timing": "NEUTRAL",
                "timestamp": datetime.now().isoformat()
            }
        elif self.agent_type == "risk_analyst":
            return {
                "risk_level": "MEDIUM",
                "var_95": -0.025,
                "compliance_status": "COMPLIANT",
                "alerts": [],
                "recommendations": [
                    "Portfolio risk within acceptable parameters",
                    "Diversification adequate for Straight Arrow strategy"
                ],
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {"error": "Unknown agent type"}

# Initialize services
market_service = MockMarketDataService()
data_agent = MockAIAgent("data_analyst")
trading_agent = MockAIAgent("trading_analyst")
risk_agent = MockAIAgent("risk_analyst")

# API Routes
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint redirect to docs"""
    return {"message": "FinAgent API", "docs": "/docs", "health": "/health"}

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        environment=os.getenv("ENVIRONMENT", "development")
    )

@app.post("/api/analyze-portfolio")
async def analyze_portfolio(request: PortfolioRequest):
    """Analyze portfolio using Straight Arrow strategy"""
    try:
        portfolio = request.portfolio
        total_value = request.total_value
        
        # Calculate current weights
        current_weights = {
            symbol: value / total_value 
            for symbol, value in portfolio.items()
        }
        
        # Straight Arrow target allocation
        target_allocation = {"VTI": 0.60, "BNDX": 0.30, "GSG": 0.10}
        
        # Calculate drift
        drift_analysis = {}
        max_drift = 0
        for symbol in target_allocation:
            current_weight = current_weights.get(symbol, 0)
            target_weight = target_allocation[symbol]
            drift = current_weight - target_weight
            drift_analysis[symbol] = {
                "current_weight": current_weight,
                "target_weight": target_weight,
                "drift": drift,
                "drift_percent": abs(drift) * 100
            }
            max_drift = max(max_drift, abs(drift))
        
        # Risk assessment
        risk_assessment = {
            "overall_risk": "MEDIUM" if max_drift < 0.05 else "HIGH",
            "max_drift": max_drift,
            "needs_rebalancing": max_drift > 0.05,
            "compliance": "COMPLIANT" if max_drift < 0.10 else "NON_COMPLIANT"
        }
        
        # Rebalancing recommendation
        rebalance_needed = max_drift > 0.05
        recommendations = []
        if rebalance_needed:
            for symbol, analysis in drift_analysis.items():
                if abs(analysis["drift"]) > 0.02:
                    action = "REDUCE" if analysis["drift"] > 0 else "INCREASE"
                    recommendations.append({
                        "symbol": symbol,
                        "action": action,
                        "current_percent": analysis["current_weight"] * 100,
                        "target_percent": analysis["target_weight"] * 100
                    })
        
        return {
            "analysis": {
                "strategy": "Straight Arrow",
                "total_value": total_value,
                "current_weights": current_weights,
                "target_allocation": target_allocation,
                "drift_analysis": drift_analysis,
                "risk_assessment": risk_assessment,
                "rebalance_needed": rebalance_needed,
                "recommendations": recommendations,
                "timestamp": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Portfolio analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/market-sentiment")
async def get_market_sentiment():
    """Get current market sentiment analysis"""
    try:
        sentiment = await market_service.get_market_sentiment()
        return {"sentiment": sentiment}
    except Exception as e:
        logger.error(f"Market sentiment error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/strategy-performance")
async def get_strategy_performance(period: str = "1y"):
    """Get Straight Arrow strategy performance"""
    try:
        # Mock performance data
        performance = {
            "strategy_name": "Straight Arrow",
            "period": period,
            "total_return": 0.08,
            "annual_volatility": 0.10,
            "sharpe_ratio": 0.65,
            "max_drawdown": -0.08,
            "allocation": {"VTI": 0.60, "BNDX": 0.30, "GSG": 0.10},
            "last_updated": datetime.now().isoformat()
        }
        return {"performance": performance}
    except Exception as e:
        logger.error(f"Strategy performance error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/agents/data-analyst")
async def data_analyst_endpoint(request: AgentRequest):
    """Data analyst AI agent endpoint"""
    try:
        analysis = await data_agent.analyze(request.query, symbols=request.symbols)
        return {"analysis": analysis}
    except Exception as e:
        logger.error(f"Data analyst error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/agents/trading-analyst")
async def trading_analyst_endpoint(request: AgentRequest):
    """Trading analyst AI agent endpoint"""
    try:
        analysis = await trading_agent.analyze(request.query, portfolio=request.portfolio)
        return {"analysis": analysis}
    except Exception as e:
        logger.error(f"Trading analyst error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/agents/risk-analyst")
async def risk_analyst_endpoint(request: AgentRequest):
    """Risk analyst AI agent endpoint"""
    try:
        analysis = await risk_agent.analyze(
            request.query, 
            portfolio_value=request.portfolio_value
        )
        return {"analysis": analysis}
    except Exception as e:
        logger.error(f"Risk analyst error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main_simple:app",
        host="0.0.0.0",
        port=port,
        reload=False
    ) 