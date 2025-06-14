"""
FinAgent - Simplified AI Investment System
Straight Arrow Strategy with Basic Enhancements
"""

import os
import asyncio
import aiohttp
from datetime import datetime
from typing import Dict, Any, Optional, List
import logging
import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from supabase import create_client, Client
import httpx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="FinAgent - Simplified AI Investment System",
    description="AI-powered investment analysis with Straight Arrow strategy",
    version="1.5.0",
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

# Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

# Initialize Supabase client
supabase: Client = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Supabase client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {e}")

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class PortfolioRequest(BaseModel):
    """Portfolio analysis request"""
    portfolio: Dict[str, float] = Field(..., description="Portfolio holdings")
    total_value: float = Field(..., gt=0, description="Total portfolio value")

class MarketDataRequest(BaseModel):
    """Market data request"""
    symbols: List[str] = Field(..., description="List of stock symbols")

class AIAnalysisRequest(BaseModel):
    """AI analysis request"""
    query: str = Field(..., description="User query")
    symbols: Optional[List[str]] = Field(default=[], description="Relevant symbols")

# ============================================================================
# STRAIGHT ARROW STRATEGY
# ============================================================================

class StraightArrowStrategy:
    """Simplified Straight Arrow investment strategy"""
    
    def __init__(self):
        # Fixed Straight Arrow allocation
        self.target_allocation = {
            "VTI": 0.60,   # 60% Total Stock Market
            "BNDX": 0.30,  # 30% International Bonds  
            "GSG": 0.10    # 10% Commodities
        }
    
    def analyze_portfolio(self, portfolio: Dict[str, float], total_value: float) -> Dict[str, Any]:
        """Analyze portfolio against Straight Arrow strategy"""
        
        # Calculate current weights
        current_weights = {
            symbol: value / total_value 
            for symbol, value in portfolio.items()
        }
        
        # Calculate drift analysis
        drift_analysis = {}
        max_drift = 0
        
        for symbol in set(list(current_weights.keys()) + list(self.target_allocation.keys())):
            current_weight = current_weights.get(symbol, 0)
            target_weight = self.target_allocation.get(symbol, 0)
            drift = current_weight - target_weight
            drift_percent = abs(drift) * 100
            
            drift_analysis[symbol] = {
                "current_weight": current_weight,
                "target_weight": target_weight,
                "drift": drift,
                "drift_percent": drift_percent
            }
            
            max_drift = max(max_drift, abs(drift))
        
        # Risk assessment
        risk_level = "LOW" if max_drift < 0.05 else "MODERATE" if max_drift < 0.1 else "HIGH"
        needs_rebalancing = max_drift > 0.05
        
        # Generate recommendations
        recommendations = []
        if needs_rebalancing:
            for symbol, analysis in drift_analysis.items():
                if abs(analysis["drift"]) > 0.02:
                    action = "REDUCE" if analysis["drift"] > 0 else "INCREASE"
                    recommendations.append({
                        "symbol": symbol,
                        "action": action,
                        "current_percent": analysis["current_weight"] * 100,
                        "target_percent": analysis["target_weight"] * 100,
                        "priority": "HIGH" if abs(analysis["drift"]) > 0.05 else "MEDIUM"
                    })
        
        return {
            "strategy": "Straight Arrow",
            "total_value": total_value,
            "current_weights": current_weights,
            "target_allocation": self.target_allocation,
            "drift_analysis": drift_analysis,
            "risk_assessment": {
                "overall_risk": risk_level,
                "max_drift": max_drift,
                "needs_rebalancing": needs_rebalancing,
                "compliance": "COMPLIANT"
            },
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat()
        }

# ============================================================================
# MARKET DATA SERVICE
# ============================================================================

class MarketDataService:
    """Simple market data service"""
    
    def __init__(self):
        self.alpha_vantage_key = ALPHA_VANTAGE_API_KEY
    
    async def get_quotes(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get quotes for symbols"""
        quotes = {}
        
        for symbol in symbols:
            try:
                if self.alpha_vantage_key:
                    quote = await self._fetch_quote_alpha_vantage(symbol)
                    if quote:
                        quotes[symbol] = quote
                        continue
                
                # Mock data fallback
                quotes[symbol] = {
                    "symbol": symbol,
                    "price": 100.0,
                    "change": 1.5,
                    "change_percent": "+1.5%",
                    "volume": 1000000,
                    "timestamp": datetime.now().isoformat(),
                    "source": "mock"
                }
                
            except Exception as e:
                logger.error(f"Failed to get quote for {symbol}: {e}")
                quotes[symbol] = {"error": str(e)}
        
        return quotes
    
    async def _fetch_quote_alpha_vantage(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch quote from Alpha Vantage API"""
        url = f"https://www.alphavantage.co/query"
        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": symbol,
            "apikey": self.alpha_vantage_key
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    data = await response.json()
                    
                    if "Global Quote" in data:
                        quote_data = data["Global Quote"]
                        return {
                            "symbol": quote_data.get("01. Symbol"),
                            "price": float(quote_data.get("05. Price", 0)),
                            "change": float(quote_data.get("09. Change", 0)),
                            "change_percent": quote_data.get("10. Change Percent", "0%"),
                            "volume": int(quote_data.get("06. Volume", 0)),
                            "timestamp": datetime.now().isoformat(),
                            "source": "alpha_vantage"
                        }
        except Exception as e:
            logger.error(f"Alpha Vantage API error for {symbol}: {e}")
        
        return None

# ============================================================================
# AI SERVICE
# ============================================================================

class AIService:
    """Simple AI service"""
    
    def __init__(self):
        self.openai_key = OPENAI_API_KEY
    
    async def analyze(self, query: str, symbols: List[str] = None) -> Dict[str, Any]:
        """Simple AI analysis"""
        
        # Build context
        context = f"User query: {query}\n"
        if symbols:
            context += f"Relevant symbols: {', '.join(symbols)}\n"
        
        context += """
        You are a financial education AI assistant. Provide helpful, educational information about investing.
        Always include appropriate disclaimers about not providing personalized investment advice.
        Focus on the Straight Arrow strategy: 60% VTI, 30% BNDX, 10% GSG.
        """
        
        # Get AI response
        if self.openai_key:
            response = await self._call_openai(context)
        else:
            response = self._generate_mock_response(query)
        
        return {
            "query": query,
            "analysis": response,
            "symbols": symbols or [],
            "confidence": 0.8 if self.openai_key else 0.6,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {self.openai_key}"},
                    json={
                        "model": "gpt-3.5-turbo",
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 500,
                        "temperature": 0.7
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return data["choices"][0]["message"]["content"].strip()
                else:
                    logger.error(f"OpenAI API error: {response.status_code}")
                    return self._generate_mock_response(prompt)
                    
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return self._generate_mock_response(prompt)
    
    def _generate_mock_response(self, query: str) -> str:
        """Generate mock response"""
        return f"""Based on your question about {query}, here's some educational guidance:

The Straight Arrow strategy is a simple, diversified approach:
• 60% VTI (Total Stock Market) - Broad U.S. equity exposure
• 30% BNDX (International Bonds) - International fixed income
• 10% GSG (Commodities) - Inflation protection

Key principles:
1. Diversification across asset classes
2. Low-cost index fund approach
3. Regular rebalancing (quarterly)
4. Long-term focus

DISCLAIMER: This is educational information only, not personalized investment advice. 
Consult with a registered investment advisor before making investment decisions."""

# ============================================================================
# INITIALIZE SERVICES
# ============================================================================

strategy_service = StraightArrowStrategy()
market_service = MarketDataService()
ai_service = AIService()

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "FinAgent Simplified API v1.5", 
        "docs": "/docs", 
        "health": "/health",
        "strategy": "Straight Arrow"
    }

@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.5.0",
        "strategy": "Straight Arrow",
        "database": "supabase" if supabase else "none",
        "market_data": "alpha_vantage" if ALPHA_VANTAGE_API_KEY else "mock",
        "ai_service": "openai" if OPENAI_API_KEY else "mock"
    }

@app.post("/api/analyze-portfolio")
async def analyze_portfolio(request: PortfolioRequest):
    """Analyze portfolio using Straight Arrow strategy"""
    try:
        analysis = strategy_service.analyze_portfolio(
            request.portfolio, 
            request.total_value
        )
        
        # Save to database if available
        if supabase:
            try:
                supabase.table("portfolio_analytics").insert({
                    "analysis_date": datetime.now().isoformat(),
                    "total_value": request.total_value,
                    "allocation": analysis["current_weights"],
                    "drift_analysis": analysis["drift_analysis"],
                    "risk_assessment": analysis["risk_assessment"],
                    "rebalance_recommendation": analysis["recommendations"]
                }).execute()
            except Exception as e:
                logger.error(f"Failed to save analysis: {e}")
        
        return {"analysis": analysis}
        
    except Exception as e:
        logger.error(f"Portfolio analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/market-data")
async def get_market_data(request: MarketDataRequest):
    """Get market data"""
    try:
        quotes = await market_service.get_quotes(request.symbols)
        return {
            "quotes": quotes,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Market data error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/agents/data-analyst")
async def ai_data_analyst(request: AIAnalysisRequest):
    """AI data analyst"""
    try:
        analysis = await ai_service.analyze(request.query, request.symbols)
        return {"analysis": analysis}
    except Exception as e:
        logger.error(f"AI analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/portfolio-history")
async def get_portfolio_history():
    """Get portfolio history"""
    try:
        if supabase:
            result = supabase.table("portfolio_analytics").select("*").order("analysis_date", desc=True).limit(10).execute()
            return {"history": result.data}
        else:
            return {"history": [], "message": "Database not available"}
    except Exception as e:
        logger.error(f"Portfolio history error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# STARTUP
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("FinAgent Simplified API v1.5 starting up...")
    logger.info("Strategy: Straight Arrow (60% VTI, 30% BNDX, 10% GSG)")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main_simplified:app",
        host="0.0.0.0",
        port=port,
        reload=False
    ) 