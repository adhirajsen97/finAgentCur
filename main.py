"""
FinAgent Main Application with Supabase Integration
"""
import os
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import logging
import json

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from supabase import create_client, Client
import httpx

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

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")

# Initialize Supabase client with error handling
supabase: Client = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Supabase client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {e}")
        supabase = None

# Optional Redis for caching
redis_client = None
try:
    import redis
    REDIS_URL = os.getenv("REDIS_URL")
    if REDIS_URL:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        redis_client.ping()  # Test connection
        logger.info("Redis connected successfully")
except Exception as e:
    logger.info(f"Redis not available: {e}")

# Pydantic models
class PortfolioRequest(BaseModel):
    portfolio: Dict[str, float]
    total_value: float

class AgentRequest(BaseModel):
    query: str
    symbols: Optional[List[str]] = None
    portfolio: Optional[Dict[str, float]] = None
    portfolio_value: Optional[float] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    environment: str
    database: str
    cache: str

# Database service
class SupabaseService:
    """Supabase database service"""
    
    def __init__(self, client: Client):
        self.client = client
    
    async def save_portfolio_analysis(self, analysis: Dict[str, Any]) -> str:
        """Save portfolio analysis to Supabase"""
        try:
            result = self.client.table("portfolio_analytics").insert({
                "analysis_date": datetime.now().isoformat(),
                "total_value": analysis.get("total_value", 0),
                "allocation": analysis.get("current_weights", {}),
                "drift_analysis": analysis.get("drift_analysis", {}),
                "risk_assessment": analysis.get("risk_assessment", {}),
                "rebalance_recommendation": analysis.get("recommendations", [])
            }).execute()
            
            return result.data[0]["id"] if result.data else None
        except Exception as e:
            logger.error(f"Failed to save portfolio analysis: {e}")
            return None
    
    async def get_portfolio_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get portfolio analysis history"""
        try:
            result = self.client.table("portfolio_analytics")\
                .select("*")\
                .order("analysis_date", desc=True)\
                .limit(limit)\
                .execute()
            
            return result.data if result.data else []
        except Exception as e:
            logger.error(f"Failed to get portfolio history: {e}")
            return []
    
    async def save_agent_log(self, agent_type: str, query: str, response: Dict[str, Any]) -> None:
        """Log AI agent interactions"""
        try:
            self.client.table("agent_logs").insert({
                "agent_type": agent_type,
                "status": "COMPLETED",
                "input_data": {"query": query},
                "output_data": response,
                "created_at": datetime.now().isoformat()
            }).execute()
        except Exception as e:
            logger.error(f"Failed to save agent log: {e}")

# Cache service
class CacheService:
    """Redis caching service with fallback to memory"""
    
    def __init__(self, redis_client=None):
        self.redis = redis_client
        self.memory_cache = {}  # Fallback to memory cache
    
    async def get(self, key: str) -> Optional[str]:
        """Get cached value"""
        try:
            if self.redis:
                return self.redis.get(key)
            else:
                # Memory cache with expiration check
                if key in self.memory_cache:
                    value, expires_at = self.memory_cache[key]
                    if datetime.now() < expires_at:
                        return value
                    else:
                        del self.memory_cache[key]
                return None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    async def set(self, key: str, value: str, ttl: int = 300) -> None:
        """Set cached value with TTL"""
        try:
            if self.redis:
                self.redis.setex(key, ttl, value)
            else:
                # Memory cache with expiration
                expires_at = datetime.now() + timedelta(seconds=ttl)
                self.memory_cache[key] = (value, expires_at)
        except Exception as e:
            logger.error(f"Cache set error: {e}")

# Market data service
class MarketDataService:
    """Market data service with caching"""
    
    def __init__(self, cache_service: CacheService):
        self.cache = cache_service
        self.alpha_vantage_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    
    async def get_current_price(self, symbol: str) -> Dict[str, Any]:
        """Get current stock price with caching"""
        cache_key = f"price:{symbol}"
        
        # Try cache first
        cached = await self.cache.get(cache_key)
        if cached:
            return json.loads(cached)
        
        # Fetch from API
        try:
            if self.alpha_vantage_key:
                url = f"https://www.alphavantage.co/query"
                params = {
                    "function": "GLOBAL_QUOTE",
                    "symbol": symbol,
                    "apikey": self.alpha_vantage_key
                }
                
                async with httpx.AsyncClient() as client:
                    response = await client.get(url, params=params)
                    data = response.json()
                
                if "Global Quote" in data:
                    quote = data["Global Quote"]
                    result = {
                        "symbol": symbol,
                        "price": float(quote.get("05. price", 0)),
                        "change": float(quote.get("09. change", 0)),
                        "change_percent": float(quote.get("10. change percent", "0%").replace("%", "")),
                        "volume": int(quote.get("06. volume", 0)),
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    # Cache for 5 minutes
                    await self.cache.set(cache_key, json.dumps(result), 300)
                    return result
            
            # Fallback to mock data
            return await self._get_mock_price(symbol)
            
        except Exception as e:
            logger.error(f"Market data error for {symbol}: {e}")
            return await self._get_mock_price(symbol)
    
    async def _get_mock_price(self, symbol: str) -> Dict[str, Any]:
        """Mock price data for demo"""
        mock_prices = {
            "VTI": {"price": 225.50, "change": 0.025, "volume": 1000000},
            "BNDX": {"price": 54.20, "change": -0.01, "volume": 500000},
            "GSG": {"price": 19.10, "change": 0.03, "volume": 200000}
        }
        base_data = mock_prices.get(symbol, {"price": 100.0, "change": 0.0, "volume": 100000})
        return {
            "symbol": symbol,
            "price": base_data["price"],
            "change": base_data["change"],
            "change_percent": base_data["change"] * 100,
            "volume": base_data["volume"],
            "timestamp": datetime.now().isoformat()
        }

# AI Agent service
class AIAgentService:
    """AI agent service with OpenAI integration"""
    
    def __init__(self, db_service: SupabaseService):
        self.db = db_service
        self.openai_key = os.getenv("OPENAI_API_KEY")
    
    async def analyze_data(self, query: str, symbols: List[str] = None) -> Dict[str, Any]:
        """Data analyst agent"""
        try:
            if self.openai_key and symbols:
                # Real AI analysis would go here
                # For now, return structured mock response
                analysis = {
                    "analysis": f"Market analysis for {', '.join(symbols)}: {query}",
                    "recommendations": [
                        "Current market conditions show moderate volatility",
                        "Three-fund portfolio allocation remains appropriate",
                        "Monitor bond yields for potential rebalancing opportunities"
                    ],
                    "confidence": 0.75,
                    "symbols_analyzed": symbols,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                analysis = {
                    "analysis": f"Analysis for query: {query}",
                    "recommendations": ["Enable OpenAI API key for detailed analysis"],
                    "confidence": 0.5,
                    "timestamp": datetime.now().isoformat()
                }
            
            # Log to database
            await self.db.save_agent_log("data_analyst", query, analysis)
            return analysis
            
        except Exception as e:
            logger.error(f"Data analyst error: {e}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}

# Initialize services
cache_service = CacheService(redis_client)
market_service = MarketDataService(cache_service)
db_service = SupabaseService(supabase) if supabase else None
ai_service = AIAgentService(db_service) if db_service else None

# API Routes
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "FinAgent API with Supabase", 
        "docs": "/docs", 
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        environment=os.getenv("ENVIRONMENT", "development"),
        database="supabase" if supabase else "none",
        cache="redis" if redis_client else "memory"
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
        
        # Rebalancing recommendations
        recommendations = []
        if max_drift > 0.05:
            for symbol, analysis in drift_analysis.items():
                if abs(analysis["drift"]) > 0.02:
                    action = "REDUCE" if analysis["drift"] > 0 else "INCREASE"
                    recommendations.append({
                        "symbol": symbol,
                        "action": action,
                        "current_percent": analysis["current_weight"] * 100,
                        "target_percent": analysis["target_weight"] * 100
                    })
        
        analysis_result = {
            "strategy": "Straight Arrow",
            "total_value": total_value,
            "current_weights": current_weights,
            "target_allocation": target_allocation,
            "drift_analysis": drift_analysis,
            "risk_assessment": risk_assessment,
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save to Supabase
        if db_service:
            analysis_id = await db_service.save_portfolio_analysis(analysis_result)
            analysis_result["analysis_id"] = analysis_id
        
        return {"analysis": analysis_result}
        
    except Exception as e:
        logger.error(f"Portfolio analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/market-sentiment")
async def get_market_sentiment():
    """Get market sentiment analysis"""
    try:
        # Check cache first
        cache_key = "market_sentiment"
        cached = await cache_service.get(cache_key)
        if cached:
            return {"sentiment": json.loads(cached)}
        
        # Generate sentiment analysis
        sentiment = {
            "overall_sentiment": "NEUTRAL",
            "confidence": 0.65,
            "factors": ["Mixed economic indicators", "Moderate volatility"],
            "timestamp": datetime.now().isoformat()
        }
        
        # Cache for 15 minutes
        await cache_service.set(cache_key, json.dumps(sentiment), 900)
        
        return {"sentiment": sentiment}
    except Exception as e:
        logger.error(f"Market sentiment error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/agents/data-analyst")
async def data_analyst_endpoint(request: AgentRequest):
    """Data analyst AI agent"""
    try:
        if not ai_service:
            raise HTTPException(status_code=503, detail="AI service not available")
        
        analysis = await ai_service.analyze_data(request.query, request.symbols)
        return {"analysis": analysis}
    except Exception as e:
        logger.error(f"Data analyst error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/portfolio-history")
async def get_portfolio_history(limit: int = 10):
    """Get portfolio analysis history from Supabase"""
    try:
        if not db_service:
            raise HTTPException(status_code=503, detail="Database service not available")
        
        history = await db_service.get_portfolio_history(limit)
        return {"history": history}
    except Exception as e:
        logger.error(f"Portfolio history error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main_supabase:app",
        host="0.0.0.0",
        port=port,
        reload=False
    ) 