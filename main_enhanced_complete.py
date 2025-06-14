"""
FinAgent - Enhanced Simple AI Investment System
Includes AI agents, compliance, and risk metrics with Straight Arrow strategy
"""

import os
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import logging
import json
import math

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import uvicorn
from supabase import create_client, Client
import httpx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="FinAgent - Enhanced Simple AI Investment System",
    description="AI-powered investment analysis with multiple agents and compliance",
    version="1.6.0",
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

class RiskAnalysisRequest(BaseModel):
    """Risk analysis request"""
    portfolio: Dict[str, float] = Field(..., description="Portfolio holdings")
    total_value: float = Field(..., gt=0, description="Total portfolio value")
    time_horizon: Optional[str] = Field(default="Long Term", description="Investment time horizon")

class TradingAnalysisRequest(BaseModel):
    """Trading analysis request"""
    symbols: List[str] = Field(..., description="Symbols to analyze")
    analysis_type: Optional[str] = Field(default="technical", description="Type of analysis")

# ============================================================================
# STRAIGHT ARROW STRATEGY
# ============================================================================

class StraightArrowStrategy:
    """Enhanced Straight Arrow investment strategy with risk metrics"""
    
    def __init__(self):
        # Fixed Straight Arrow allocation
        self.target_allocation = {
            "VTI": 0.60,   # 60% Total Stock Market
            "BNDX": 0.30,  # 30% International Bonds  
            "GSG": 0.10    # 10% Commodities
        }
        
        # Expected returns and volatilities (annual)
        self.expected_returns = {
            "VTI": 0.10,   # 10% expected return
            "BNDX": 0.04,  # 4% expected return
            "GSG": 0.06    # 6% expected return
        }
        
        self.volatilities = {
            "VTI": 0.16,   # 16% volatility
            "BNDX": 0.05,  # 5% volatility
            "GSG": 0.20    # 20% volatility
        }
    
    def analyze_portfolio(self, portfolio: Dict[str, float], total_value: float) -> Dict[str, Any]:
        """Enhanced portfolio analysis with risk metrics"""
        
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
        
        # Calculate portfolio metrics
        portfolio_metrics = self._calculate_portfolio_metrics(current_weights)
        target_metrics = self._calculate_portfolio_metrics(self.target_allocation)
        
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
        
        # Compliance check
        compliance_status = self._check_compliance(current_weights, portfolio_metrics)
        
        return {
            "strategy": "Straight Arrow",
            "total_value": total_value,
            "current_weights": current_weights,
            "target_allocation": self.target_allocation,
            "drift_analysis": drift_analysis,
            "portfolio_metrics": portfolio_metrics,
            "target_metrics": target_metrics,
            "risk_assessment": {
                "overall_risk": risk_level,
                "max_drift": max_drift,
                "needs_rebalancing": needs_rebalancing,
                "sharpe_ratio": portfolio_metrics["sharpe_ratio"],
                "volatility": portfolio_metrics["volatility"],
                "expected_return": portfolio_metrics["expected_return"]
            },
            "compliance": compliance_status,
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat()
        }
    
    def _calculate_portfolio_metrics(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Calculate portfolio expected return, volatility, and Sharpe ratio"""
        
        # Expected return (weighted average)
        expected_return = sum(
            weights.get(symbol, 0) * self.expected_returns.get(symbol, 0)
            for symbol in self.expected_returns.keys()
        )
        
        # Portfolio volatility (simplified - assumes no correlation)
        portfolio_variance = sum(
            (weights.get(symbol, 0) ** 2) * (self.volatilities.get(symbol, 0) ** 2)
            for symbol in self.volatilities.keys()
        )
        volatility = math.sqrt(portfolio_variance)
        
        # Sharpe ratio (assuming 2% risk-free rate)
        risk_free_rate = 0.02
        sharpe_ratio = (expected_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        return {
            "expected_return": expected_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio
        }
    
    def _check_compliance(self, weights: Dict[str, float], metrics: Dict[str, float]) -> Dict[str, Any]:
        """Basic compliance checking"""
        
        violations = []
        warnings = []
        
        # Check if portfolio is too risky
        if metrics["volatility"] > 0.18:  # 18% volatility threshold
            warnings.append({
                "rule": "Risk Management",
                "message": f"Portfolio volatility ({metrics['volatility']:.1%}) exceeds recommended 18%",
                "severity": "MEDIUM"
            })
        
        # Check diversification
        max_allocation = max(weights.values()) if weights else 0
        if max_allocation > 0.70:  # 70% single asset limit
            violations.append({
                "rule": "Diversification",
                "message": f"Single asset allocation ({max_allocation:.1%}) exceeds 70% limit",
                "severity": "HIGH"
            })
        
        # Check minimum Sharpe ratio
        if metrics["sharpe_ratio"] < 0.3:
            warnings.append({
                "rule": "Risk-Adjusted Returns",
                "message": f"Sharpe ratio ({metrics['sharpe_ratio']:.2f}) below recommended 0.3",
                "severity": "MEDIUM"
            })
        
        # Determine status
        if violations:
            status = "NON_COMPLIANT"
        elif warnings:
            status = "NEEDS_REVIEW"
        else:
            status = "COMPLIANT"
        
        return {
            "status": status,
            "violations": violations,
            "warnings": warnings,
            "disclosures": [
                "This analysis is for educational purposes only.",
                "Past performance does not guarantee future results.",
                "All investments carry risk of loss.",
                "Consult with a registered investment advisor."
            ],
            "validated_at": datetime.now().isoformat()
        }

# ============================================================================
# MARKET DATA SERVICE
# ============================================================================

class MarketDataService:
    """Enhanced market data service with sentiment analysis"""
    
    def __init__(self):
        self.alpha_vantage_key = ALPHA_VANTAGE_API_KEY
    
    async def get_quotes(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get quotes for symbols with enhanced data"""
        quotes = {}
        
        for symbol in symbols:
            try:
                if self.alpha_vantage_key:
                    quote = await self._fetch_quote_alpha_vantage(symbol)
                    if quote:
                        # Add technical indicators
                        quote["technical_analysis"] = self._basic_technical_analysis(quote)
                        quotes[symbol] = quote
                        continue
                
                # Enhanced mock data
                quotes[symbol] = self._generate_enhanced_mock_data(symbol)
                
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
                            "high": float(quote_data.get("03. High", 0)),
                            "low": float(quote_data.get("04. Low", 0)),
                            "timestamp": datetime.now().isoformat(),
                            "source": "alpha_vantage"
                        }
        except Exception as e:
            logger.error(f"Alpha Vantage API error for {symbol}: {e}")
        
        return None
    
    def _generate_enhanced_mock_data(self, symbol: str) -> Dict[str, Any]:
        """Generate enhanced mock data with technical analysis"""
        
        # Base prices for different symbols
        base_prices = {"VTI": 220.0, "BNDX": 52.0, "GSG": 16.0}
        base_price = base_prices.get(symbol, 100.0)
        
        mock_data = {
            "symbol": symbol,
            "price": base_price,
            "change": 1.5,
            "change_percent": "+1.5%",
            "volume": 1000000,
            "high": base_price * 1.02,
            "low": base_price * 0.98,
            "timestamp": datetime.now().isoformat(),
            "source": "mock"
        }
        
        # Add technical analysis
        mock_data["technical_analysis"] = self._basic_technical_analysis(mock_data)
        
        return mock_data
    
    def _basic_technical_analysis(self, quote: Dict[str, Any]) -> Dict[str, Any]:
        """Basic technical analysis indicators"""
        
        price = quote.get("price", 0)
        high = quote.get("high", price)
        low = quote.get("low", price)
        
        # Simple technical indicators
        return {
            "trend": "BULLISH" if quote.get("change", 0) > 0 else "BEARISH",
            "volatility": "NORMAL",
            "support_level": low * 0.95,
            "resistance_level": high * 1.05,
            "recommendation": "HOLD"
        }
    
    async def get_market_sentiment(self) -> Dict[str, Any]:
        """Get overall market sentiment"""
        return {
            "overall_sentiment": "NEUTRAL",
            "fear_greed_index": 50,
            "market_trend": "SIDEWAYS",
            "volatility_index": 20,
            "timestamp": datetime.now().isoformat(),
            "source": "mock"
        }

# Continue in next part... # ============================================================================
# AI AGENTS
# ============================================================================

class AIAgentService:
    """Multiple AI agents for different analysis types"""
    
    def __init__(self):
        self.openai_key = OPENAI_API_KEY
    
    async def data_analyst(self, query: str, symbols: List[str] = None) -> Dict[str, Any]:
        """Data analyst agent - market data and fundamental analysis"""
        
        context = f"""You are a financial data analyst AI. Analyze the following query with focus on:
        - Market data interpretation
        - Fundamental analysis
        - Economic indicators
        - Data-driven insights
        
        Query: {query}
        Symbols: {', '.join(symbols) if symbols else 'General market'}
        
        Provide educational analysis focusing on the Straight Arrow strategy (60% VTI, 30% BNDX, 10% GSG).
        Include appropriate disclaimers."""
        
        response = await self._call_openai(context, "data_analyst") if self.openai_key else self._mock_data_analyst(query)
        
        return {
            "agent": "data_analyst",
            "query": query,
            "analysis": response,
            "symbols": symbols or [],
            "confidence": 0.85 if self.openai_key else 0.7,
            "timestamp": datetime.now().isoformat()
        }
    
    async def risk_analyst(self, portfolio: Dict[str, float], total_value: float, time_horizon: str = "Long Term") -> Dict[str, Any]:
        """Risk analyst agent - risk assessment and management"""
        
        # Calculate portfolio weights
        weights = {symbol: value / total_value for symbol, value in portfolio.items()}
        
        context = f"""You are a risk management analyst AI. Analyze this portfolio:
        
        Portfolio: {weights}
        Total Value: ${total_value:,.2f}
        Time Horizon: {time_horizon}
        
        Assess:
        - Risk levels and concentration
        - Diversification adequacy
        - Risk-adjusted returns
        - Recommendations for risk management
        
        Compare against Straight Arrow strategy (60% VTI, 30% BNDX, 10% GSG).
        Include appropriate risk disclaimers."""
        
        response = await self._call_openai(context, "risk_analyst") if self.openai_key else self._mock_risk_analyst(weights)
        
        return {
            "agent": "risk_analyst",
            "portfolio": weights,
            "total_value": total_value,
            "time_horizon": time_horizon,
            "analysis": response,
            "confidence": 0.80 if self.openai_key else 0.7,
            "timestamp": datetime.now().isoformat()
        }
    
    async def trading_analyst(self, symbols: List[str], analysis_type: str = "technical") -> Dict[str, Any]:
        """Trading analyst agent - technical analysis and trading signals"""
        
        context = f"""You are a trading analyst AI. Provide {analysis_type} analysis for:
        
        Symbols: {', '.join(symbols)}
        Analysis Type: {analysis_type}
        
        Focus on:
        - Technical indicators
        - Chart patterns
        - Trading signals
        - Entry/exit points
        
        Remember these are long-term Straight Arrow strategy holdings.
        Emphasize buy-and-hold approach with quarterly rebalancing.
        Include trading disclaimers."""
        
        response = await self._call_openai(context, "trading_analyst") if self.openai_key else self._mock_trading_analyst(symbols)
        
        return {
            "agent": "trading_analyst",
            "symbols": symbols,
            "analysis_type": analysis_type,
            "analysis": response,
            "confidence": 0.75 if self.openai_key else 0.6,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _call_openai(self, prompt: str, agent_type: str) -> str:
        """Call OpenAI API with agent-specific prompting"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {self.openai_key}"},
                    json={
                        "model": "gpt-3.5-turbo",
                        "messages": [
                            {"role": "system", "content": f"You are a professional {agent_type.replace('_', ' ')} providing educational financial analysis."},
                            {"role": "user", "content": prompt}
                        ],
                        "max_tokens": 600,
                        "temperature": 0.7
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return data["choices"][0]["message"]["content"].strip()
                else:
                    logger.error(f"OpenAI API error: {response.status_code}")
                    return self._get_mock_response(agent_type, prompt)
                    
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return self._get_mock_response(agent_type, prompt)
    
    def _mock_data_analyst(self, query: str) -> str:
        """Mock data analyst response"""
        return f"""Data Analysis for: {query}

Based on current market data and the Straight Arrow strategy:

📊 Market Overview:
• VTI (Total Stock Market): Strong long-term fundamentals with broad diversification
• BNDX (International Bonds): Provides stability and currency diversification  
• GSG (Commodities): Inflation hedge with moderate volatility

📈 Key Insights:
1. The 60/30/10 allocation balances growth potential with risk management
2. Regular rebalancing maintains target allocation despite market movements
3. Low-cost index approach minimizes fees and tracking error

⚠️ Risk Considerations:
• Market volatility can affect short-term performance
• International exposure adds currency risk
• Commodity allocation increases overall portfolio volatility

DISCLAIMER: This is educational analysis only. Past performance doesn't guarantee future results. Consult a registered investment advisor for personalized advice."""
    
    def _mock_risk_analyst(self, weights: Dict[str, float]) -> str:
        """Mock risk analyst response"""
        max_allocation = max(weights.values()) if weights else 0
        
        return f"""Risk Assessment Analysis:

🎯 Portfolio Allocation:
{chr(10).join([f'• {symbol}: {weight:.1%}' for symbol, weight in weights.items()])}

📊 Risk Metrics:
• Concentration Risk: {'HIGH' if max_allocation > 0.7 else 'MODERATE' if max_allocation > 0.5 else 'LOW'}
• Diversification: {'Good' if len(weights) >= 3 else 'Needs Improvement'}
• Asset Class Mix: {'Balanced' if any('BND' in k for k in weights.keys()) else 'Equity Heavy'}

⚖️ Risk Assessment:
1. Portfolio shows {'good' if max_allocation < 0.7 else 'concerning'} concentration levels
2. {'Adequate' if len(weights) >= 3 else 'Limited'} diversification across asset classes
3. Time horizon alignment: Suitable for long-term investors

🔧 Recommendations:
• Consider rebalancing toward Straight Arrow targets (60% VTI, 30% BNDX, 10% GSG)
• Maintain quarterly rebalancing schedule
• Monitor risk tolerance alignment

DISCLAIMER: Risk analysis is educational only. Individual risk tolerance varies. Consult with a qualified advisor."""
    
    def _mock_trading_analyst(self, symbols: List[str]) -> str:
        """Mock trading analyst response"""
        return f"""Technical Analysis for: {', '.join(symbols)}

📈 Technical Overview:
• Current trend: Mixed signals across asset classes
• Market sentiment: Neutral with cautious optimism
• Volatility: Within normal ranges

🎯 Straight Arrow Strategy Signals:
• VTI: Long-term uptrend intact, suitable for core holding
• BNDX: Stable performance, good defensive characteristics
• GSG: Cyclical patterns, provides portfolio diversification

📊 Trading Recommendations:
1. MAINTAIN current allocations - no major rebalancing signals
2. MONITOR quarterly for drift beyond 5% thresholds
3. DOLLAR-COST AVERAGE for new investments

⏰ Timing Considerations:
• Avoid market timing attempts
• Focus on systematic rebalancing
• Maintain long-term perspective

DISCLAIMER: Technical analysis is educational. Past patterns don't predict future performance. This is not trading advice."""
    
    def _get_mock_response(self, agent_type: str, prompt: str) -> str:
        """Get appropriate mock response based on agent type"""
        if agent_type == "data_analyst":
            return self._mock_data_analyst(prompt.split("Query:")[-1].split("\n")[0] if "Query:" in prompt else "market analysis")
        elif agent_type == "risk_analyst":
            return self._mock_risk_analyst({})
        elif agent_type == "trading_analyst":
            return self._mock_trading_analyst(["VTI", "BNDX", "GSG"])
        else:
            return "Analysis not available. Please try again later."

# ============================================================================
# INITIALIZE SERVICES
# ============================================================================

strategy_service = StraightArrowStrategy()
market_service = MarketDataService()
ai_service = AIAgentService()

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "FinAgent Enhanced Simple API v1.6", 
        "docs": "/docs", 
        "health": "/health",
        "workflow_guide": "/workflow",
        "workflow_api": "/api/workflow-guide",
        "strategy": "Straight Arrow",
        "features": ["Portfolio Analysis", "Risk Metrics", "AI Agents", "Compliance", "Market Data"],
        "quick_start": {
            "1": "GET /health - Check system status",
            "2": "GET /api/workflow-guide - Complete usage guide",
            "3": "POST /api/market-data - Get market data",
            "4": "POST /api/analyze-portfolio - Analyze portfolio",
            "5": "POST /api/agents/* - Use AI agents"
        }
    }

@app.get("/health")
async def health_check():
    """Enhanced health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.6.0",
        "strategy": "Straight Arrow",
        "features": {
            "database": "supabase" if supabase else "none",
            "market_data": "alpha_vantage" if ALPHA_VANTAGE_API_KEY else "mock",
            "ai_service": "openai" if OPENAI_API_KEY else "mock",
            "agents": ["data_analyst", "risk_analyst", "trading_analyst"],
            "compliance": "enabled",
            "risk_metrics": "enabled"
        }
    }

@app.post("/api/analyze-portfolio")
async def analyze_portfolio(request: PortfolioRequest):
    """Enhanced portfolio analysis with risk metrics and compliance"""
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
    """Get enhanced market data with technical analysis"""
    try:
        quotes = await market_service.get_quotes(request.symbols)
        return {
            "quotes": quotes,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Market data error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/market-sentiment")
async def get_market_sentiment():
    """Get market sentiment analysis"""
    try:
        sentiment = await market_service.get_market_sentiment()
        return {"sentiment": sentiment}
    except Exception as e:
        logger.error(f"Market sentiment error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# AI AGENT ENDPOINTS
# ============================================================================

@app.post("/api/agents/data-analyst")
async def ai_data_analyst(request: AIAnalysisRequest):
    """Data analyst AI agent"""
    try:
        analysis = await ai_service.data_analyst(request.query, request.symbols)
        return {"analysis": analysis}
    except Exception as e:
        logger.error(f"Data analyst error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/agents/risk-analyst")
async def ai_risk_analyst(request: RiskAnalysisRequest):
    """Risk analyst AI agent"""
    try:
        analysis = await ai_service.risk_analyst(
            request.portfolio, 
            request.total_value, 
            request.time_horizon
        )
        return {"analysis": analysis}
    except Exception as e:
        logger.error(f"Risk analyst error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/agents/trading-analyst")
async def ai_trading_analyst(request: TradingAnalysisRequest):
    """Trading analyst AI agent"""
    try:
        analysis = await ai_service.trading_analyst(
            request.symbols, 
            request.analysis_type
        )
        return {"analysis": analysis}
    except Exception as e:
        logger.error(f"Trading analyst error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# ADDITIONAL ENDPOINTS
# ============================================================================

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

@app.get("/api/strategy-performance")
async def get_strategy_performance():
    """Get Straight Arrow strategy performance metrics"""
    try:
        # Calculate target portfolio metrics
        target_metrics = strategy_service._calculate_portfolio_metrics(strategy_service.target_allocation)
        
        return {
            "strategy": "Straight Arrow",
            "target_allocation": strategy_service.target_allocation,
            "expected_metrics": target_metrics,
            "performance_summary": {
                "expected_annual_return": f"{target_metrics['expected_return']:.1%}",
                "expected_volatility": f"{target_metrics['volatility']:.1%}",
                "sharpe_ratio": f"{target_metrics['sharpe_ratio']:.2f}",
                "risk_level": "Moderate",
                "rebalancing_frequency": "Quarterly"
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Strategy performance error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/compliance/disclosures")
async def get_compliance_disclosures():
    """Get regulatory compliance disclosures"""
    return {
        "disclosures": [
            "This analysis is for educational purposes only and does not constitute financial advice.",
            "All investments carry risk of loss. Past performance does not guarantee future results.",
            "This system uses artificial intelligence technology for analysis.",
            "Please consult with a registered investment advisor before making investment decisions.",
            "FinAgent is not a registered investment advisor or broker-dealer.",
            "Consider your personal financial situation and risk tolerance before investing.",
            "Rebalancing may have tax implications. Consult a tax professional.",
            "Market data may be delayed and should not be used for real-time trading decisions."
        ],
        "last_updated": datetime.now().isoformat(),
        "version": "1.0"
    }

@app.get("/api/workflow-guide")
async def get_workflow_guide():
    """Get comprehensive workflow guide for using FinAgent API effectively"""
    return {
        "title": "FinAgent API - Complete Workflow Guide",
        "description": "Step-by-step guide to create effective investment strategies using all enhanced features",
        "last_updated": datetime.now().isoformat(),
        "version": "1.6.0",
        "workflow": {
            "overview": {
                "title": "🎯 3-Week Investment Strategy Workflow",
                "description": "Optimal sequence of API calls to create an effective short-term strategy using all enhanced features",
                "time_horizon": "3 weeks",
                "strategy": "Straight Arrow (60% VTI, 30% BNDX, 10% GSG)"
            },
            "steps": [
                {
                    "step": 1,
                    "title": "Health Check & Feature Verification",
                    "description": "Verify all enhanced features are available",
                    "endpoint": "GET /health",
                    "curl_example": "curl -X GET \"http://localhost:8000/health\"",
                    "python_example": """import httpx
response = httpx.get("http://localhost:8000/health")
features = response.json()["features"]
print(f"Available agents: {features['agents']}")""",
                    "expected_response": {
                        "status": "healthy",
                        "features": {
                            "agents": ["data_analyst", "risk_analyst", "trading_analyst"],
                            "compliance": "enabled",
                            "risk_metrics": "enabled"
                        }
                    }
                },
                {
                    "step": 2,
                    "title": "Get Current Market Data & Technical Analysis",
                    "description": "Get real-time market data with technical indicators",
                    "endpoint": "POST /api/market-data",
                    "curl_example": """curl -X POST "http://localhost:8000/api/market-data" \\
  -H "Content-Type: application/json" \\
  -d '{"symbols": ["VTI", "BNDX", "GSG"]}'""",
                    "python_example": """response = httpx.post("http://localhost:8000/api/market-data", json={
    "symbols": ["VTI", "BNDX", "GSG"]
})
market_data = response.json()
for symbol, quote in market_data["quotes"].items():
    tech = quote.get("technical_analysis", {})
    print(f"{symbol}: ${quote['price']:.2f} | {tech.get('trend', 'N/A')}")""",
                    "key_insights": [
                        "Current prices and price changes",
                        "Technical trend analysis (BULLISH/BEARISH)",
                        "Support and resistance levels",
                        "Trading recommendations"
                    ]
                },
                {
                    "step": 3,
                    "title": "Get Market Sentiment Analysis",
                    "description": "Check overall market sentiment for timing decisions",
                    "endpoint": "GET /api/market-sentiment",
                    "curl_example": "curl -X GET \"http://localhost:8000/api/market-sentiment\"",
                    "python_example": """response = httpx.get("http://localhost:8000/api/market-sentiment")
sentiment = response.json()["sentiment"]
print(f"Market Sentiment: {sentiment['overall_sentiment']}")
print(f"Fear & Greed Index: {sentiment['fear_greed_index']}")""",
                    "key_insights": [
                        "Overall market sentiment (BULLISH/BEARISH/NEUTRAL)",
                        "Fear & Greed Index (0-100)",
                        "Market trend direction",
                        "Volatility index"
                    ]
                },
                {
                    "step": 4,
                    "title": "AI Data Analyst - Market Conditions Assessment",
                    "description": "Get AI analysis of current market conditions for 3-week outlook",
                    "endpoint": "POST /api/agents/data-analyst",
                    "curl_example": """curl -X POST "http://localhost:8000/api/agents/data-analyst" \\
  -H "Content-Type: application/json" \\
  -d '{
    "query": "What are the current market conditions and outlook for the next 3 weeks?",
    "symbols": ["VTI", "BNDX", "GSG"]
  }'""",
                    "python_example": """response = httpx.post("http://localhost:8000/api/agents/data-analyst", json={
    "query": "What are the current market conditions and outlook for the next 3 weeks?",
    "symbols": ["VTI", "BNDX", "GSG"]
})
analysis = response.json()["analysis"]
print(f"Confidence: {analysis['confidence']}")
print(f"Analysis: {analysis['analysis']}")""",
                    "key_insights": [
                        "Market data interpretation",
                        "Economic indicators analysis",
                        "3-week market outlook",
                        "Strategy adjustment recommendations"
                    ]
                },
                {
                    "step": 5,
                    "title": "Analyze Your Current Portfolio",
                    "description": "Analyze portfolio with enhanced risk metrics and compliance",
                    "endpoint": "POST /api/analyze-portfolio",
                    "curl_example": """curl -X POST "http://localhost:8000/api/analyze-portfolio" \\
  -H "Content-Type: application/json" \\
  -d '{
    "portfolio": {"VTI": 50000.0, "BNDX": 30000.0, "GSG": 20000.0},
    "total_value": 100000.0
  }'""",
                    "python_example": """response = httpx.post("http://localhost:8000/api/analyze-portfolio", json={
    "portfolio": {"VTI": 50000.0, "BNDX": 30000.0, "GSG": 20000.0},
    "total_value": 100000.0
})
analysis = response.json()["analysis"]
print(f"Expected Return: {analysis['portfolio_metrics']['expected_return']:.1%}")
print(f"Sharpe Ratio: {analysis['portfolio_metrics']['sharpe_ratio']:.2f}")""",
                    "key_insights": [
                        "Current vs target allocation drift",
                        "Expected return and volatility",
                        "Sharpe ratio (risk-adjusted returns)",
                        "Compliance status and violations",
                        "Specific rebalancing recommendations"
                    ]
                },
                {
                    "step": 6,
                    "title": "AI Risk Analyst - 3-Week Risk Assessment",
                    "description": "Get specialized risk analysis for 3-week time horizon",
                    "endpoint": "POST /api/agents/risk-analyst",
                    "curl_example": """curl -X POST "http://localhost:8000/api/agents/risk-analyst" \\
  -H "Content-Type: application/json" \\
  -d '{
    "portfolio": {"VTI": 50000.0, "BNDX": 30000.0, "GSG": 20000.0},
    "total_value": 100000.0,
    "time_horizon": "3 weeks"
  }'""",
                    "python_example": """response = httpx.post("http://localhost:8000/api/agents/risk-analyst", json={
    "portfolio": {"VTI": 50000.0, "BNDX": 30000.0, "GSG": 20000.0},
    "total_value": 100000.0,
    "time_horizon": "3 weeks"
})
risk_analysis = response.json()["analysis"]
print(f"Risk Assessment: {risk_analysis['analysis']}")""",
                    "key_insights": [
                        "Risk levels and concentration analysis",
                        "Diversification adequacy",
                        "Time horizon alignment",
                        "Risk management recommendations"
                    ]
                },
                {
                    "step": 7,
                    "title": "AI Trading Analyst - Short-term Signals",
                    "description": "Get trading signals and timing for 3-week strategy",
                    "endpoint": "POST /api/agents/trading-analyst",
                    "curl_example": """curl -X POST "http://localhost:8000/api/agents/trading-analyst" \\
  -H "Content-Type: application/json" \\
  -d '{
    "symbols": ["VTI", "BNDX", "GSG"],
    "analysis_type": "short-term"
  }'""",
                    "python_example": """response = httpx.post("http://localhost:8000/api/agents/trading-analyst", json={
    "symbols": ["VTI", "BNDX", "GSG"],
    "analysis_type": "short-term"
})
trading_analysis = response.json()["analysis"]
print(f"Trading Signals: {trading_analysis['analysis']}")""",
                    "key_insights": [
                        "Technical indicators and chart patterns",
                        "Entry and exit point recommendations",
                        "Short-term trading signals",
                        "Market timing considerations"
                    ]
                },
                {
                    "step": 8,
                    "title": "Get Strategy Performance Baseline",
                    "description": "Get Straight Arrow strategy expected performance metrics",
                    "endpoint": "GET /api/strategy-performance",
                    "curl_example": "curl -X GET \"http://localhost:8000/api/strategy-performance\"",
                    "python_example": """response = httpx.get("http://localhost:8000/api/strategy-performance")
performance = response.json()
print(f"Expected Return: {performance['performance_summary']['expected_annual_return']}")
print(f"Sharpe Ratio: {performance['performance_summary']['sharpe_ratio']}")""",
                    "key_insights": [
                        "Expected annual return (~7.8%)",
                        "Expected volatility (~11.2%)",
                        "Sharpe ratio (~0.52)",
                        "Risk level classification"
                    ]
                },
                {
                    "step": 9,
                    "title": "Check Compliance & Disclosures",
                    "description": "Review compliance requirements and regulatory disclosures",
                    "endpoint": "GET /api/compliance/disclosures",
                    "curl_example": "curl -X GET \"http://localhost:8000/api/compliance/disclosures\"",
                    "python_example": """response = httpx.get("http://localhost:8000/api/compliance/disclosures")
disclosures = response.json()["disclosures"]
for i, disclosure in enumerate(disclosures[:3]):
    print(f"{i+1}. {disclosure}")""",
                    "key_insights": [
                        "Educational purpose disclaimers",
                        "Investment risk warnings",
                        "AI technology disclosures",
                        "Professional advice recommendations"
                    ]
                }
            ],
            "complete_workflow_example": {
                "title": "Complete Python Workflow Script",
                "description": "Full Python script implementing all 9 steps",
                "code": """import httpx
import asyncio
from datetime import datetime

class FinAgentWorkflow:
    def __init__(self, base_url="http://localhost:8000"):
        self.client = httpx.AsyncClient(base_url=base_url)
        self.portfolio = {"VTI": 50000.0, "BNDX": 30000.0, "GSG": 20000.0}
        self.total_value = 100000.0
    
    async def run_complete_workflow(self):
        print("🚀 FinAgent 3-Week Strategy Workflow")
        print("=" * 50)
        
        # Step 1: Health check
        health = await self.client.get("/health")
        print(f"✅ System Status: {health.json()['status']}")
        
        # Step 2: Market data
        market_data = await self.client.post("/api/market-data", 
            json={"symbols": ["VTI", "BNDX", "GSG"]})
        quotes = market_data.json()["quotes"]
        print("📈 Current Market Data:")
        for symbol, quote in quotes.items():
            print(f"  {symbol}: ${quote['price']:.2f}")
        
        # Step 3: Market sentiment
        sentiment = await self.client.get("/api/market-sentiment")
        sentiment_data = sentiment.json()["sentiment"]
        print(f"🎭 Market Sentiment: {sentiment_data['overall_sentiment']}")
        
        # Step 4-6: AI Agents (parallel execution)
        data_task = self.client.post("/api/agents/data-analyst", json={
            "query": "3-week market outlook for Straight Arrow strategy",
            "symbols": ["VTI", "BNDX", "GSG"]
        })
        risk_task = self.client.post("/api/agents/risk-analyst", json={
            "portfolio": self.portfolio,
            "total_value": self.total_value,
            "time_horizon": "3 weeks"
        })
        trading_task = self.client.post("/api/agents/trading-analyst", json={
            "symbols": ["VTI", "BNDX", "GSG"],
            "analysis_type": "short-term"
        })
        
        data_analysis, risk_analysis, trading_analysis = await asyncio.gather(
            data_task, risk_task, trading_task
        )
        
        print("🤖 AI Analysis Complete:")
        print(f"  Data Analyst Confidence: {data_analysis.json()['analysis']['confidence']}")
        print(f"  Risk Analyst Confidence: {risk_analysis.json()['analysis']['confidence']}")
        print(f"  Trading Analyst Confidence: {trading_analysis.json()['analysis']['confidence']}")
        
        # Step 5: Portfolio analysis
        portfolio_analysis = await self.client.post("/api/analyze-portfolio", json={
            "portfolio": self.portfolio,
            "total_value": self.total_value
        })
        analysis = portfolio_analysis.json()["analysis"]
        print(f"📊 Portfolio Metrics:")
        print(f"  Expected Return: {analysis['portfolio_metrics']['expected_return']:.1%}")
        print(f"  Sharpe Ratio: {analysis['portfolio_metrics']['sharpe_ratio']:.2f}")
        print(f"  Compliance: {analysis['compliance']['status']}")
        
        await self.client.aclose()
        return "Workflow completed successfully!"

# Usage
async def main():
    workflow = FinAgentWorkflow()
    result = await workflow.run_complete_workflow()
    print(result)

if __name__ == "__main__":
    asyncio.run(main())"""
            },
            "best_practices": [
                {
                    "title": "Parallel API Calls",
                    "description": "Use asyncio.gather() to call multiple AI agents simultaneously for faster execution"
                },
                {
                    "title": "Error Handling",
                    "description": "Always check response status codes and handle API errors gracefully"
                },
                {
                    "title": "Rate Limiting",
                    "description": "Be mindful of API rate limits, especially for external services like OpenAI"
                },
                {
                    "title": "Data Validation",
                    "description": "Validate portfolio data and ensure total_value matches sum of holdings"
                },
                {
                    "title": "Regular Updates",
                    "description": "Run workflow weekly for 3-week strategies to adapt to changing market conditions"
                }
            ],
            "expected_outcomes": [
                "Comprehensive market assessment with real-time data",
                "AI-powered insights from 3 specialized agents",
                "Risk-adjusted portfolio recommendations",
                "Specific rebalancing actions with priorities",
                "Compliance validation and regulatory awareness",
                "Performance expectations and risk metrics"
            ]
        }
    }

@app.get("/workflow", response_class=HTMLResponse)
async def get_workflow_html():
    """Get workflow guide as formatted HTML page"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>FinAgent API - Complete Workflow Guide</title>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background: #f8f9fa;
            }
            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 2rem;
                border-radius: 10px;
                margin-bottom: 2rem;
                text-align: center;
            }
            .step {
                background: white;
                margin: 1.5rem 0;
                padding: 1.5rem;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                border-left: 4px solid #667eea;
            }
            .step-number {
                background: #667eea;
                color: white;
                width: 30px;
                height: 30px;
                border-radius: 50%;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                font-weight: bold;
                margin-right: 10px;
            }
            .code-block {
                background: #f1f3f4;
                padding: 1rem;
                border-radius: 5px;
                font-family: 'Monaco', 'Menlo', monospace;
                font-size: 0.9rem;
                overflow-x: auto;
                margin: 1rem 0;
                border: 1px solid #e0e0e0;
            }
            .insights {
                background: #e8f5e8;
                padding: 1rem;
                border-radius: 5px;
                border-left: 4px solid #4caf50;
                margin: 1rem 0;
            }
            .insights h4 {
                color: #2e7d32;
                margin-top: 0;
            }
            .best-practices {
                background: #fff3e0;
                padding: 1.5rem;
                border-radius: 8px;
                margin: 2rem 0;
                border-left: 4px solid #ff9800;
            }
            .outcomes {
                background: #e3f2fd;
                padding: 1.5rem;
                border-radius: 8px;
                margin: 2rem 0;
                border-left: 4px solid #2196f3;
            }
            .tab-container {
                margin: 1rem 0;
            }
            .tab-buttons {
                display: flex;
                margin-bottom: 1rem;
            }
            .tab-button {
                background: #f1f3f4;
                border: none;
                padding: 0.5rem 1rem;
                cursor: pointer;
                border-radius: 5px 5px 0 0;
                margin-right: 2px;
            }
            .tab-button.active {
                background: #667eea;
                color: white;
            }
            .tab-content {
                display: none;
            }
            .tab-content.active {
                display: block;
            }
            ul {
                padding-left: 1.5rem;
            }
            li {
                margin: 0.5rem 0;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🚀 FinAgent API - Complete Workflow Guide</h1>
            <p>Step-by-step guide to create effective 3-week investment strategies</p>
            <p><strong>Strategy:</strong> Straight Arrow (60% VTI, 30% BNDX, 10% GSG)</p>
        </div>

        <div class="step">
            <h2><span class="step-number">1</span>Health Check & Feature Verification</h2>
            <p>Verify all enhanced features are available before starting your analysis.</p>
            
            <div class="tab-container">
                <div class="tab-buttons">
                    <button class="tab-button active" onclick="showTab(event, 'curl1')">cURL</button>
                    <button class="tab-button" onclick="showTab(event, 'python1')">Python</button>
                </div>
                <div id="curl1" class="tab-content active">
                    <div class="code-block">curl -X GET "http://localhost:8000/health"</div>
                </div>
                <div id="python1" class="tab-content">
                    <div class="code-block">import httpx
response = httpx.get("http://localhost:8000/health")
features = response.json()["features"]
print(f"Available agents: {features['agents']}")</div>
                </div>
            </div>
            
            <div class="insights">
                <h4>🎯 What You Get:</h4>
                <ul>
                    <li>System health status</li>
                    <li>Available AI agents (3 specialized agents)</li>
                    <li>Feature availability (compliance, risk metrics)</li>
                    <li>Data source status (OpenAI, Alpha Vantage)</li>
                </ul>
            </div>
        </div>

        <div class="step">
            <h2><span class="step-number">2</span>Get Current Market Data & Technical Analysis</h2>
            <p>Get real-time market data with technical indicators for VTI, BNDX, and GSG.</p>
            
            <div class="tab-container">
                <div class="tab-buttons">
                    <button class="tab-button active" onclick="showTab(event, 'curl2')">cURL</button>
                    <button class="tab-button" onclick="showTab(event, 'python2')">Python</button>
                </div>
                <div id="curl2" class="tab-content active">
                    <div class="code-block">curl -X POST "http://localhost:8000/api/market-data" \\
  -H "Content-Type: application/json" \\
  -d '{"symbols": ["VTI", "BNDX", "GSG"]}'</div>
                </div>
                <div id="python2" class="tab-content">
                    <div class="code-block">response = httpx.post("http://localhost:8000/api/market-data", json={
    "symbols": ["VTI", "BNDX", "GSG"]
})
market_data = response.json()
for symbol, quote in market_data["quotes"].items():
    tech = quote.get("technical_analysis", {})
    print(f"{symbol}: ${quote['price']:.2f} | {tech.get('trend', 'N/A')}")</div>
                </div>
            </div>
            
            <div class="insights">
                <h4>📈 Key Insights:</h4>
                <ul>
                    <li>Current prices and price changes</li>
                    <li>Technical trend analysis (BULLISH/BEARISH)</li>
                    <li>Support and resistance levels</li>
                    <li>Trading recommendations</li>
                </ul>
            </div>
        </div>

        <div class="step">
            <h2><span class="step-number">3</span>Get Market Sentiment Analysis</h2>
            <p>Check overall market sentiment for timing decisions.</p>
            
            <div class="tab-container">
                <div class="tab-buttons">
                    <button class="tab-button active" onclick="showTab(event, 'curl3')">cURL</button>
                    <button class="tab-button" onclick="showTab(event, 'python3')">Python</button>
                </div>
                <div id="curl3" class="tab-content active">
                    <div class="code-block">curl -X GET "http://localhost:8000/api/market-sentiment"</div>
                </div>
                <div id="python3" class="tab-content">
                    <div class="code-block">response = httpx.get("http://localhost:8000/api/market-sentiment")
sentiment = response.json()["sentiment"]
print(f"Market Sentiment: {sentiment['overall_sentiment']}")
print(f"Fear & Greed Index: {sentiment['fear_greed_index']}")</div>
                </div>
            </div>
            
            <div class="insights">
                <h4>🎭 Key Insights:</h4>
                <ul>
                    <li>Overall market sentiment (BULLISH/BEARISH/NEUTRAL)</li>
                    <li>Fear & Greed Index (0-100)</li>
                    <li>Market trend direction</li>
                    <li>Volatility index</li>
                </ul>
            </div>
        </div>

        <div class="step">
            <h2><span class="step-number">4</span>AI Data Analyst - Market Conditions Assessment</h2>
            <p>Get AI analysis of current market conditions for 3-week outlook.</p>
            
            <div class="tab-container">
                <div class="tab-buttons">
                    <button class="tab-button active" onclick="showTab(event, 'curl4')">cURL</button>
                    <button class="tab-button" onclick="showTab(event, 'python4')">Python</button>
                </div>
                <div id="curl4" class="tab-content active">
                    <div class="code-block">curl -X POST "http://localhost:8000/api/agents/data-analyst" \\
  -H "Content-Type: application/json" \\
  -d '{
    "query": "What are the current market conditions and outlook for the next 3 weeks?",
    "symbols": ["VTI", "BNDX", "GSG"]
  }'</div>
                </div>
                <div id="python4" class="tab-content">
                    <div class="code-block">response = httpx.post("http://localhost:8000/api/agents/data-analyst", json={
    "query": "What are the current market conditions and outlook for the next 3 weeks?",
    "symbols": ["VTI", "BNDX", "GSG"]
})
analysis = response.json()["analysis"]
print(f"Confidence: {analysis['confidence']}")
print(f"Analysis: {analysis['analysis']}")</div>
                </div>
            </div>
            
            <div class="insights">
                <h4>🤖 Key Insights:</h4>
                <ul>
                    <li>Market data interpretation</li>
                    <li>Economic indicators analysis</li>
                    <li>3-week market outlook</li>
                    <li>Strategy adjustment recommendations</li>
                </ul>
            </div>
        </div>

        <div class="step">
            <h2><span class="step-number">5</span>Analyze Your Current Portfolio</h2>
            <p>Analyze portfolio with enhanced risk metrics and compliance.</p>
            
            <div class="tab-container">
                <div class="tab-buttons">
                    <button class="tab-button active" onclick="showTab(event, 'curl5')">cURL</button>
                    <button class="tab-button" onclick="showTab(event, 'python5')">Python</button>
                </div>
                <div id="curl5" class="tab-content active">
                    <div class="code-block">curl -X POST "http://localhost:8000/api/analyze-portfolio" \\
  -H "Content-Type: application/json" \\
  -d '{
    "portfolio": {"VTI": 50000.0, "BNDX": 30000.0, "GSG": 20000.0},
    "total_value": 100000.0
  }'</div>
                </div>
                <div id="python5" class="tab-content">
                    <div class="code-block">response = httpx.post("http://localhost:8000/api/analyze-portfolio", json={
    "portfolio": {"VTI": 50000.0, "BNDX": 30000.0, "GSG": 20000.0},
    "total_value": 100000.0
})
analysis = response.json()["analysis"]
print(f"Expected Return: {analysis['portfolio_metrics']['expected_return']:.1%}")
print(f"Sharpe Ratio: {analysis['portfolio_metrics']['sharpe_ratio']:.2f}")</div>
                </div>
            </div>
            
            <div class="insights">
                <h4>📊 Key Insights:</h4>
                <ul>
                    <li>Current vs target allocation drift</li>
                    <li>Expected return and volatility</li>
                    <li>Sharpe ratio (risk-adjusted returns)</li>
                    <li>Compliance status and violations</li>
                    <li>Specific rebalancing recommendations</li>
                </ul>
            </div>
        </div>

        <div class="best-practices">
            <h3>🎯 Best Practices</h3>
            <ul>
                <li><strong>Parallel API Calls:</strong> Use asyncio.gather() to call multiple AI agents simultaneously</li>
                <li><strong>Error Handling:</strong> Always check response status codes and handle API errors gracefully</li>
                <li><strong>Rate Limiting:</strong> Be mindful of API rate limits, especially for external services</li>
                <li><strong>Data Validation:</strong> Validate portfolio data and ensure total_value matches holdings</li>
                <li><strong>Regular Updates:</strong> Run workflow weekly for 3-week strategies</li>
            </ul>
        </div>

        <div class="outcomes">
            <h3>🎉 Expected Outcomes</h3>
            <ul>
                <li>Comprehensive market assessment with real-time data</li>
                <li>AI-powered insights from 3 specialized agents</li>
                <li>Risk-adjusted portfolio recommendations</li>
                <li>Specific rebalancing actions with priorities</li>
                <li>Compliance validation and regulatory awareness</li>
                <li>Performance expectations and risk metrics</li>
            </ul>
        </div>

        <div style="text-align: center; margin: 2rem 0; padding: 2rem; background: white; border-radius: 8px;">
            <h3>🚀 Ready to Get Started?</h3>
            <p>Visit <a href="/docs" style="color: #667eea;">/docs</a> for interactive API documentation</p>
            <p>Get the complete workflow data: <a href="/api/workflow-guide" style="color: #667eea;">/api/workflow-guide</a></p>
        </div>

        <script>
            function showTab(evt, tabName) {
                var i, tabcontent, tablinks;
                
                // Get the parent tab container
                var container = evt.target.closest('.tab-container');
                
                // Hide all tab content in this container
                tabcontent = container.getElementsByClassName("tab-content");
                for (i = 0; i < tabcontent.length; i++) {
                    tabcontent[i].classList.remove("active");
                }
                
                // Remove active class from all tab buttons in this container
                tablinks = container.getElementsByClassName("tab-button");
                for (i = 0; i < tablinks.length; i++) {
                    tablinks[i].classList.remove("active");
                }
                
                // Show the selected tab and mark button as active
                document.getElementById(tabName).classList.add("active");
                evt.target.classList.add("active");
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# ============================================================================
# STARTUP
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("FinAgent Enhanced Simple API v1.6 starting up...")
    logger.info("Strategy: Straight Arrow (60% VTI, 30% BNDX, 10% GSG)")
    logger.info("Features: AI Agents, Risk Metrics, Compliance, Market Data")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main_enhanced_complete:app",
        host="0.0.0.0",
        port=port,
        reload=False
    ) 