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

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

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
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

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

class TickerRequest(BaseModel):
    """Single ticker request"""
    symbol: str = Field(..., description="Stock ticker symbol", min_length=1, max_length=10)

class TickerResponse(BaseModel):
    """Single ticker response"""
    symbol: str = Field(..., description="Stock ticker symbol")
    price: float = Field(..., description="Current stock price")
    change: float = Field(..., description="Price change")
    change_percent: str = Field(..., description="Percentage change")
    high: float = Field(..., description="Day's high price")
    low: float = Field(..., description="Day's low price")
    previous_close: float = Field(..., description="Previous day's closing price")
    timestamp: str = Field(..., description="Response timestamp")
    source: str = Field(..., description="Data source")

class BulkTickerRequest(BaseModel):
    """Bulk ticker request for multiple symbols"""
    symbols: List[str] = Field(..., description="List of stock ticker symbols", min_items=1, max_items=20)

class BulkTickerResponse(BaseModel):
    """Bulk ticker response for multiple symbols"""
    success_count: int = Field(..., description="Number of successfully fetched symbols")
    error_count: int = Field(..., description="Number of symbols that failed")
    total_requested: int = Field(..., description="Total number of symbols requested")
    tickers: Dict[str, TickerResponse] = Field(..., description="Dictionary of ticker responses keyed by symbol")
    errors: Dict[str, str] = Field(default={}, description="Dictionary of error messages keyed by symbol")
    timestamp: str = Field(..., description="Response timestamp")
    source: str = Field(..., description="Data source")

class AIAnalysisRequest(BaseModel):
    """AI analysis request"""
    query: str = Field(..., description="User query")
    symbols: Optional[List[str]] = Field(default=[], description="Relevant symbols")

class QuestionnaireRequest(BaseModel):
    """Questionnaire analysis request"""
    questionnaire: str = Field(..., description="Stringified questionnaire JSON")

class QuestionnaireResponse(BaseModel):
    """Questionnaire analysis response"""
    risk_score: int = Field(..., description="Risk score from 1-5")
    risk_level: str = Field(..., description="Risk level description")
    portfolio_strategy_name: str = Field(..., description="Portfolio strategy name")
    analysis_details: Dict[str, Any] = Field(..., description="Detailed analysis")

class RiskAnalysisRequest(BaseModel):
    """Risk analysis request"""
    portfolio: Dict[str, float] = Field(..., description="Portfolio holdings")
    total_value: float = Field(..., gt=0, description="Total portfolio value")
    time_horizon: Optional[str] = Field(default="Long Term", description="Investment time horizon")

class TradingAnalysisRequest(BaseModel):
    """Trading analysis request"""
    symbols: List[str] = Field(..., description="Symbols to analyze")
    analysis_type: Optional[str] = Field(default="technical", description="Type of analysis")

class UnifiedStrategyRequest(BaseModel):
    """Enhanced unified investment strategy request with questionnaire integration"""
    # Questionnaire-derived risk attributes (required)
    risk_score: int = Field(..., ge=1, le=5, description="Risk score from questionnaire (1-5)")
    risk_level: str = Field(..., description="Risk level from questionnaire")
    portfolio_strategy_name: str = Field(..., description="Portfolio strategy name from questionnaire")
    
    # Investment parameters
    investment_amount: float = Field(..., gt=0, description="Cash amount available for investment")
    investment_restrictions: Optional[List[str]] = Field(default=[], description="Investment restrictions from questionnaire")
    sector_preferences: Optional[List[str]] = Field(default=[], description="Preferred sectors from questionnaire")
    
    # Optional questionnaire context
    time_horizon: Optional[str] = Field(default="5-10 years", description="Investment time horizon from questionnaire")
    experience_level: Optional[str] = Field(default="Some experience", description="Investment experience level")
    liquidity_needs: Optional[str] = Field(default="20-40% accessible", description="Liquidity requirements")
    
    # Current portfolio (optional for new investments)
    current_portfolio: Optional[Dict[str, float]] = Field(default={}, description="Existing portfolio holdings")
    current_portfolio_value: Optional[float] = Field(default=0.0, description="Current portfolio total value")
    
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
        self.finnhub_key = FINNHUB_API_KEY
    
    async def get_quotes(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get quotes for symbols with enhanced data"""
        if not self.finnhub_key:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "Finnhub API key not configured",
                    "message": "Real market data unavailable - no Finnhub API key configured",
                    "symbols_requested": symbols
                }
            )
        
        quotes = {}
        errors = []
        
        for symbol in symbols:
            quote = await self._fetch_quote_finnhub(symbol)
            if quote:
                # Add technical indicators
                quote["technical_analysis"] = self._basic_technical_analysis(quote)
                quotes[symbol] = quote
                
                # Track errors for symbols that failed
                if quote.get("source") == "mock" and "error" in quote:
                    errors.append(f"{symbol}: {quote['error']}")
            else:
                error_msg = f"{symbol}: Unknown error fetching data"
                errors.append(error_msg)
                quotes[symbol] = {
                    "symbol": symbol,
                    "error": "Failed to fetch data",
                    "source": "error",
                    "timestamp": datetime.now().isoformat()
                }
        
        # If all symbols failed, raise an error
        if all(quote.get("source") in ["mock", "error"] for quote in quotes.values()):
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "All market data requests failed",
                    "message": "Unable to fetch real market data for any requested symbols",
                    "symbols_requested": symbols,
                    "errors": errors
                }
            )
        
        return quotes
    
    async def _fetch_quote_finnhub(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch quote from Finnhub API"""
        url = "https://finnhub.io/api/v1/quote"
        params = {
            "symbol": symbol,
            "token": self.finnhub_key
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Finnhub returns: {"c": current_price, "d": change, "dp": change_percent, "h": high, "l": low, "o": open, "pc": previous_close}
                        if data.get("c") and data.get("c") > 0:  # 'c' is current price
                            current_price = float(data.get("c", 0))
                            change = float(data.get("d", 0))  # 'd' is change
                            change_percent = f"{data.get('dp', 0):.2f}%"  # 'dp' is change percent
                            high = float(data.get("h", current_price))  # 'h' is high
                            low = float(data.get("l", current_price))   # 'l' is low
                            previous_close = float(data.get("pc", current_price))  # 'pc' is previous close
                            
                            return {
                                "symbol": symbol,
                                "price": current_price,
                                "change": change,
                                "change_percent": change_percent,
                                "volume": 0,  # Finnhub basic quote doesn't include volume
                                "high": high,
                                "low": low,
                                "previous_close": previous_close,
                                "timestamp": datetime.now().isoformat(),
                                "source": "finnhub"
                            }
                        else:
                            # Check for error in response
                            error_msg = f"Finnhub returned invalid data for {symbol}"
                            if "error" in data:
                                error_msg = f"Finnhub error: {data['error']}"
                            
                            return {
                                "symbol": symbol,
                                "price": 0,
                                "change": 0,
                                "change_percent": "0%",
                                "volume": 0,
                                "high": 0,
                                "low": 0,
                                "timestamp": datetime.now().isoformat(),
                                "source": "mock",
                                "error": error_msg
                            }
                    else:
                        error_msg = f"Finnhub API error: HTTP {response.status}"
                        if response.status == 429:
                            error_msg = "Finnhub API rate limit exceeded"
                        elif response.status == 401:
                            error_msg = "Finnhub API authentication failed - check API key"
                        
                        return {
                            "symbol": symbol,
                            "price": 0,
                            "change": 0,
                            "change_percent": "0%",
                            "volume": 0,
                            "high": 0,
                            "low": 0,
                            "timestamp": datetime.now().isoformat(),
                            "source": "mock",
                            "error": error_msg
                        }
                        
        except Exception as e:
            logger.error(f"Finnhub API error for {symbol}: {e}")
            return {
                "symbol": symbol,
                "price": 0,
                "change": 0,
                "change_percent": "0%",
                "volume": 0,
                "high": 0,
                "low": 0,
                "timestamp": datetime.now().isoformat(),
                "source": "mock",
                "error": f"Network error: {str(e)}"
            }
    
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
        """Get overall market sentiment using real market data"""
        # Get VIX (volatility index) from Alpha Vantage
        vix_data = await self._fetch_vix_data()
        fear_greed_data = await self._fetch_fear_greed_index()
        
        # Check if we have real data
        vix_is_real = vix_data.get("source") != "mock"
        fear_greed_is_real = fear_greed_data.get("source") != "mock"
        
        if not vix_is_real and not fear_greed_is_real:
            # No real data available - return error
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "Market sentiment data unavailable",
                    "message": "Unable to fetch real market data from any source",
                    "vix_error": vix_data.get("error", "No UVXY volatility data available"),
                    "fear_greed_error": fear_greed_data.get("error", "No market indicators available"),
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        # Calculate overall sentiment based on available real data
        overall_sentiment = self._calculate_overall_sentiment(vix_data, fear_greed_data)
        market_trend = self._determine_market_trend(vix_data, fear_greed_data)
        
        return {
            "overall_sentiment": overall_sentiment,
            "fear_greed_index": fear_greed_data.get("value", 50),
            "fear_greed_text": fear_greed_data.get("text", "Neutral"),
            "market_trend": market_trend,
            "volatility_index": vix_data.get("price", 20),
            "vix_change": vix_data.get("change", 0),
            "data_sources": {
                "vix_source": vix_data.get("source", "unavailable"),
                "fear_greed_source": fear_greed_data.get("source", "unavailable")
            },
            "data_quality": {
                "vix_available": vix_is_real,
                "fear_greed_available": fear_greed_is_real,
                "overall_quality": "partial" if (vix_is_real != fear_greed_is_real) else "complete"
            },
            "timestamp": datetime.now().isoformat(),
            "source": "real_market_data"
        }
    
    async def _fetch_vix_data(self) -> Dict[str, Any]:
        """Fetch volatility data using VIX directly from Finnhub"""
        if not self.finnhub_key:
            return {"price": 20, "change": 0, "source": "mock", "error": "No Finnhub API key configured"}
            
        try:
            # Try to get VIX directly from Finnhub
            vix_quote = await self._fetch_quote_finnhub("^VIX")
            if vix_quote and vix_quote.get("source") == "finnhub":
                return {
                    "price": vix_quote.get("price", 20),
                    "change": vix_quote.get("change", 0),
                    "change_percent": vix_quote.get("change_percent", "0%"),
                    "source": "finnhub_vix",
                    "high": vix_quote.get("high"),
                    "low": vix_quote.get("low")
                }
            else:
                # Fallback to UVXY as volatility proxy if VIX not available
                uvxy_quote = await self._fetch_quote_finnhub("UVXY")
                if uvxy_quote and uvxy_quote.get("source") == "finnhub":
                    uvxy_price = uvxy_quote.get("price", 10)
                    uvxy_change = uvxy_quote.get("change", 0)
                    
                    # Convert UVXY to approximate VIX equivalent
                    # UVXY typically trades 10-30, VIX typically 10-80
                    # Rough conversion: VIX ≈ UVXY * 1.5 + 5
                    estimated_vix = (uvxy_price * 1.5) + 5
                    estimated_vix_change = uvxy_change * 1.5
                    
                    return {
                        "price": round(estimated_vix, 2),
                        "change": round(estimated_vix_change, 2),
                        "change_percent": uvxy_quote.get("change_percent", "0%"),
                        "source": "finnhub_uvxy_proxy",
                        "raw_uvxy_price": uvxy_price
                    }
                else:
                    error_msg = uvxy_quote.get("error", "Failed to get volatility data") if uvxy_quote else "No volatility data available"
                    return {"price": 20, "change": 0, "source": "mock", "error": error_msg}
                        
        except Exception as e:
            logger.error(f"Failed to fetch volatility data: {e}")
            return {"price": 20, "change": 0, "source": "mock", "error": f"Network error: {str(e)}"}
    
    async def _fetch_fear_greed_index(self) -> Dict[str, Any]:
        """Calculate Fear & Greed Index based on market indicators"""
        try:
            # Use multiple market indicators to calculate our own Fear & Greed Index
            # Get key market data points
            market_indicators = await self._get_market_indicators_for_sentiment()
            
            if market_indicators.get("source") != "mock":
                # Calculate composite fear/greed score
                fear_greed_value = self._calculate_fear_greed_score(market_indicators)
                
                # Determine text classification
                if fear_greed_value <= 25:
                    text = "Extreme Fear"
                elif fear_greed_value <= 45:
                    text = "Fear"
                elif fear_greed_value <= 55:
                    text = "Neutral"
                elif fear_greed_value <= 75:
                    text = "Greed"
                else:
                    text = "Extreme Greed"
                    
                return {
                    "value": fear_greed_value,
                    "text": text,
                    "components": market_indicators.get("components", {}),
                    "source": "calculated_from_market_data"
                }
                            
        except Exception as e:
            logger.error(f"Failed to calculate Fear & Greed Index: {e}")
            
        return {"value": 50, "text": "Neutral", "source": "mock"}
    
    async def _get_market_indicators_for_sentiment(self) -> Dict[str, Any]:
        """Get key market indicators for sentiment calculation"""
        try:
            # Get data for key market indicators
            symbols = ["SPY", "QQQ", "VTI"]  # Major market ETFs
            market_data = {}
            errors = []
            
            for symbol in symbols:
                quote = await self._fetch_quote_finnhub(symbol)
                if quote and quote.get("source") != "mock":
                    market_data[symbol] = quote
                else:
                    error_msg = f"Failed to get {symbol} data"
                    if quote and "error" in quote:
                        error_msg = f"{symbol}: {quote['error']}"
                    errors.append(error_msg)
            
            if market_data:
                return {
                    "market_data": market_data,
                    "source": "finnhub",
                    "components": self._extract_sentiment_components(market_data),
                    "partial_errors": errors if errors else None
                }
            else:
                return {
                    "source": "mock", 
                    "error": f"No market data available for any symbols: {'; '.join(errors)}"
                }
                
        except Exception as e:
            logger.error(f"Failed to get market indicators: {e}")
            return {"source": "mock", "error": f"Network error: {str(e)}"}
    
    def _extract_sentiment_components(self, market_data: Dict) -> Dict[str, Any]:
        """Extract sentiment components from market data"""
        components = {}
        
        for symbol, data in market_data.items():
            change_percent_str = data.get("change_percent", "0%")
            # Extract numeric value from percentage string like "+1.5%" or "-0.8%"
            try:
                change_percent = float(change_percent_str.replace("%", "").replace("+", ""))
                components[f"{symbol}_change_percent"] = change_percent
            except:
                components[f"{symbol}_change_percent"] = 0
                
        return components
    
    def _calculate_fear_greed_score(self, market_indicators: Dict) -> int:
        """Calculate Fear & Greed score from market indicators"""
        components = market_indicators.get("components", {})
        
        # Start with neutral (50)
        score = 50
        
        # Market momentum component (based on major ETF performance)
        spy_change = components.get("SPY_change_percent", 0)
        qqq_change = components.get("QQQ_change_percent", 0) 
        vti_change = components.get("VTI_change_percent", 0)
        
        # Average market performance
        avg_market_change = (spy_change + qqq_change + vti_change) / 3
        
        # Convert market performance to sentiment score
        # Strong positive performance -> Greed (higher score)
        # Strong negative performance -> Fear (lower score)
        if avg_market_change > 2:
            score += 25  # Strong positive = greed
        elif avg_market_change > 1:
            score += 15  # Moderate positive = mild greed
        elif avg_market_change > 0:
            score += 5   # Slight positive = mild optimism
        elif avg_market_change > -1:
            score -= 5   # Slight negative = mild pessimism
        elif avg_market_change > -2:
            score -= 15  # Moderate negative = fear
        else:
            score -= 25  # Strong negative = extreme fear
            
        # Ensure score stays within 0-100 range
        return max(0, min(100, score))
    
    def _calculate_overall_sentiment(self, vix_data: Dict, fear_greed_data: Dict) -> str:
        """Calculate overall market sentiment based on VIX and Fear & Greed Index"""
        vix_price = vix_data.get("price", 20)
        fear_greed = fear_greed_data.get("value", 50)
        
        # VIX interpretation (lower = less fear)
        # VIX < 15: Low volatility/complacency
        # VIX 15-25: Normal volatility  
        # VIX > 25: High volatility/fear
        
        sentiment_score = 0
        
        # VIX component (40% weight)
        if vix_price < 15:
            sentiment_score += 20  # Bullish (low fear)
        elif vix_price < 25:
            sentiment_score += 0   # Neutral
        else:
            sentiment_score -= 20  # Bearish (high fear)
            
        # Fear & Greed component (60% weight)
        if fear_greed > 55:
            sentiment_score += 30  # Bullish (greed)
        elif fear_greed > 45:
            sentiment_score += 0   # Neutral
        else:
            sentiment_score -= 30  # Bearish (fear)
            
        # Determine overall sentiment
        if sentiment_score > 15:
            return "BULLISH"
        elif sentiment_score < -15:
            return "BEARISH"
        else:
            return "NEUTRAL"
    
    def _determine_market_trend(self, vix_data: Dict, fear_greed_data: Dict) -> str:
        """Determine market trend based on volatility and sentiment"""
        vix_price = vix_data.get("price", 20)
        vix_change = vix_data.get("change", 0)
        fear_greed = fear_greed_data.get("value", 50)
        
        # Rising VIX = increasing fear/uncertainty
        # Falling VIX = decreasing fear
        # Fear & Greed extremes can indicate trend reversals
        
        if vix_change > 2 and vix_price > 25:
            return "VOLATILE_DOWN"  # High volatility, rising fear
        elif vix_change < -2 and fear_greed > 60:
            return "TRENDING_UP"    # Decreasing fear, greed rising
        elif fear_greed > 75:
            return "OVERHEATED"     # Extreme greed - caution
        elif fear_greed < 25:
            return "OVERSOLD"       # Extreme fear - potential opportunity
        elif vix_price > 30:
            return "HIGH_VOLATILITY"
        elif vix_price < 15:
            return "LOW_VOLATILITY"
        else:
            return "SIDEWAYS"

# Continue in next part... # ============================================================================
# INVESTMENT PROFILE TEMPLATES
# ============================================================================

INVESTMENT_PROFILE_TEMPLATES = {
    1: {
        "name": "Ultra Conservative Capital Preservation",
        "core_strategy": "Capital Preservation + Inflation Hedge",
        "base_allocation": {"SGOV": 50, "VPU": 30, "TIPS": 20},
        "risk_controls": {"RealYieldTracking": True, "MaxDrawdown": "<3%", "TreasuryRoll": "6mo"},
        "risk_level": "Very Low",
        "description": "Ultra-conservative approach focused on capital preservation and inflation protection",
        "asset_allocation": "50% Short-term Treasury, 30% Utilities, 20% Inflation-Protected Bonds",
        "investment_focus": "Capital preservation with inflation hedge protection",
        "recommended_products": ["Short-term Treasury ETFs (SGOV)", "Utilities ETF (VPU)", "TIPS", "High-grade bonds"],
        "time_horizon_fit": "Short to medium term with capital preservation priority",
        "volatility_expectation": "<3% annual volatility",
        "expected_return": "3-5% annually (inflation-adjusted)"
    },
    2: {
        "name": "Conservative Balanced Growth", 
        "core_strategy": "Diversified Three-Fund Style",
        "base_allocation": {"VTI": 60, "BNDX": 30, "GSG": 10},
        "risk_controls": {"Sharpe": ">0.5", "VolatilityCap": "<12%", "TaxLossHarvesting": True},
        "risk_level": "Low",
        "description": "Classic diversified approach with balanced growth and income",
        "asset_allocation": "60% Total Stock Market, 30% International Bonds, 10% Commodities",
        "investment_focus": "Balanced growth and income with global diversification",
        "recommended_products": ["Total Stock Market ETF (VTI)", "International Bond ETF (BNDX)", "Commodities ETF (GSG)"],
        "time_horizon_fit": "Medium to long term with moderate risk tolerance",
        "volatility_expectation": "10-12% annual volatility",
        "expected_return": "6-8% annually over long term"
    },
    3: {
        "name": "Moderate Growth with Value Focus",
        "core_strategy": "Graham-Buffett Value + Momentum Tilt",
        "base_allocation": {"VTI": 70, "VTV": 15, "MTUM": 10, "VMOT": 5},
        "risk_controls": {"P/E": "<20", "Dividend": ">2%", "Drift Rebalance": "5%"},
        "risk_level": "Moderate",
        "description": "Value-focused approach with momentum tilt and selective sector exposure",
        "asset_allocation": "70% Total Market, 15% Value ETFs, 10% Momentum, 5% Cash Buffer",
        "investment_focus": "Value investing with momentum signals and tactical allocation",
        "recommended_products": ["Total Market ETF (VTI)", "Value ETFs (VTV)", "Momentum ETFs (MTUM)", "Dividend ETFs"],
        "time_horizon_fit": "Long term with active management overlay",
        "volatility_expectation": "12-15% annual volatility", 
        "expected_return": "7-10% annually with value tilt"
    },
    4: {
        "name": "Aggressive Growth with Trend Following",
        "core_strategy": "Social Sentiment Mirroring + Growth Tilt",
        "base_allocation": {"QQQ": 40, "VUG": 30, "ARKK": 20, "VMOT": 10},
        "risk_controls": {"DelayTrades": "24h", "SocialSentiment": True, "BuzzVolumeMonitor": True},
        "risk_level": "High", 
        "description": "Trend-following approach using growth stocks and innovation themes",
        "asset_allocation": "40% NASDAQ, 30% Growth ETFs, 20% Innovation ETFs, 10% Momentum",
        "investment_focus": "Growth-oriented with trend following and innovation exposure",
        "recommended_products": ["NASDAQ ETF (QQQ)", "Growth ETF (VUG)", "Innovation ETF (ARKK)", "Momentum ETFs"],
        "time_horizon_fit": "Medium to long term with active trend monitoring",
        "volatility_expectation": "15-20% annual volatility",
        "expected_return": "8-12% annually with higher variance"
    },
    5: {
        "name": "Maximum Growth High-Risk Portfolio",
        "core_strategy": "Barbell: Stability + High Volatility Growth", 
        "base_allocation": {"BND": 30, "TQQQ": 25, "SOXL": 20, "ARKK": 15, "SPXL": 10},
        "risk_controls": {"VolatilityCap": "<40%", "AutoSellDrop": "15%", "DrawdownCap": "<25%"},
        "risk_level": "Very High",
        "description": "High-risk strategy combining stable bonds with leveraged growth exposure",
        "asset_allocation": "30% Bonds, 25% 3x Tech, 20% 3x Semiconductors, 15% Innovation, 10% 3x S&P",
        "investment_focus": "Maximum growth potential with leveraged exposure and sector concentration",
        "recommended_products": ["Bond ETF (BND)", "3x Leveraged Tech (TQQQ)", "3x Semiconductors (SOXL)", "Innovation (ARKK)"],
        "time_horizon_fit": "Long term with high risk tolerance and volatility acceptance",
        "volatility_expectation": "25-40% annual volatility",
        "expected_return": "10-15% annually with high risk/reward"
    }
}

# ============================================================================
# AI AGENTS
# ============================================================================

class AIAgentService:
    """Multiple AI agents for different analysis types"""
    
    def __init__(self):
        self.openai_key = OPENAI_API_KEY
    
    async def data_analyst(self, query: str, symbols: List[str] = None) -> Dict[str, Any]:
        """Data analyst agent - market data and fundamental analysis"""
        
        if not self.openai_key:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "OpenAI API key not configured",
                    "message": "AI analysis unavailable - no OpenAI API key configured",
                    "agent_type": "data_analyst"
                }
            )
        
        context = f"""You are a financial data analyst AI. Analyze the following query with focus on:
        - Market data interpretation
        - Fundamental analysis
        - Economic indicators
        - Data-driven insights
        
        Query: {query}
        Symbols: {', '.join(symbols) if symbols else 'General market'}
        
        Provide educational analysis focusing on the Straight Arrow strategy (60% VTI, 30% BNDX, 10% GSG).
        Include appropriate disclaimers."""
        
        # This will raise HTTPException on error instead of falling back
        response = await self._call_openai(context, "data_analyst")
        
        return {
            "agent": "data_analyst",
            "query": query,
            "analysis": response,
            "symbols": symbols or [],
            "confidence": 0.85,
            "timestamp": datetime.now().isoformat(),
            "source": "openai_gpt3.5"
        }
    
    async def risk_analyst(self, portfolio: Dict[str, float], total_value: float, time_horizon: str = "Long Term") -> Dict[str, Any]:
        """Risk analyst agent - risk assessment and management"""
        
        if not self.openai_key:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "OpenAI API key not configured",
                    "message": "AI risk analysis unavailable - no OpenAI API key configured",
                    "agent_type": "risk_analyst"
                }
            )
        
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
        
        # This will raise HTTPException on error instead of falling back
        response = await self._call_openai(context, "risk_analyst")
        
        return {
            "agent": "risk_analyst",
            "portfolio": weights,
            "total_value": total_value,
            "time_horizon": time_horizon,
            "analysis": response,
            "confidence": 0.80,
            "timestamp": datetime.now().isoformat(),
            "source": "openai_gpt3.5"
        }
    
    async def trading_analyst(self, symbols: List[str], analysis_type: str = "technical") -> Dict[str, Any]:
        """Trading analyst agent - technical analysis and trading signals"""
        
        if not self.openai_key:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "OpenAI API key not configured",
                    "message": "AI trading analysis unavailable - no OpenAI API key configured",
                    "agent_type": "trading_analyst"
                }
            )
        
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
        
        # This will raise HTTPException on error instead of falling back
        response = await self._call_openai(context, "trading_analyst")
        
        return {
            "agent": "trading_analyst",
            "symbols": symbols,
            "analysis_type": analysis_type,
            "analysis": response,
            "confidence": 0.75,
            "timestamp": datetime.now().isoformat(),
            "source": "openai_gpt3.5"
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
                elif response.status_code == 429:
                    raise HTTPException(
                        status_code=429,
                        detail={
                            "error": "OpenAI API rate limit exceeded",
                            "message": "Too many requests to OpenAI API. Please try again later.",
                            "agent_type": agent_type,
                            "retry_after": response.headers.get("retry-after", "60 seconds")
                        }
                    )
                else:
                    response_text = await response.aread() if hasattr(response, 'aread') else str(response.content)
                    raise HTTPException(
                        status_code=502,
                        detail={
                            "error": f"OpenAI API error: HTTP {response.status_code}",
                            "message": "Failed to get AI analysis from OpenAI",
                            "agent_type": agent_type,
                            "api_response": response_text[:200] if response_text else "No response content"
                        }
                    )
                    
        except HTTPException:
            raise  # Re-raise HTTP exceptions
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "OpenAI API connection failed",
                    "message": f"Network error connecting to OpenAI: {str(e)}",
                    "agent_type": agent_type
                }
            )
    
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
    
    async def analyze_questionnaire(self, questionnaire_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze questionnaire and determine risk profile and portfolio strategy"""
        try:
            # Calculate risk score based on questionnaire responses
            risk_score = self._calculate_risk_score(questionnaire_data)
            
            # Determine risk level
            risk_level = self._determine_risk_level(risk_score)
            
            # Generate portfolio strategy name
            portfolio_strategy_name = self._generate_portfolio_strategy_name(risk_score, risk_level)
            
            # Get detailed analysis from AI if available
            context = f"""
            Analyze this investment questionnaire to provide detailed insights:
            
            Investment Goal: {questionnaire_data.get('investment_goal', 'Not specified')}
            Time Horizon: {questionnaire_data.get('time_horizon', 'Not specified')}
            Risk Tolerance: {questionnaire_data.get('risk_tolerance', 'Not specified')}
            Experience Level: {questionnaire_data.get('experience_level', 'Not specified')}
            Income Level: {questionnaire_data.get('income_level', 'Not specified')}
            Net Worth: {questionnaire_data.get('net_worth', 'Not specified')}
            Liquidity Needs: {questionnaire_data.get('liquidity_needs', 'Not specified')}
            
            Calculated Risk Score: {risk_score}/5
            Risk Level: {risk_level}
            
            Provide detailed analysis explaining why this risk profile is appropriate and what investment strategy considerations apply.
            """
            
            # Get AI analysis if OpenAI is available, otherwise provide basic analysis
            if self.openai_key:
                try:
                    analysis_details = await self._call_openai(context, "questionnaire_analyst")
                except HTTPException as e:
                    # For questionnaire analysis, we can fall back to basic analysis since 
                    # the core risk scoring doesn't depend on AI
                    analysis_details = f"AI analysis unavailable ({e.detail.get('error', 'Unknown error')}). Risk score calculation completed using questionnaire responses."
            else:
                analysis_details = "AI analysis unavailable - no OpenAI API key configured. Risk score calculated from questionnaire responses."
            
            return {
                "risk_score": risk_score,
                "risk_level": risk_level,
                "portfolio_strategy_name": portfolio_strategy_name,
                "analysis_details": {
                    "detailed_analysis": analysis_details,
                    "questionnaire_breakdown": self._breakdown_questionnaire_factors(questionnaire_data),
                    "investment_recommendations": self._generate_investment_recommendations(risk_score, questionnaire_data),
                    "strategy_rationale": self._generate_strategy_rationale(risk_score, risk_level, questionnaire_data)
                },
                "confidence": 0.85 if self.openai_key else 0.7,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Questionnaire analysis error: {e}")
            default_template = INVESTMENT_PROFILE_TEMPLATES[3]  # Individualist as default
            return {
                "risk_score": 3,  # Default moderate risk
                "risk_level": default_template["risk_level"],
                "portfolio_strategy_name": default_template["name"],
                "analysis_details": {
                    "error": str(e),
                    "detailed_analysis": f"Unable to complete full analysis due to error. Default {default_template['name']} profile assigned.",
                    "investment_recommendations": self._generate_investment_recommendations(3, {}),
                    "strategy_rationale": f"Default assignment to {default_template['name']} due to analysis error."
                },
                "confidence": 0.3,
                "timestamp": datetime.now().isoformat()
            }
    
    def _calculate_risk_score(self, questionnaire_data: Dict[str, Any]) -> int:
        """Calculate risk score from 1-5 based on questionnaire responses"""
        score_components = []
        
        # 1. Investment Goal (20% weight)
        goal = questionnaire_data.get('investment_goal', '').lower().replace('_', ' ')
        if 'preserve capital' in goal or 'capital preservation' in goal:
            score_components.append(1)
        elif 'income' in goal:
            score_components.append(2)
        elif 'major purchase' in goal or 'balanced growth' in goal:
            score_components.append(3)
        elif 'retirement' in goal:
            score_components.append(3)
        elif 'wealth accumulation' in goal:
            score_components.append(4)
        else:
            score_components.append(3)  # Default
        
        # 2. Time Horizon (25% weight)
        horizon = questionnaire_data.get('time_horizon', '').lower().replace('_', ' ')
        if 'less than 1' in horizon or 'under 1' in horizon:
            score_components.append(1)
        elif '1 to 3' in horizon or '1-3' in horizon:
            score_components.append(2)
        elif '3 to 5' in horizon or '3-5' in horizon:
            score_components.append(3)
        elif '5 to 10' in horizon or '5-10' in horizon:
            score_components.append(4)
        elif 'more than 10' in horizon or '10+' in horizon or 'over 10' in horizon:
            score_components.append(5)
        else:
            score_components.append(3)  # Default
        
        # 3. Risk Tolerance - Portfolio Decline Reaction (30% weight)
        risk_tolerance = questionnaire_data.get('risk_tolerance', '').lower().replace('_', ' ')
        if 'sell everything' in risk_tolerance:
            score_components.append(1)
        elif 'accept small losses' in risk_tolerance or 'small losses' in risk_tolerance:
            score_components.append(2)
        elif 'moderate losses' in risk_tolerance or 'hold and wait' in risk_tolerance:
            score_components.append(3)
        elif 'accept significant losses' in risk_tolerance or 'buy more' in risk_tolerance:
            score_components.append(4)
        elif 'excited about opportunity' in risk_tolerance or 'excited about' in risk_tolerance:
            score_components.append(5)
        else:
            score_components.append(3)  # Default
        
        # 4. Experience Level (15% weight)
        experience = questionnaire_data.get('experience_level', '').lower().replace('_', ' ')
        if 'no experience' in experience or 'no investment' in experience:
            score_components.append(1)
        elif 'basic' in experience:
            score_components.append(2)
        elif 'some experience' in experience:
            score_components.append(3)
        elif 'experienced' in experience:
            score_components.append(4)
        elif 'professional' in experience:
            score_components.append(5)
        else:
            score_components.append(3)  # Default
        
        # 5. Liquidity Needs (10% weight) - inverse relationship to risk
        liquidity = questionnaire_data.get('liquidity_needs', '').lower().replace('_', ' ')
        if 'more than 60' in liquidity or '60+' in liquidity:
            score_components.append(1)
        elif '40 to 60' in liquidity or '40-60' in liquidity:
            score_components.append(2)
        elif '20 to 40' in liquidity or '20-40' in liquidity:
            score_components.append(3)
        elif '10 to 20' in liquidity or '10-20' in liquidity:
            score_components.append(4)
        elif 'less than 20' in liquidity or 'none accessible' in liquidity or 'none' in liquidity:
            score_components.append(5)
        else:
            score_components.append(3)  # Default
        
        # Calculate weighted average with weights [0.2, 0.25, 0.3, 0.15, 0.1]
        weights = [0.2, 0.25, 0.3, 0.15, 0.1]
        if len(score_components) >= 5:
            weighted_score = sum(score * weight for score, weight in zip(score_components[:5], weights))
        else:
            # Fallback to simple average
            weighted_score = sum(score_components) / len(score_components) if score_components else 3
        
        # Round to nearest integer and ensure in range 1-5
        final_score = max(1, min(5, round(weighted_score)))
        return final_score
    
    def _determine_risk_level(self, risk_score: int) -> str:
        """Map risk score to risk level description from investment profile templates"""
        template = INVESTMENT_PROFILE_TEMPLATES.get(risk_score, INVESTMENT_PROFILE_TEMPLATES[3])
        return template["risk_level"]
    
    def _generate_portfolio_strategy_name(self, risk_score: int, risk_level: str) -> str:
        """Generate portfolio strategy name from investment profile templates"""
        template = INVESTMENT_PROFILE_TEMPLATES.get(risk_score, INVESTMENT_PROFILE_TEMPLATES[3])
        return template["name"]
    
    def _breakdown_questionnaire_factors(self, questionnaire_data: Dict[str, Any]) -> Dict[str, Any]:
        """Provide breakdown of key questionnaire factors"""
        return {
            "primary_factors": {
                "investment_goal": questionnaire_data.get('investment_goal', 'Not specified'),
                "time_horizon": questionnaire_data.get('time_horizon', 'Not specified'),
                "risk_tolerance": questionnaire_data.get('risk_tolerance', 'Not specified')
            },
            "supporting_factors": {
                "experience_level": questionnaire_data.get('experience_level', 'Not specified'),
                "income_level": questionnaire_data.get('income_level', 'Not specified'),
                "net_worth": questionnaire_data.get('net_worth', 'Not specified'),
                "liquidity_needs": questionnaire_data.get('liquidity_needs', 'Not specified')
            },
            "preferences": {
                "sector_preferences": questionnaire_data.get('sector_preferences', []),
                "investment_restrictions": questionnaire_data.get('investment_restrictions', []),
                "market_insights": questionnaire_data.get('market_insights', 'Not specified')
            }
        }
    
    def _generate_investment_recommendations(self, risk_score: int, questionnaire_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate specific investment recommendations based on risk score from templates"""
        template = INVESTMENT_PROFILE_TEMPLATES.get(risk_score, INVESTMENT_PROFILE_TEMPLATES[3])
        
        return {
            "profile_name": template["name"],
            "core_strategy": template["core_strategy"], 
            "asset_allocation": template["asset_allocation"],
            "base_allocation": template["base_allocation"],
            "investment_focus": template["investment_focus"],
            "recommended_products": template["recommended_products"],
            "time_horizon_fit": template["time_horizon_fit"],
            "volatility_expectation": template["volatility_expectation"],
            "expected_return": template["expected_return"],
            "risk_controls": template["risk_controls"],
            "description": template["description"]
        }
    
    def _generate_strategy_rationale(self, risk_score: int, risk_level: str, questionnaire_data: Dict[str, Any]) -> str:
        """Generate explanation of why this strategy is appropriate using template data"""
        template = INVESTMENT_PROFILE_TEMPLATES.get(risk_score, INVESTMENT_PROFILE_TEMPLATES[3])
        time_horizon = questionnaire_data.get('time_horizon', '')
        goal = questionnaire_data.get('investment_goal', '')
        
        rationale = f"Based on your {risk_level.lower()} risk profile (score {risk_score}/5), we recommend the '{template['name']}' strategy. "
        rationale += f"{template['description']} "
        
        rationale += f"This {template['core_strategy']} approach is designed to "
        
        if 'preservation' in goal.lower():
            rationale += "prioritize capital preservation while providing inflation-adjusted returns."
        elif 'income' in goal.lower():
            rationale += "generate regular income while maintaining growth potential."
        elif 'accumulation' in goal.lower():
            rationale += "build long-term wealth through strategic asset allocation."
        else:
            rationale += "balance growth potential with appropriate risk management."
        
        # Add time horizon context
        if '10+' in time_horizon:
            rationale += " Your long-term time horizon allows for potential market volatility recovery and compound growth."
        elif '5-10' in time_horizon:
            rationale += " Your medium to long-term horizon provides flexibility for market cycles."
        else:
            rationale += " Your shorter time horizon emphasizes the importance of risk management and liquidity."
        
        # Add strategy-specific insights
        rationale += f" Expected volatility: {template['volatility_expectation']} with anticipated returns of {template['expected_return']}."
        
        return rationale
    
    def _mock_questionnaire_analysis(self, questionnaire_data: Dict[str, Any], risk_score: int, risk_level: str) -> str:
        """Mock detailed questionnaire analysis using investment profile templates"""
        template = INVESTMENT_PROFILE_TEMPLATES.get(risk_score, INVESTMENT_PROFILE_TEMPLATES[3])
        
        return f"""Based on your questionnaire responses, you have been assigned a {risk_level.lower()} risk profile with a score of {risk_score}/5.

INVESTMENT PROFILE: {template['name']}
Strategy: {template['core_strategy']}

Key factors in this assessment:
- Investment time horizon: {questionnaire_data.get('time_horizon', 'Not specified')}
- Risk tolerance for market volatility: {questionnaire_data.get('risk_tolerance', 'Not specified')}
- Investment experience level: {questionnaire_data.get('experience_level', 'Not specified')}
- Primary investment goal: {questionnaire_data.get('investment_goal', 'Not specified')}

RECOMMENDED PORTFOLIO ALLOCATION:
{template['asset_allocation']}

STRATEGY DETAILS:
- Focus: {template['investment_focus']}
- Expected Volatility: {template['volatility_expectation']}
- Expected Returns: {template['expected_return']}
- Time Horizon Fit: {template['time_horizon_fit']}

RISK CONTROLS:
{', '.join(f"{k}: {v}" for k, v in template['risk_controls'].items())}

This {template['name']} profile is designed to {template['description'].lower()}. The recommended allocation balances your stated objectives with appropriate risk management for your risk tolerance level.

Please note: This analysis is for educational purposes only and should not be considered as personalized financial advice. Consider consulting with a qualified financial advisor for comprehensive investment planning."""

# ============================================================================
# UNIFIED STRATEGY ORCHESTRATOR
# ============================================================================

class UnifiedStrategyOrchestrator:
    """Enhanced orchestrator that integrates questionnaire results with market data and AI analysis"""
    
    def __init__(self, strategy_service, market_service, ai_service):
        self.strategy_service = strategy_service
        self.market_service = market_service
        self.ai_service = ai_service
    
    async def create_investment_strategy(self, request: UnifiedStrategyRequest) -> Dict[str, Any]:
        """Create comprehensive investment strategy integrating questionnaire results with market data"""
        try:
            logger.info(f"Creating enhanced strategy for risk score {request.risk_score} with ${request.investment_amount:,.2f}")
            
            # Step 1: Fetch investment profile template based on risk score
            logger.info("Fetching investment profile template...")
            investment_profile = self._get_investment_profile(request.risk_score)
            theoretical_allocations = investment_profile["base_allocation"]
            
            # Step 2: Get AI recommendations for actual stock symbols
            logger.info("Getting AI stock recommendations...")
            stock_recommendations = await self._get_ai_stock_recommendations(
                investment_profile, request.sector_preferences, request.investment_restrictions
            )
            
            # Step 3: Fetch real market data for recommended stocks
            logger.info("Fetching real market data...")
            market_data = await self._fetch_market_data_for_recommendations(stock_recommendations)
            
            # Step 4: Calculate confidence score based on risk level
            confidence_score = self._calculate_confidence_score(request.risk_score, market_data)
            
            # Step 5: Re-evaluate strategy with AI using actual market values
            logger.info("Re-evaluating strategy with AI...")
            final_strategy_analysis = await self._ai_strategy_evaluation(
                investment_profile, stock_recommendations, market_data, request, confidence_score
            )
            
            # Step 6: Generate specific investment allocations
            logger.info("Generating investment allocations...")
            investment_allocations = self._generate_investment_allocations(
                stock_recommendations, market_data, request.investment_amount, theoretical_allocations
            )
            
            # Step 7: Calculate re-evaluation date
            next_review_date = self._calculate_reevaluation_date(request.risk_score, market_data)
            
            # Step 8: Create comprehensive strategy response
            strategy = {
                "strategy_id": f"enhanced_strategy_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "created_at": datetime.now().isoformat(),
                "strategy_type": "AI-Enhanced Questionnaire-Based Strategy",
                
                # Risk Profile Information
                "risk_profile": {
                    "risk_score": request.risk_score,
                    "risk_level": request.risk_level,
                    "profile_name": investment_profile["name"],
                    "core_strategy": investment_profile["core_strategy"],
                    "confidence_score": confidence_score
                },
                
                # Investment Allocations
                "investment_allocations": investment_allocations,
                
                # Theoretical vs Actual
                "allocation_comparison": {
                    "theoretical_allocations": theoretical_allocations,
                    "recommended_stocks": stock_recommendations,
                    "current_market_prices": {symbol: data.get("price", 0) for symbol, data in market_data.items()}
                },
                
                # AI Strategy Analysis
                "ai_strategy_analysis": final_strategy_analysis,
                
                # Market Context
                "market_context": {
                    "market_data": market_data,
                    "market_sentiment": await self.market_service.get_market_sentiment()
                },
                
                # Investment Guidelines
                "investment_guidelines": {
                    "time_horizon": request.time_horizon,
                    "experience_level": request.experience_level,
                    "liquidity_needs": request.liquidity_needs,
                    "sector_preferences": request.sector_preferences,
                    "investment_restrictions": request.investment_restrictions
                },
                
                # Future Planning
                "next_review_date": next_review_date,
                "review_triggers": self._get_review_triggers(request.risk_score),
                
                # Performance Expectations
                "performance_expectations": {
                    "expected_return": investment_profile["expected_return"],
                    "volatility_expectation": investment_profile["volatility_expectation"],
                    "time_horizon_fit": investment_profile["time_horizon_fit"]
                }
            }
            
            # Save enhanced strategy to database
            if supabase:
                try:
                    supabase.table("enhanced_strategies").insert({
                        "strategy_id": strategy["strategy_id"],
                        "created_at": strategy["created_at"],
                        "risk_score": request.risk_score,
                        "investment_amount": request.investment_amount,
                        "confidence_score": confidence_score,
                        "investment_allocations": investment_allocations,
                        "next_review_date": next_review_date
                    }).execute()
                except Exception as e:
                    logger.error(f"Failed to save enhanced strategy: {e}")
            
            return strategy
            
        except Exception as e:
            logger.error(f"Enhanced unified strategy creation error: {e}")
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to create enhanced investment strategy: {str(e)}"
            )
    
    def _get_investment_profile(self, risk_score: int) -> Dict[str, Any]:
        """Fetch investment profile template based on risk score"""
        return INVESTMENT_PROFILE_TEMPLATES.get(risk_score, INVESTMENT_PROFILE_TEMPLATES[3])
    
    async def _get_ai_stock_recommendations(self, investment_profile: Dict[str, Any], 
                                          sector_preferences: List[str], 
                                          investment_restrictions: List[str]) -> Dict[str, List[str]]:
        """Get AI recommendations for actual stock symbols based on profile and preferences"""
        
        profile_name = investment_profile["name"]
        base_allocation = investment_profile["base_allocation"]
        core_strategy = investment_profile["core_strategy"]
        
        # Build AI query for stock recommendations
        ai_query = f"""
        Based on the investment profile '{profile_name}' with strategy '{core_strategy}', 
        recommend specific ETF symbols for each allocation category:
        
        Base allocation template: {base_allocation}
        Sector preferences: {sector_preferences if sector_preferences else 'None specified'}
        Investment restrictions: {investment_restrictions if investment_restrictions else 'None'}
        
        For each allocation category, provide 1-2 specific ETF symbols that match the strategy.
        Focus on liquid, well-established ETFs with low expense ratios.
        Consider sector preferences and avoid restricted investments.
        
        Return specific ticker symbols that can be traded in the market.
        """
        
        try:
            if self.ai_service.openai_key:
                ai_response = await self.ai_service._call_openai(ai_query, "stock_recommendation")
            else:
                ai_response = self._mock_ai_stock_recommendations(investment_profile, sector_preferences)
            
            # Parse AI response to extract stock symbols
            recommendations = self._parse_stock_recommendations(ai_response, base_allocation)
            
        except Exception as e:
            logger.error(f"AI stock recommendation error: {e}")
            recommendations = self._fallback_stock_recommendations(investment_profile)
        
        return recommendations
    
    def _parse_stock_recommendations(self, ai_response: str, base_allocation: Dict[str, float]) -> Dict[str, List[str]]:
        """Parse AI response to extract stock recommendations"""
        recommendations = {}
        
        # For mock/simple parsing, use predefined mappings
        category_mappings = {
            "SGOV": ["SGOV", "BIL"],  # Short-term Treasury
            "VPU": ["VPU", "XLU"],   # Utilities
            "TIPS": ["SCHP", "VTEB"], # Inflation-protected
            "VTI": ["VTI", "ITOT"],  # Total market
            "BNDX": ["BNDX", "IAGG"], # International bonds
            "GSG": ["GSG", "DJP"],   # Commodities
            "VTV": ["VTV", "IWD"],   # Value
            "MTUM": ["MTUM", "PDP"], # Momentum
            "VMOT": ["VMOT", "BIL"], # Short-term
            "QQQ": ["QQQ", "ONEQ"],  # NASDAQ
            "VUG": ["VUG", "IWF"],   # Growth
            "ARKK": ["ARKK", "QTEC"], # Innovation
            "BND": ["BND", "AGG"],   # Total bond market
            "TQQQ": ["TQQQ", "TECL"], # 3x Tech
            "SOXL": ["SOXL", "USD"], # 3x Semiconductors
            "SPXL": ["SPXL", "UPRO"]  # 3x S&P
        }
        
        for category, allocation in base_allocation.items():
            if category in category_mappings:
                recommendations[category] = category_mappings[category]
            else:
                # Fallback mapping
                recommendations[category] = [category]
        
        return recommendations
    
    def _mock_ai_stock_recommendations(self, investment_profile: Dict[str, Any], sector_preferences: List[str]) -> str:
        """Mock AI stock recommendations for testing"""
        profile_name = investment_profile["name"]
        return f"""For {profile_name}, I recommend the following ETFs:
        
        - Treasury/Bond allocation: SGOV, BND, BNDX for stability
        - Equity allocation: VTI, QQQ, VUG for growth
        - Sector allocation: Based on preferences {sector_preferences}
        - Alternative allocation: GSG, ARKK for diversification
        
        These selections provide optimal risk-return characteristics for the specified profile."""
    
    def _fallback_stock_recommendations(self, investment_profile: Dict[str, Any]) -> Dict[str, List[str]]:
        """Fallback stock recommendations if AI fails"""
        base_allocation = investment_profile["base_allocation"]
        fallback_map = {
            "SGOV": ["SGOV"], "VPU": ["VPU"], "TIPS": ["SCHP"],
            "VTI": ["VTI"], "BNDX": ["BNDX"], "GSG": ["GSG"],
            "VTV": ["VTV"], "MTUM": ["MTUM"], "VMOT": ["BIL"],
            "QQQ": ["QQQ"], "VUG": ["VUG"], "ARKK": ["ARKK"],
            "BND": ["BND"], "TQQQ": ["TQQQ"], "SOXL": ["SOXL"], "SPXL": ["SPXL"]
        }
        
        return {category: fallback_map.get(category, [category]) for category in base_allocation.keys()}
    
    async def _fetch_market_data_for_recommendations(self, stock_recommendations: Dict[str, List[str]]) -> Dict[str, Dict[str, Any]]:
        """Fetch real market data for all recommended stocks"""
        all_symbols = []
        for symbol_list in stock_recommendations.values():
            all_symbols.extend(symbol_list)
        
        # Remove duplicates
        unique_symbols = list(set(all_symbols))
        
        # Fetch market data
        market_data = await self.market_service.get_quotes(unique_symbols)
        
        return market_data
    
    def _calculate_confidence_score(self, risk_score: int, market_data: Dict[str, Dict[str, Any]]) -> float:
        """Calculate confidence score based on strategy quality and market conditions"""
        # Minimum acceptable confidence thresholds for each risk profile
        min_confidence_thresholds = {
            1: 0.85,  # Ultra conservative minimum threshold
            2: 0.75,  # Conservative minimum threshold  
            3: 0.65,  # Moderate minimum threshold
            4: 0.55,  # Aggressive minimum threshold
            5: 0.45   # Maximum risk minimum threshold
        }
        
        # Start with a base confidence that can go high for any profile
        base_confidence = 0.80  # Good starting point for all profiles
        
        # Adjust based on market conditions and strategy quality
        market_adjustment = 0.0
        
        # 1. Data Quality Assessment (up to +15% boost)
        valid_prices = sum(1 for data in market_data.values() if data.get("price", 0) > 0)
        total_symbols = len(market_data)
        
        if total_symbols > 0:
            data_quality = valid_prices / total_symbols
            # High data quality boosts confidence significantly
            market_adjustment += (data_quality - 0.5) * 0.15  # +/- 7.5% based on data quality
        
        # 2. Market Sentiment Analysis (up to +10% boost)
        try:
            positive_trends = sum(1 for data in market_data.values() 
                                if data.get("technical_analysis", {}).get("trend") == "BULLISH")
            if total_symbols > 0:
                trend_ratio = positive_trends / total_symbols
                # Favorable trends boost confidence
                market_adjustment += (trend_ratio - 0.5) * 0.10  # +/- 5% based on trends
        except:
            pass
        
        # 3. Strategy-Market Alignment Bonus (up to +10% boost)
        # High-risk strategies get bonus in bull markets, conservative strategies get bonus in uncertain times
        try:
            if total_symbols > 0:
                bullish_ratio = sum(1 for data in market_data.values() 
                                  if data.get("technical_analysis", {}).get("trend") == "BULLISH") / total_symbols
                
                if risk_score >= 4 and bullish_ratio > 0.6:  # Aggressive strategies in bull market
                    market_adjustment += 0.08
                elif risk_score <= 2 and bullish_ratio < 0.4:  # Conservative strategies in uncertain market
                    market_adjustment += 0.06
        except:
            pass
        
        # Calculate final confidence
        calculated_confidence = base_confidence + market_adjustment
        
        # Ensure minimum threshold is met, but allow high scores
        min_threshold = min_confidence_thresholds.get(risk_score, 0.65)
        final_confidence = max(min_threshold, min(0.99, calculated_confidence))
        
        return round(final_confidence, 2)
    
    async def _ai_strategy_evaluation(self, investment_profile: Dict[str, Any], 
                                    stock_recommendations: Dict[str, List[str]],
                                    market_data: Dict[str, Dict[str, Any]],
                                    request: UnifiedStrategyRequest,
                                    confidence_score: float) -> Dict[str, Any]:
        """Re-evaluate strategy with AI using actual market values"""
        
        # Prepare market context
        market_summary = {}
        for category, symbols in stock_recommendations.items():
            for symbol in symbols:
                if symbol in market_data:
                    data = market_data[symbol]
                    market_summary[symbol] = {
                        "price": data.get("price", 0),
                        "change_percent": data.get("change_percent", "0%"),
                        "trend": data.get("technical_analysis", {}).get("trend", "NEUTRAL")
                    }
        
        ai_query = f"""
        Evaluate this investment strategy with current market conditions:
        
        INVESTMENT PROFILE: {investment_profile['name']}
        Risk Score: {request.risk_score}/5 
        Investment Amount: ${request.investment_amount:,.2f}
        Time Horizon: {request.time_horizon}
        
        THEORETICAL ALLOCATION: {investment_profile['base_allocation']}
        RECOMMENDED STOCKS: {stock_recommendations}
        
        CURRENT MARKET DATA: {market_summary}
        
        INVESTMENT RESTRICTIONS: {request.investment_restrictions}
        SECTOR PREFERENCES: {request.sector_preferences}
        
        Current AI Confidence Score: {confidence_score}
        
        Provide a comprehensive analysis including:
        1. Strategy assessment given current market conditions
        2. Any micro-adjustments needed to the allocation
        3. Market timing considerations
        4. Risk assessment with current prices
        5. Final recommendation with rationale
        
        Focus on accuracy and actionable insights.
        """
        
        try:
            if self.ai_service.openai_key:
                analysis = await self.ai_service._call_openai(ai_query, "strategy_evaluation")
            else:
                analysis = self._mock_strategy_evaluation(investment_profile, market_summary, confidence_score)
        except Exception as e:
            logger.error(f"AI strategy evaluation error: {e}")
            analysis = f"Strategy evaluation completed with {confidence_score:.0%} confidence. Market conditions appear suitable for the selected investment profile."
        
        return {
            "detailed_analysis": analysis,
            "confidence_score": confidence_score,
            "market_assessment": "FAVORABLE" if confidence_score > 0.7 else "CAUTIOUS" if confidence_score > 0.6 else "CONSERVATIVE",
            "recommendation": "PROCEED" if confidence_score > 0.65 else "PROCEED_WITH_CAUTION",
            "key_insights": self._extract_key_insights(market_data, investment_profile)
        }
    
    def _mock_strategy_evaluation(self, investment_profile: Dict[str, Any], market_summary: Dict[str, Any], confidence_score: float) -> str:
        """Mock strategy evaluation for testing"""
        profile_name = investment_profile["name"]
        
        return f"""Strategy Evaluation for {profile_name}:
        
        Current market conditions are {'favorable' if confidence_score > 0.7 else 'mixed'} for this investment profile.
        
        Key observations:
        - Portfolio alignment with risk tolerance: GOOD
        - Current market valuations: {'REASONABLE' if confidence_score > 0.7 else 'ELEVATED'}
        - Recommended allocation appears well-balanced
        
        Micro-adjustments suggested:
        - Maintain core allocation percentages
        - Consider dollar-cost averaging for large positions
        - Monitor market conditions for entry timing
        
        Overall Assessment: Strategy is well-suited for the specified risk profile with {confidence_score:.0%} confidence.
        Recommend proceeding with gradual implementation."""
    
    def _extract_key_insights(self, market_data: Dict[str, Dict[str, Any]], investment_profile: Dict[str, Any]) -> List[str]:
        """Extract key insights from market data and profile"""
        insights = []
        
        # Analyze price trends
        bullish_count = sum(1 for data in market_data.values() 
                           if data.get("technical_analysis", {}).get("trend") == "BULLISH")
        total_count = len(market_data)
        
        if total_count > 0:
            bullish_ratio = bullish_count / total_count
            if bullish_ratio > 0.6:
                insights.append("Market sentiment is generally positive across recommended securities")
            elif bullish_ratio < 0.4:
                insights.append("Market showing some caution - consider phased entry approach")
            else:
                insights.append("Mixed market signals - stick to strategic allocation plan")
        
        # Profile-specific insights
        risk_level = investment_profile.get("risk_level", "Moderate")
        if risk_level == "Very Low":
            insights.append("Conservative approach aligns well with current market uncertainty")
        elif risk_level == "Very High":
            insights.append("High-risk strategy suitable for investors with long time horizons")
        
        insights.append(f"Expected volatility: {investment_profile.get('volatility_expectation', 'Normal')}")
        
        return insights
    
    def _generate_investment_allocations(self, stock_recommendations: Dict[str, List[str]], 
                                       market_data: Dict[str, Dict[str, Any]], 
                                       investment_amount: float, 
                                       theoretical_allocations: Dict[str, float]) -> List[Dict[str, Any]]:
        """Generate specific investment allocations with dollar amounts and quantities"""
        allocations = []
        
        for category, allocation_percent in theoretical_allocations.items():
            if category in stock_recommendations:
                recommended_symbols = stock_recommendations[category]
                
                # Choose the first available symbol with valid market data
                selected_symbol = None
                current_price = 0
                
                for symbol in recommended_symbols:
                    if symbol in market_data and market_data[symbol].get("price", 0) > 0:
                        selected_symbol = symbol
                        current_price = market_data[symbol]["price"]
                        break
                
                if selected_symbol and current_price > 0:
                    # Calculate dollar allocation
                    dollar_allocation = (allocation_percent / 100) * investment_amount
                    quantity = dollar_allocation / current_price
                    
                    allocation = {
                        "category": category,
                        "symbol": selected_symbol,
                        "allocation_percent": allocation_percent,
                        "dollar_amount": round(dollar_allocation, 2),
                        "quantity": round(quantity, 2),
                        "current_price": current_price,
                        "market_value": round(quantity * current_price, 2),
                        "alternative_symbols": [s for s in recommended_symbols if s != selected_symbol],
                        "rationale": f"{allocation_percent}% allocation to {category} via {selected_symbol}"
                    }
                    allocations.append(allocation)
                else:
                    # Fallback if no valid market data
                    dollar_allocation = (allocation_percent / 100) * investment_amount
                    allocation = {
                        "category": category,
                        "symbol": recommended_symbols[0] if recommended_symbols else category,
                        "allocation_percent": allocation_percent,
                        "dollar_amount": round(dollar_allocation, 2),
                        "quantity": 0,
                        "current_price": 0,
                        "market_value": 0,
                        "alternative_symbols": recommended_symbols[1:] if len(recommended_symbols) > 1 else [],
                        "rationale": f"{allocation_percent}% allocation to {category} (market data unavailable)",
                        "warning": "Current market price unavailable - manual verification required"
                    }
                    allocations.append(allocation)
        
        return allocations
    
    def _calculate_reevaluation_date(self, risk_score: int, market_data: Dict[str, Dict[str, Any]]) -> str:
        """Calculate recommended re-evaluation date based on risk score and market conditions"""
        
        # Base review intervals by risk score
        base_intervals = {
            1: 90,   # Ultra conservative: Quarterly (90 days)
            2: 60,   # Conservative: Bi-monthly (60 days)  
            3: 45,   # Moderate: Every 45 days
            4: 30,   # Aggressive: Monthly (30 days)
            5: 21    # Maximum risk: Every 3 weeks (21 days)
        }
        
        base_days = base_intervals.get(risk_score, 45)
        
        # Adjust based on market conditions
        adjustment_days = 0
        
        try:
            # Check market volatility indicators
            volatile_count = 0
            total_checked = 0
            
            for symbol, data in market_data.items():
                price = data.get("price", 0)
                if price > 0:
                    total_checked += 1
                    # Simple volatility check based on change percentage
                    change_str = data.get("change_percent", "0%").replace("%", "").replace("+", "")
                    try:
                        change_abs = abs(float(change_str))
                        if change_abs > 3:  # More than 3% daily change indicates volatility
                            volatile_count += 1
                    except:
                        pass
            
            if total_checked > 0:
                volatility_ratio = volatile_count / total_checked
                if volatility_ratio > 0.5:  # High volatility
                    adjustment_days = -7  # Review sooner
                elif volatility_ratio < 0.1:  # Low volatility
                    adjustment_days = +7  # Can wait longer
        
        except Exception as e:
            logger.error(f"Error calculating market volatility adjustment: {e}")
        
        # Calculate final review date
        final_days = max(14, base_days + adjustment_days)  # Minimum 2 weeks
        review_date = datetime.now() + timedelta(days=final_days)
        
        return review_date.isoformat()
    
    def _get_review_triggers(self, risk_score: int) -> List[str]:
        """Get conditions that should trigger early portfolio review"""
        base_triggers = [
            "Market volatility exceeds 20% for any position",
            "Individual position gains/losses exceed 15%",
            "Major economic events or policy changes",
            "Personal financial situation changes significantly"
        ]
        
        # Add risk-specific triggers
        if risk_score <= 2:  # Conservative profiles
            base_triggers.extend([
                "Any position loses more than 5% in value",
                "Interest rate changes affecting bond positions",
                "Inflation rate changes significantly"
            ])
        elif risk_score >= 4:  # Aggressive profiles
            base_triggers.extend([
                "Portfolio gains exceed 25% (consider profit taking)",
                "Sector concentration exceeds planned limits",
                "Leveraged positions show significant moves"
            ])
        
        return base_triggers

# ============================================================================
# INITIALIZE SERVICES
# ============================================================================

strategy_service = StraightArrowStrategy()
market_service = MarketDataService()
ai_service = AIAgentService()
orchestrator_service = UnifiedStrategyOrchestrator(strategy_service, market_service, ai_service)

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
            "3": "POST /api/analyze-questionnaire - Analyze risk questionnaire",
            "4": "POST /api/market-data - Get market data",
            "5": "POST /api/analyze-portfolio - Analyze portfolio",
            "6": "POST /api/agents/* - Use AI agents"
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
            "market_data": "finnhub" if FINNHUB_API_KEY else "mock",
            "ai_service": "openai" if OPENAI_API_KEY else "mock",
            "agents": ["data_analyst", "risk_analyst", "trading_analyst", "questionnaire_analyst"],
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

@app.post("/api/ticker", response_model=TickerResponse)
async def get_ticker_price(request: TickerRequest):
    """Get real-time price data for a single ticker from Finnhub API"""
    try:
        # Call Finnhub API directly for the ticker
        quote = await market_service._fetch_quote_finnhub(request.symbol.upper())
        
        if not quote:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "Failed to fetch ticker data",
                    "message": f"No data available for ticker {request.symbol}",
                    "ticker": request.symbol.upper()
                }
            )
        
        if quote.get("source") == "mock" or "error" in quote:
            error_msg = quote.get("error", "Unknown error occurred")
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "Ticker data unavailable",
                    "message": error_msg,
                    "ticker": request.symbol.upper()
                }
            )
        
        # Return the ticker data in the response model format
        return TickerResponse(
            symbol=quote["symbol"],
            price=quote["price"],
            change=quote["change"],
            change_percent=quote["change_percent"],
            high=quote["high"],
            low=quote["low"],
            previous_close=quote["previous_close"],
            timestamp=quote["timestamp"],
            source=quote["source"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ticker API error for {request.symbol}: {e}")
        raise HTTPException(
            status_code=500, 
            detail={
                "error": "Internal server error",
                "message": f"Failed to process ticker request for {request.symbol}",
                "ticker": request.symbol.upper()
            }
        )

@app.get("/api/ticker/{symbol}", response_model=TickerResponse)
async def get_ticker_price_by_path(symbol: str):
    """Get real-time price data for a single ticker via URL path (alternative endpoint)"""
    # Validate symbol length
    if len(symbol) > 10 or len(symbol) < 1:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid ticker symbol",
                "message": "Ticker symbol must be between 1-10 characters",
                "ticker": symbol.upper()
            }
        )
    
    # Create request object and call the main function
    request = TickerRequest(symbol=symbol)
    return await get_ticker_price(request)

@app.post("/api/ticker/bulk", response_model=BulkTickerResponse)
async def get_bulk_ticker_prices(request: BulkTickerRequest):
    """
    🚀 BULK TICKER API
    
    Get real-time price data for multiple tickers simultaneously from Finnhub API.
    Efficiently fetches data for up to 20 symbols concurrently.
    
    Features:
    - Concurrent data fetching for optimal performance
    - Individual symbol error handling (continues with other symbols if some fail)
    - Comprehensive response with success/error statistics
    - Real-time market data with technical analysis
    - Rate limit aware (respects Finnhub's 60 calls/minute limit)
    
    Perfect for portfolio analysis, market screening, and bulk data retrieval.
    """
    try:
        # Validate and clean symbols
        cleaned_symbols = [symbol.upper().strip() for symbol in request.symbols]
        cleaned_symbols = list(dict.fromkeys(cleaned_symbols))  # Remove duplicates while preserving order
        
        # Validate symbol format
        invalid_symbols = []
        valid_symbols = []
        for symbol in cleaned_symbols:
            if len(symbol) < 1 or len(symbol) > 10 or not symbol.isalnum():
                invalid_symbols.append(symbol)
            else:
                valid_symbols.append(symbol)
        
        if invalid_symbols:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid ticker symbols",
                    "message": f"Invalid symbols detected: {', '.join(invalid_symbols)}",
                    "invalid_symbols": invalid_symbols,
                    "requirement": "Symbols must be 1-10 alphanumeric characters"
                }
            )
        
        if not valid_symbols:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "No valid symbols",
                    "message": "No valid ticker symbols provided after validation",
                    "symbols_provided": request.symbols
                }
            )
        
        # Create concurrent tasks for fetching ticker data
        logger.info(f"Fetching bulk ticker data for {len(valid_symbols)} symbols: {', '.join(valid_symbols)}")
        
        async def fetch_single_ticker(symbol: str) -> tuple[str, dict]:
            """Fetch a single ticker with error handling"""
            try:
                quote = await market_service._fetch_quote_finnhub(symbol)
                if quote and quote.get("source") != "mock" and "error" not in quote:
                    # Create TickerResponse object
                    ticker_response = TickerResponse(
                        symbol=quote["symbol"],
                        price=quote["price"],
                        change=quote["change"],
                        change_percent=quote["change_percent"],
                        high=quote["high"],
                        low=quote["low"],
                        previous_close=quote["previous_close"],
                        timestamp=quote["timestamp"],
                        source=quote["source"]
                    )
                    return symbol, {"success": True, "data": ticker_response}
                else:
                    error_msg = quote.get("error", "Unknown error") if quote else "Failed to fetch data"
                    return symbol, {"success": False, "error": error_msg}
            except Exception as e:
                logger.error(f"Error fetching ticker {symbol}: {e}")
                return symbol, {"success": False, "error": str(e)}
        
        # Execute all requests concurrently
        tasks = [fetch_single_ticker(symbol) for symbol in valid_symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        successful_tickers = {}
        error_messages = {}
        success_count = 0
        error_count = 0
        
        for result in results:
            if isinstance(result, Exception):
                error_count += 1
                error_messages["unknown"] = str(result)
                continue
                
            symbol, result_data = result
            if result_data["success"]:
                successful_tickers[symbol] = result_data["data"]
                success_count += 1
            else:
                error_messages[symbol] = result_data["error"]
                error_count += 1
        
        # Check if we got at least some data
        if success_count == 0:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "All ticker requests failed",
                    "message": "Unable to fetch data for any of the requested symbols",
                    "requested_symbols": valid_symbols,
                    "errors": error_messages
                }
            )
        
        # Create response
        response = BulkTickerResponse(
            success_count=success_count,
            error_count=error_count,
            total_requested=len(valid_symbols),
            tickers=successful_tickers,
            errors=error_messages,
            timestamp=datetime.now().isoformat(),
            source="finnhub"
        )
        
        logger.info(f"Bulk ticker request completed: {success_count} success, {error_count} errors")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Bulk ticker API error: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "message": f"Failed to process bulk ticker request: {str(e)}",
                "symbols_requested": request.symbols
            }
        )

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

@app.post("/api/analyze-questionnaire", response_model=QuestionnaireResponse)
async def analyze_questionnaire(request: QuestionnaireRequest):
    """
    🎯 QUESTIONNAIRE ANALYSIS API
    
    Analyzes user investment questionnaire to determine:
    - Risk score (1-5)
    - Risk level (e.g., "Very Low", "Moderate", "High")
    - Portfolio strategy name (e.g., "Ultra Conservative Growth Portfolio - 1/5")
    
    The AI analyzes factors like:
    - Investment goals and time horizon
    - Risk tolerance and experience level
    - Income, net worth, and liquidity needs
    - Sector preferences and restrictions
    
    Returns comprehensive risk assessment and strategy recommendations.
    """
    try:
        # Parse the stringified questionnaire JSON
        import json
        try:
            questionnaire_data = json.loads(request.questionnaire)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in questionnaire: {e}")
            raise HTTPException(status_code=400, detail="Invalid questionnaire JSON format")
        
        # Analyze the questionnaire using AI service
        analysis = await ai_service.analyze_questionnaire(questionnaire_data)
        
        # Save analysis to database if available
        if supabase:
            try:
                supabase.table("questionnaire_analysis").insert({
                    "analysis_date": datetime.now().isoformat(),
                    "risk_score": analysis["risk_score"],
                    "risk_level": analysis["risk_level"],
                    "portfolio_strategy": analysis["portfolio_strategy_name"],
                    "questionnaire_data": questionnaire_data,
                    "confidence": analysis["confidence"]
                }).execute()
            except Exception as e:
                logger.error(f"Failed to save questionnaire analysis: {e}")
        
        return QuestionnaireResponse(
            risk_score=analysis["risk_score"],
            risk_level=analysis["risk_level"],
            portfolio_strategy_name=analysis["portfolio_strategy_name"],
            analysis_details=analysis["analysis_details"]
        )
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Questionnaire analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze questionnaire: {str(e)}")

# ============================================================================
# UNIFIED INVESTMENT STRATEGY ENDPOINT
# ============================================================================

@app.post("/api/unified-strategy")
async def create_unified_investment_strategy(request: UnifiedStrategyRequest):
    """
    🚀 UNIFIED INVESTMENT STRATEGY API
    
    Creates a comprehensive investment strategy integrating questionnaire results with market data:
    - Fetches investment profile based on risk score
    - Gets AI recommendations for actual stock symbols
    - Fetches real market data from Alpha Vantage
    - Re-evaluates strategy with AI using actual market values
    - Provides confidence score (inverse to risk score)
    - Recommends portfolio re-evaluation timeline
    
    Returns detailed investment allocations with market-based recommendations.
    """
    try:
        logger.info(f"Creating enhanced strategy for risk score {request.risk_score} with ${request.investment_amount:,.2f}")
        
        # Call the orchestrator to create comprehensive strategy
        strategy = await orchestrator_service.create_investment_strategy(request)
        
        return {
            "status": "success",
            "strategy": strategy,
            "execution_ready": True,
            "api_version": "1.6.0",
            "disclaimer": "This strategy is for educational purposes only. Not financial advice."
        }
        
    except Exception as e:
        logger.error(f"Enhanced unified strategy creation error: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to create enhanced investment strategy: {str(e)}"
        )

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
                "description": "Full Python script implementing all 9 steps - see WORKFLOW_GUIDE_UNIFIED.md for complete examples"
            }
        }
    }

@app.get("/api/contracts")
async def get_api_contracts():
    """Get comprehensive API contracts and TypeScript interfaces"""
    return {
        "title": "FinAgent API Contracts & TypeScript Interfaces",
        "description": "Complete API contracts with TypeScript interfaces for frontend integration",
        "last_updated": datetime.now().isoformat(),
        "version": "1.6.0",
        "base_url": "https://finagentcur.onrender.com",
        "contracts": {
            "unified_strategy": {
                "endpoint": "POST /api/unified-strategy",
                "description": "Main endpoint that orchestrates all services to create actionable investment strategies",
                "request_interface": """
interface UnifiedStrategyRequest {
  portfolio: { [symbol: string]: number };
  total_value: number;
  available_cash?: number;
  time_horizon?: string;
  risk_tolerance?: 'conservative' | 'moderate' | 'aggressive';
  investment_goals?: string[];
}""",
                "response_interface": """
interface UnifiedStrategyResponse {
  status: 'success' | 'error';
  strategy: {
    strategy_id: string;
    created_at: string;
    time_horizon: string;
    risk_tolerance: string;
    trade_orders: TradeOrder[];
    execution_guidelines: ExecutionGuidelines;
    risk_warnings: string[];
    performance_expectations: PerformanceExpectations;
    next_review_date: string;
  };
  execution_ready: boolean;
  api_version: string;
  disclaimer: string;
}""",
                "example_request": {
                    "portfolio": {"VTI": 50000.0, "BNDX": 30000.0, "GSG": 20000.0},
                    "total_value": 100000.0,
                    "available_cash": 10000.0,
                    "time_horizon": "3 weeks",
                    "risk_tolerance": "moderate",
                    "investment_goals": ["rebalancing", "growth"]
                }
            },
            "trade_order": {
                "description": "Individual trade order structure",
                "interface": """
interface TradeOrder {
  symbol: string;
  action: 'BUY' | 'SELL' | 'HOLD';
  order_type: 'MARKET' | 'LIMIT' | 'STOP';
  quantity: number;
  dollar_amount: number;
  current_price: number;
  target_price?: number;
  priority: 'HIGH' | 'MEDIUM' | 'LOW';
  reason: string;
  expected_impact: string;
  technical_context: string;
  timing_suggestion: string;
}"""
            },
            "execution_guidelines": {
                "description": "Trade execution guidelines",
                "interface": """
interface ExecutionGuidelines {
  execution_order: string;
  timing: string;
  market_hours: string;
  monitoring: string;
  high_priority_count: number;
  suggested_sequence: string[];
}"""
            },
            "portfolio_analysis": {
                "endpoint": "POST /api/analyze-portfolio",
                "description": "Analyze portfolio allocation and risk metrics",
                "request_interface": """
interface PortfolioRequest {
  portfolio: { [symbol: string]: number };
  total_value: number;
}""",
                "response_interface": """
interface PortfolioAnalysisResponse {
  analysis: {
    strategy: string;
    total_value: number;
    current_weights: { [symbol: string]: number };
    target_allocation: { [symbol: string]: number };
    drift_analysis: { [symbol: string]: DriftAnalysis };
    portfolio_metrics: PortfolioMetrics;
    risk_assessment: RiskAssessment;
    compliance: ComplianceStatus;
    recommendations: Recommendation[];
    timestamp: string;
  };
}"""
            },
            "market_data": {
                "endpoint": "POST /api/market-data",
                "description": "Get real-time market data with technical analysis",
                "request_interface": """
interface MarketDataRequest {
  symbols: string[];
}""",
                "response_interface": """
interface MarketDataResponse {
  quotes: {
    [symbol: string]: {
      symbol: string;
      price: number;
      change: number;
      change_percent: number;
      volume: number;
      high: number;
      low: number;
      technical_analysis: TechnicalAnalysis;
      timestamp: string;
    };
  };
  timestamp: string;
}"""
            },
            "ai_agents": {
                "endpoints": [
                    "POST /api/agents/data-analyst",
                    "POST /api/agents/risk-analyst", 
                    "POST /api/agents/trading-analyst"
                ],
                "description": "AI-powered analysis agents",
                "request_interface": """
interface AIAnalysisRequest {
  query: string;
  symbols?: string[];
  portfolio?: { [symbol: string]: number };
  total_value?: number;
  time_horizon?: string;
  analysis_type?: string;
}""",
                "response_interface": """
interface AIAnalysisResponse {
  analysis: {
    agent_type: 'data_analyst' | 'risk_analyst' | 'trading_analyst';
    analysis: string;
    confidence: number;
    key_insights: string[];
    recommendations: string[];
    market_context: any;
    timestamp: string;
  };
}"""
            }
        },
        "error_responses": {
            "description": "Standardized error response format",
            "interface": """
interface ErrorResponse {
  status: 'error';
  error: {
    code: number;
    message: string;
    details?: string;
    timestamp: string;
  };
}""",
            "common_errors": {
                "400": "Bad Request - Invalid input parameters",
                "401": "Unauthorized - Authentication required",
                "429": "Too Many Requests - Rate limit exceeded",
                "500": "Internal Server Error - Server processing error"
            }
        },
        "authentication": {
            "description": "Currently no authentication required for demo purposes",
            "note": "Production deployments should implement proper authentication"
        },
        "rate_limits": {
            "description": "API rate limiting information",
            "limits": {
                "per_minute": 60,
                "per_hour": 1000,
                "burst": 10
            }
        }
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main_enhanced_complete:app",
        host="0.0.0.0",
        port=port,
        reload=False
    ) 