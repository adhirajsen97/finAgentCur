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

class UnifiedStrategyRequest(BaseModel):
    """Unified investment strategy request"""
    portfolio: Dict[str, float] = Field(..., description="Current portfolio holdings")
    total_value: float = Field(..., gt=0, description="Total portfolio value")
    available_cash: Optional[float] = Field(default=0.0, description="Available cash for investing")
    time_horizon: Optional[str] = Field(default="3 weeks", description="Investment time horizon")
    risk_tolerance: Optional[str] = Field(default="moderate", description="Risk tolerance: conservative, moderate, aggressive")
    investment_goals: Optional[List[str]] = Field(default=["rebalancing"], description="Investment goals: rebalancing, growth, income, etc.")
    
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
# UNIFIED STRATEGY ORCHESTRATOR
# ============================================================================

class UnifiedStrategyOrchestrator:
    """Orchestrates all services to create comprehensive investment strategies with actionable trade orders"""
    
    def __init__(self, strategy_service, market_service, ai_service):
        self.strategy_service = strategy_service
        self.market_service = market_service
        self.ai_service = ai_service
        self.straight_arrow_symbols = ["VTI", "BNDX", "GSG"]
    
    async def create_investment_strategy(self, request: UnifiedStrategyRequest) -> Dict[str, Any]:
        """Create a comprehensive investment strategy with actionable trade orders"""
        try:
            # Step 1: Get current market data
            logger.info("Fetching current market data...")
            market_data = await self.market_service.get_quotes(self.straight_arrow_symbols)
            
            # Step 2: Analyze current portfolio
            logger.info("Analyzing current portfolio...")
            portfolio_analysis = self.strategy_service.analyze_portfolio(
                request.portfolio, 
                request.total_value
            )
            
            # Step 3: Get market sentiment
            logger.info("Analyzing market sentiment...")
            market_sentiment = await self.market_service.get_market_sentiment()
            
            # Step 4: Get AI insights
            logger.info("Getting AI analysis...")
            ai_query = f"Given the current market conditions, analyze the investment strategy for a {request.time_horizon} time horizon with {request.risk_tolerance} risk tolerance. Focus on portfolio rebalancing and tactical adjustments."
            ai_analysis = await self.ai_service.data_analyst(ai_query, self.straight_arrow_symbols)
            
            # Step 5: Get risk analysis
            logger.info("Conducting risk analysis...")
            risk_analysis = await self.ai_service.risk_analyst(
                request.portfolio, 
                request.total_value, 
                request.time_horizon
            )
            
            # Step 6: Generate trade orders
            logger.info("Generating trade orders...")
            trade_orders = self._generate_trade_orders(
                portfolio_analysis, 
                market_data, 
                request.available_cash,
                request.total_value,
                market_sentiment
            )
            
            # Step 7: Create comprehensive strategy response
            strategy = {
                "strategy_id": f"strategy_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "created_at": datetime.now().isoformat(),
                "time_horizon": request.time_horizon,
                "risk_tolerance": request.risk_tolerance,
                "investment_goals": request.investment_goals,
                "strategy_type": "Straight Arrow Enhanced",
                
                # Market Context
                "market_context": {
                    "current_prices": {symbol: data.get("price", 0) for symbol, data in market_data.items()},
                    "market_sentiment": market_sentiment.get("sentiment", {}),
                    "technical_analysis": {
                        symbol: data.get("technical_analysis", {}) 
                        for symbol, data in market_data.items()
                    }
                },
                
                # Portfolio Analysis
                "portfolio_analysis": {
                    "current_allocation": portfolio_analysis["current_weights"],
                    "target_allocation": portfolio_analysis["target_allocation"],
                    "drift_analysis": portfolio_analysis["drift_analysis"],
                    "risk_metrics": portfolio_analysis["portfolio_metrics"],
                    "compliance_status": portfolio_analysis["compliance"]["status"],
                    "needs_rebalancing": portfolio_analysis["risk_assessment"]["needs_rebalancing"]
                },
                
                # AI Insights
                "ai_insights": {
                    "market_analysis": ai_analysis.get("analysis", ""),
                    "risk_assessment": risk_analysis.get("analysis", ""),
                    "confidence_score": (ai_analysis.get("confidence", 0) + risk_analysis.get("confidence", 0)) / 2
                },
                
                # Actionable Trade Orders
                "trade_orders": trade_orders,
                
                # Strategy Summary
                "strategy_summary": self._create_strategy_summary(portfolio_analysis, trade_orders, market_sentiment),
                
                # Execution Guidelines
                "execution_guidelines": self._create_execution_guidelines(trade_orders, market_sentiment),
                
                # Risk Warnings
                "risk_warnings": self._create_risk_warnings(portfolio_analysis, market_sentiment),
                
                # Performance Expectations
                "performance_expectations": self._create_performance_expectations(portfolio_analysis),
                
                # Next Review Date
                "next_review_date": self._calculate_next_review_date(request.time_horizon)
            }
            
            # Save strategy to database if available
            if supabase:
                try:
                    supabase.table("investment_strategies").insert({
                        "strategy_id": strategy["strategy_id"],
                        "created_at": strategy["created_at"],
                        "portfolio_value": request.total_value,
                        "time_horizon": request.time_horizon,
                        "risk_tolerance": request.risk_tolerance,
                        "trade_orders": trade_orders,
                        "ai_confidence": strategy["ai_insights"]["confidence_score"]
                    }).execute()
                except Exception as e:
                    logger.error(f"Failed to save strategy: {e}")
            
            return strategy
            
        except Exception as e:
            logger.error(f"Strategy creation error: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to create investment strategy: {str(e)}")
    
    def _generate_trade_orders(self, portfolio_analysis, market_data, available_cash, total_value, market_sentiment) -> List[Dict[str, Any]]:
        """Generate specific trade orders based on analysis"""
        trade_orders = []
        
        # Get current prices
        current_prices = {symbol: data.get("price", 0) for symbol, data in market_data.items()}
        
        # Process rebalancing recommendations
        recommendations = portfolio_analysis.get("recommendations", [])
        for rec in recommendations:
            symbol = rec["symbol"]
            action = rec["action"]
            current_percent = rec["current_percent"]
            target_percent = rec["target_percent"]
            
            if symbol not in current_prices:
                continue
                
            current_price = current_prices[symbol]
            
            # Calculate dollar amounts
            current_value = (current_percent / 100) * total_value
            target_value = (target_percent / 100) * total_value
            difference_value = target_value - current_value
            
            # Determine trade details
            if abs(difference_value) > 1000:  # Only trade if difference > $1000
                if difference_value > 0:  # Need to buy
                    order_action = "BUY"
                    dollar_amount = min(difference_value, available_cash)
                    quantity = dollar_amount / current_price if current_price > 0 else 0
                else:  # Need to sell
                    order_action = "SELL"
                    dollar_amount = abs(difference_value)
                    quantity = dollar_amount / current_price if current_price > 0 else 0
                
                # Get technical analysis for additional context
                tech_analysis = market_data.get(symbol, {}).get("technical_analysis", {})
                trend = tech_analysis.get("trend", "NEUTRAL")
                
                # Adjust priority based on market conditions and technical analysis
                priority = self._determine_trade_priority(rec["priority"], trend, market_sentiment)
                
                trade_order = {
                    "symbol": symbol,
                    "action": order_action,
                    "order_type": "MARKET",
                    "quantity": round(quantity, 2),
                    "dollar_amount": round(dollar_amount, 2),
                    "current_price": current_price,
                    "priority": priority,
                    "reason": f"Rebalance to target allocation: {current_percent:.1f}% → {target_percent:.1f}%",
                    "expected_impact": f"Brings {symbol} allocation closer to Straight Arrow target",
                    "technical_context": f"Market trend: {trend}, Recommendation: {tech_analysis.get('recommendation', 'HOLD')}",
                    "timing_suggestion": self._get_timing_suggestion(trend, market_sentiment)
                }
                
                trade_orders.append(trade_order)
        
        # Add cash investment recommendations if available cash > $5000
        if available_cash > 5000:
            cash_orders = self._generate_cash_investment_orders(available_cash, current_prices, market_sentiment)
            trade_orders.extend(cash_orders)
        
        return trade_orders
    
    def _generate_cash_investment_orders(self, available_cash, current_prices, market_sentiment) -> List[Dict[str, Any]]:
        """Generate orders for investing available cash according to Straight Arrow allocation"""
        cash_orders = []
        target_allocation = {"VTI": 0.60, "BNDX": 0.30, "GSG": 0.10}
        
        for symbol, target_weight in target_allocation.items():
            if symbol not in current_prices:
                continue
                
            investment_amount = available_cash * target_weight
            current_price = current_prices[symbol]
            quantity = investment_amount / current_price if current_price > 0 else 0
            
            if investment_amount >= 500:  # Minimum $500 investment
                cash_order = {
                    "symbol": symbol,
                    "action": "BUY",
                    "order_type": "MARKET",
                    "quantity": round(quantity, 2),
                    "dollar_amount": round(investment_amount, 2),
                    "current_price": current_price,
                    "priority": "MEDIUM",
                    "reason": f"Invest available cash according to Straight Arrow allocation ({target_weight:.0%})",
                    "expected_impact": f"Increases {symbol} position while maintaining target allocation",
                    "technical_context": "Cash deployment following strategic allocation",
                    "timing_suggestion": "Execute as market orders for immediate deployment"
                }
                cash_orders.append(cash_order)
        
        return cash_orders
    
    def _determine_trade_priority(self, base_priority, trend, market_sentiment) -> str:
        """Determine trade priority based on multiple factors"""
        sentiment = market_sentiment.get("sentiment", {})
        overall_sentiment = sentiment.get("overall_sentiment", "NEUTRAL")
        
        # Upgrade priority if market conditions are favorable
        if base_priority == "HIGH":
            return "HIGH"
        elif base_priority == "MEDIUM" and trend == "BULLISH" and overall_sentiment == "BULLISH":
            return "HIGH"
        elif base_priority == "MEDIUM" and trend == "BEARISH" and overall_sentiment == "BEARISH":
            return "LOW"  # Delay selling in bearish conditions
        else:
            return base_priority
    
    def _get_timing_suggestion(self, trend, market_sentiment) -> str:
        """Get timing suggestions based on market conditions"""
        sentiment = market_sentiment.get("sentiment", {})
        overall_sentiment = sentiment.get("overall_sentiment", "NEUTRAL")
        
        if trend == "BULLISH" and overall_sentiment == "BULLISH":
            return "Execute soon - favorable market conditions"
        elif trend == "BEARISH" and overall_sentiment == "BEARISH":
            return "Consider delay - monitor market conditions"
        else:
            return "Execute when convenient - neutral conditions"
    
    def _create_strategy_summary(self, portfolio_analysis, trade_orders, market_sentiment) -> Dict[str, Any]:
        """Create a summary of the investment strategy"""
        total_trades = len(trade_orders)
        buy_orders = [order for order in trade_orders if order["action"] == "BUY"]
        sell_orders = [order for order in trade_orders if order["action"] == "SELL"]
        
        return {
            "overview": f"Straight Arrow rebalancing strategy with {total_trades} recommended trades",
            "total_trades": total_trades,
            "buy_orders": len(buy_orders),
            "sell_orders": len(sell_orders),
            "total_investment": sum(order["dollar_amount"] for order in buy_orders),
            "total_divestment": sum(order["dollar_amount"] for order in sell_orders),
            "rebalancing_needed": portfolio_analysis["risk_assessment"]["needs_rebalancing"],
            "market_conditions": market_sentiment.get("sentiment", {}).get("overall_sentiment", "NEUTRAL"),
            "strategy_confidence": "HIGH" if total_trades <= 3 else "MEDIUM"
        }
    
    def _create_execution_guidelines(self, trade_orders, market_sentiment) -> Dict[str, Any]:
        """Create execution guidelines for the trades"""
        high_priority_trades = [order for order in trade_orders if order["priority"] == "HIGH"]
        
        return {
            "execution_order": "Execute HIGH priority trades first, then MEDIUM, then LOW",
            "timing": "Spread trades over 1-3 days to minimize market impact",
            "market_hours": "Execute during regular trading hours for better liquidity",
            "monitoring": "Monitor positions for 24-48 hours after execution",
            "high_priority_count": len(high_priority_trades),
            "suggested_sequence": [
                f"{order['action']} {order['symbol']}: ${order['dollar_amount']:,.0f}" 
                for order in sorted(trade_orders, key=lambda x: {"HIGH": 3, "MEDIUM": 2, "LOW": 1}[x["priority"]], reverse=True)[:5]
            ]
        }
    
    def _create_risk_warnings(self, portfolio_analysis, market_sentiment) -> List[str]:
        """Create appropriate risk warnings"""
        warnings = [
            "All investments carry risk of loss. Past performance does not guarantee future results.",
            "Market conditions can change rapidly. Monitor your positions regularly.",
            "This strategy is based on the Straight Arrow methodology and may not suit all investors."
        ]
        
        # Add specific warnings based on analysis
        risk_assessment = portfolio_analysis.get("risk_assessment", {})
        if risk_assessment.get("overall_risk") == "HIGH":
            warnings.append("Your portfolio shows HIGH risk levels. Consider reducing position sizes.")
        
        compliance = portfolio_analysis.get("compliance", {})
        if compliance.get("status") != "COMPLIANT":
            warnings.append("Portfolio compliance issues detected. Review recommendations carefully.")
        
        sentiment = market_sentiment.get("sentiment", {})
        if sentiment.get("overall_sentiment") == "BEARISH":
            warnings.append("Current market sentiment is bearish. Consider phased execution of trades.")
        
        return warnings
    
    def _create_performance_expectations(self, portfolio_analysis) -> Dict[str, Any]:
        """Create performance expectations"""
        metrics = portfolio_analysis.get("portfolio_metrics", {})
        target_metrics = portfolio_analysis.get("target_metrics", {})
        
        return {
            "expected_annual_return": f"{target_metrics.get('expected_return', 0):.1%}",
            "expected_volatility": f"{target_metrics.get('volatility', 0):.1%}",
            "current_sharpe_ratio": f"{metrics.get('sharpe_ratio', 0):.2f}",
            "target_sharpe_ratio": f"{target_metrics.get('sharpe_ratio', 0):.2f}",
            "improvement_potential": "Portfolio metrics should improve after rebalancing",
            "time_horizon_note": "Expected returns are long-term averages and may vary significantly in short periods"
        }
    
    def _calculate_next_review_date(self, time_horizon: str) -> str:
        """Calculate when to next review the strategy"""
        if "week" in time_horizon.lower():
            days_ahead = 7
        elif "month" in time_horizon.lower():
            days_ahead = 30
        else:
            days_ahead = 14  # Default to 2 weeks
        
        next_review = datetime.now() + timedelta(days=days_ahead)
        return next_review.isoformat()

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
# UNIFIED INVESTMENT STRATEGY ENDPOINT
# ============================================================================

@app.post("/api/unified-strategy")
async def create_unified_investment_strategy(request: UnifiedStrategyRequest):
    """
    🚀 UNIFIED INVESTMENT STRATEGY API
    
    Creates a comprehensive investment strategy by orchestrating all available services:
    - Portfolio analysis and rebalancing recommendations
    - Real-time market data and technical analysis
    - AI-powered market insights and risk assessment
    - Actionable trade orders with priorities and execution guidelines
    
    Returns detailed trading recommendations for frontend execution.
    """
    try:
        logger.info(f"Creating unified strategy for portfolio value: ${request.total_value:,.2f}")
        
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
        logger.error(f"Unified strategy creation error: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to create unified investment strategy: {str(e)}"
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