"""
Market Data Service - Fetches live market data from multiple sources
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import aiohttp
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentaldata import FundamentalData
import os
from dataclasses import dataclass
import json
import redis
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MarketData:
    """Market data structure"""
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class FinancialMetrics:
    """Financial metrics structure"""
    symbol: str
    revenue: Optional[float] = None
    net_income: Optional[float] = None
    total_debt: Optional[float] = None
    free_cash_flow: Optional[float] = None
    roe: Optional[float] = None
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None

def cache_result(expiry_seconds: int = 300):
    """Cache decorator for market data"""
    def decorator(func):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            # Create cache key
            cache_key = f"{func.__name__}:{':'.join(map(str, args))}"
            
            try:
                if self.redis_client:
                    cached = self.redis_client.get(cache_key)
                    if cached:
                        return json.loads(cached)
            except Exception as e:
                logger.warning(f"Cache read error: {e}")
            
            # Execute function
            result = await func(self, *args, **kwargs)
            
            # Cache result
            try:
                if self.redis_client and result:
                    self.redis_client.setex(
                        cache_key, 
                        expiry_seconds, 
                        json.dumps(result, default=str)
                    )
            except Exception as e:
                logger.warning(f"Cache write error: {e}")
                
            return result
        return wrapper
    return decorator

class MarketDataService:
    """Unified market data service"""
    
    def __init__(self):
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        self.redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
        self.redis_client = None
        self._setup_redis()
        
        # Initialize Alpha Vantage clients
        if self.alpha_vantage_key:
            self.ts = TimeSeries(key=self.alpha_vantage_key, output_format='pandas')
            self.fd = FundamentalData(key=self.alpha_vantage_key, output_format='pandas')
        else:
            logger.warning("Alpha Vantage API key not found")
    
    def _setup_redis(self):
        """Setup Redis connection"""
        try:
            self.redis_client = redis.from_url(self.redis_url)
            self.redis_client.ping()
            logger.info("Redis connected successfully")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            self.redis_client = None
    
    @cache_result(expiry_seconds=60)
    async def get_real_time_quote(self, symbol: str) -> Optional[MarketData]:
        """Get real-time quote for a symbol"""
        try:
            # Try yfinance first (free and reliable)
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            if not info:
                return None
            
            # Get current price
            current_price = info.get('currentPrice') or info.get('regularMarketPrice')
            if not current_price:
                return None
            
            # Calculate change
            previous_close = info.get('previousClose', current_price)
            change = current_price - previous_close
            change_percent = (change / previous_close) * 100 if previous_close > 0 else 0
            
            return MarketData(
                symbol=symbol,
                price=current_price,
                change=change,
                change_percent=change_percent,
                volume=info.get('volume', 0),
                market_cap=info.get('marketCap'),
                pe_ratio=info.get('trailingPE'),
                dividend_yield=info.get('dividendYield')
            )
            
        except Exception as e:
            logger.error(f"Error fetching real-time quote for {symbol}: {e}")
            return None
    
    @cache_result(expiry_seconds=300)
    async def get_historical_data(self, symbol: str, period: str = "1y") -> Optional[pd.DataFrame]:
        """Get historical data for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            
            if hist.empty:
                return None
            
            return hist
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return None
    
    @cache_result(expiry_seconds=3600)
    async def get_financial_metrics(self, symbol: str) -> Optional[FinancialMetrics]:
        """Get financial metrics for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            if not info:
                return None
            
            # Get financial statements
            try:
                financials = ticker.financials
                balance_sheet = ticker.balance_sheet
                cashflow = ticker.cashflow
            except:
                financials = balance_sheet = cashflow = None
            
            return FinancialMetrics(
                symbol=symbol,
                revenue=info.get('totalRevenue'),
                net_income=info.get('netIncomeToCommon'),
                total_debt=info.get('totalDebt'),
                free_cash_flow=info.get('freeCashflow'),
                roe=info.get('returnOnEquity'),
                debt_to_equity=info.get('debtToEquity'),
                current_ratio=info.get('currentRatio'),
                quick_ratio=info.get('quickRatio')
            )
            
        except Exception as e:
            logger.error(f"Error fetching financial metrics for {symbol}: {e}")
            return None
    
    async def get_portfolio_data(self, symbols: List[str]) -> Dict[str, MarketData]:
        """Get real-time data for multiple symbols"""
        tasks = [self.get_real_time_quote(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        portfolio_data = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, MarketData):
                portfolio_data[symbol] = result
            else:
                logger.error(f"Error fetching data for {symbol}: {result}")
        
        return portfolio_data
    
    async def get_straight_arrow_portfolio(self) -> Dict[str, MarketData]:
        """Get data for Straight Arrow strategy portfolio"""
        symbols = ['VTI', 'BNDX', 'GSG']  # From investment_strategy.py
        return await self.get_portfolio_data(symbols)
    
    @cache_result(expiry_seconds=1800)
    async def get_market_sentiment(self) -> Dict[str, Any]:
        """Get market sentiment indicators"""
        try:
            # Get VIX (fear index)
            vix_data = await self.get_real_time_quote('^VIX')
            
            # Get major indices
            indices = ['SPY', 'QQQ', 'IWM']  # S&P 500, NASDAQ, Russell 2000
            indices_data = await self.get_portfolio_data(indices)
            
            return {
                'vix': vix_data.price if vix_data else None,
                'indices': indices_data,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error fetching market sentiment: {e}")
            return {}
    
    async def calculate_portfolio_metrics(self, symbols: List[str], weights: List[float]) -> Dict[str, float]:
        """Calculate portfolio-level metrics"""
        try:
            # Get historical data for all symbols
            hist_data = {}
            for symbol in symbols:
                hist = await self.get_historical_data(symbol, "1y")
                if hist is not None:
                    hist_data[symbol] = hist['Close'].pct_change().dropna()
            
            if not hist_data:
                return {}
            
            # Calculate portfolio returns
            portfolio_returns = pd.Series(0, index=list(hist_data.values())[0].index)
            for symbol, weight in zip(symbols, weights):
                if symbol in hist_data:
                    portfolio_returns += hist_data[symbol] * weight
            
            # Calculate metrics
            annual_return = portfolio_returns.mean() * 252
            annual_volatility = portfolio_returns.std() * (252 ** 0.5)
            sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
            
            # Calculate max drawdown
            cumulative_returns = (1 + portfolio_returns).cumprod()
            peak = cumulative_returns.cummax()
            drawdown = (cumulative_returns - peak) / peak
            max_drawdown = drawdown.min()
            
            return {
                'annual_return': annual_return,
                'annual_volatility': annual_volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'total_return': cumulative_returns.iloc[-1] - 1
            }
            
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {e}")
            return {}

# Initialize global service instance
market_data_service = MarketDataService()

async def get_market_data():
    """Factory function to get market data service"""
    return market_data_service 