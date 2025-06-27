"""
Portfolio Performance API - Complete P&L Tracking System
Integrates with existing FinAgent infrastructure
"""

import math
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from fastapi import HTTPException
from pydantic import BaseModel, Field
import asyncio

# ============================================================================
# PORTFOLIO PERFORMANCE MODELS
# ============================================================================

class TradeExecution(BaseModel):
    """Record of an executed trade"""
    symbol: str = Field(..., description="Trading symbol")
    action: str = Field(..., description="BUY or SELL")
    quantity: float = Field(..., gt=0, description="Number of shares traded")
    execution_price: float = Field(..., gt=0, description="Price per share at execution")
    execution_date: str = Field(..., description="Date and time of execution (ISO format)")
    total_amount: float = Field(..., description="Total trade amount (quantity * price)")
    fees: Optional[float] = Field(default=0.0, description="Trading fees and commissions")
    strategy_source: Optional[str] = Field(default="manual", description="Source strategy (e.g., unified_strategy)")

class TimeSeriesDataPoint(BaseModel):
    """Single data point for time series charts"""
    date: str = Field(..., description="Date in ISO format")
    portfolio_value: float = Field(..., description="Total portfolio value")
    total_return: float = Field(..., description="Total return amount")
    total_return_percent: float = Field(..., description="Total return percentage")
    day_change: float = Field(..., description="Day over day change")
    day_change_percent: float = Field(..., description="Day over day change percentage")

class PositionPerformance(BaseModel):
    """Performance data for individual position"""
    symbol: str = Field(..., description="Stock symbol")
    current_shares: float = Field(..., description="Current number of shares held")
    average_cost_basis: float = Field(..., description="Average cost per share")
    current_price: float = Field(..., description="Current market price per share")
    current_value: float = Field(..., description="Current market value of position")
    total_cost: float = Field(..., description="Total amount invested in this position")
    unrealized_pnl: float = Field(..., description="Unrealized profit/loss")
    unrealized_pnl_percent: float = Field(..., description="Unrealized profit/loss percentage")
    total_dividends: Optional[float] = Field(default=0.0, description="Total dividends received")
    realized_pnl: Optional[float] = Field(default=0.0, description="Realized profit/loss from sells")
    last_updated: str = Field(..., description="Last price update timestamp")
    weight_in_portfolio: float = Field(..., description="Position weight as percentage of portfolio")

class PortfolioPerformanceRequest(BaseModel):
    """Request for portfolio performance analysis"""
    trades: List[TradeExecution] = Field(..., description="List of executed trades", min_items=1)
    benchmark_symbol: Optional[str] = Field(default="VTI", description="Benchmark for comparison (default: VTI)")
    start_date: Optional[str] = Field(default=None, description="Start date for analysis (ISO format)")
    end_date: Optional[str] = Field(default=None, description="End date for analysis (ISO format)")
    include_dividends: Optional[bool] = Field(default=False, description="Include dividend tracking")

class PortfolioPerformanceResponse(BaseModel):
    """Complete portfolio performance analysis"""
    # Summary metrics
    total_invested: float = Field(..., description="Total cash invested")
    current_value: float = Field(..., description="Current portfolio value")
    total_return: float = Field(..., description="Total return (realized + unrealized)")
    total_return_percent: float = Field(..., description="Total return percentage")
    annualized_return: float = Field(..., description="Annualized return percentage")
    
    # Individual positions
    positions: List[PositionPerformance] = Field(..., description="Performance by position")
    
    # Time series data for charts
    time_series: List[TimeSeriesDataPoint] = Field(..., description="Historical performance data")
    
    # Comparison metrics
    benchmark_performance: Dict[str, Any] = Field(..., description="Benchmark comparison data")
    
    # Risk metrics
    volatility: float = Field(..., description="Portfolio volatility (annualized)")
    sharpe_ratio: float = Field(..., description="Risk-adjusted returns")
    max_drawdown: float = Field(..., description="Maximum drawdown percentage")
    
    # Additional insights
    best_performer: Optional[PositionPerformance] = Field(..., description="Best performing position")
    worst_performer: Optional[PositionPerformance] = Field(..., description="Worst performing position")
    
    # Metadata
    analysis_period: Dict[str, str] = Field(..., description="Start and end dates of analysis")
    data_quality: Dict[str, Any] = Field(..., description="Data quality and source information")
    timestamp: str = Field(..., description="Analysis timestamp")
    source: str = Field(..., description="Data source")

# ============================================================================
# PORTFOLIO PERFORMANCE SERVICE
# ============================================================================

class PortfolioPerformanceService:
    """Portfolio performance tracking and analysis service"""
    
    def __init__(self, market_service):
        self.market_service = market_service
    
    async def analyze_portfolio_performance(self, request: PortfolioPerformanceRequest) -> Dict[str, Any]:
        """Comprehensive portfolio performance analysis"""
        
        # 1. Validate and process trades
        trades = self._validate_trades(request.trades)
        
        # 2. Calculate current positions from trade history
        positions = self._calculate_current_positions(trades)
        
        # 3. Get current market prices using bulk ticker API
        symbols = list(positions.keys())
        if request.benchmark_symbol:
            symbols.append(request.benchmark_symbol)
        
        market_data = await self._fetch_current_prices(symbols)
        
        # 4. Calculate position performance
        position_performances = []
        total_invested = 0
        current_value = 0
        
        for symbol, position_data in positions.items():
            current_price = market_data.get(symbol, {}).get("price", 0)
            if current_price <= 0:
                continue
                
            perf = self._calculate_position_performance(
                symbol, position_data, current_price, market_data[symbol]
            )
            position_performances.append(perf)
            total_invested += perf.total_cost
            current_value += perf.current_value
        
        # 5. Calculate portfolio weights
        for position in position_performances:
            position.weight_in_portfolio = (position.current_value / current_value * 100) if current_value > 0 else 0
        
        # 6. Calculate portfolio-level metrics
        total_return = current_value - total_invested
        total_return_percent = (total_return / total_invested * 100) if total_invested > 0 else 0
        
        # 7. Generate time series data for charts
        time_series_data = await self._generate_time_series(
            trades, request.start_date, request.end_date, market_data
        )
        
        # 8. Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics(time_series_data)
        
        # 9. Benchmark comparison
        benchmark_performance = await self._calculate_benchmark_performance(
            request.benchmark_symbol, total_invested, request.start_date, request.end_date, market_data
        )
        
        # 10. Identify best/worst performers
        best_performer = max(position_performances, key=lambda x: x.unrealized_pnl_percent) if position_performances else None
        worst_performer = min(position_performances, key=lambda x: x.unrealized_pnl_percent) if position_performances else None
        
        return {
            "total_invested": total_invested,
            "current_value": current_value,
            "total_return": total_return,
            "total_return_percent": total_return_percent,
            "annualized_return": self._calculate_annualized_return(total_return_percent, trades),
            "positions": position_performances,
            "time_series": time_series_data,
            "benchmark_performance": benchmark_performance,
            "volatility": risk_metrics["volatility"],
            "sharpe_ratio": risk_metrics["sharpe_ratio"],
            "max_drawdown": risk_metrics["max_drawdown"],
            "best_performer": best_performer,
            "worst_performer": worst_performer,
            "analysis_period": {
                "start_date": request.start_date or trades[0].execution_date,
                "end_date": request.end_date or datetime.now().isoformat()
            },
            "data_quality": {
                "trades_analyzed": len(trades),
                "positions_current": len(position_performances),
                "market_data_source": "finnhub",
                "price_data_freshness": "real_time",
                "benchmark_available": request.benchmark_symbol in market_data
            },
            "timestamp": datetime.now().isoformat(),
            "source": "finagent_performance_analysis"
        }
    
    def _validate_trades(self, trades: List[TradeExecution]) -> List[TradeExecution]:
        """Validate and sort trades by execution date"""
        valid_trades = []
        for trade in trades:
            if trade.action.upper() not in ["BUY", "SELL"]:
                continue
            if trade.quantity <= 0 or trade.execution_price <= 0:
                continue
            valid_trades.append(trade)
        
        # Sort by execution date
        return sorted(valid_trades, key=lambda t: t.execution_date)
    
    def _calculate_current_positions(self, trades: List[TradeExecution]) -> Dict[str, Dict[str, float]]:
        """Calculate current positions from trade history"""
        positions = {}
        
        for trade in trades:
            symbol = trade.symbol.upper()
            
            if symbol not in positions:
                positions[symbol] = {"shares": 0.0, "total_cost": 0.0, "realized_pnl": 0.0}
            
            if trade.action.upper() == "BUY":
                positions[symbol]["shares"] += trade.quantity
                positions[symbol]["total_cost"] += trade.total_amount + trade.fees
            elif trade.action.upper() == "SELL":
                if positions[symbol]["shares"] >= trade.quantity:
                    # Calculate realized P&L for this sale
                    avg_cost_basis = positions[symbol]["total_cost"] / positions[symbol]["shares"] if positions[symbol]["shares"] > 0 else 0
                    sale_proceeds = trade.total_amount - trade.fees
                    cost_of_sold_shares = avg_cost_basis * trade.quantity
                    realized_gain = sale_proceeds - cost_of_sold_shares
                    
                    positions[symbol]["shares"] -= trade.quantity
                    positions[symbol]["total_cost"] -= cost_of_sold_shares
                    positions[symbol]["realized_pnl"] += realized_gain
        
        # Remove positions with zero shares
        return {symbol: data for symbol, data in positions.items() if data["shares"] > 0}
    
    async def _fetch_current_prices(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Fetch current prices using existing bulk ticker functionality"""
        try:
            # Import the bulk ticker function - this would be from main module
            from main_enhanced_complete import get_bulk_ticker_prices, BulkTickerRequest
            
            bulk_request = BulkTickerRequest(symbols=symbols)
            bulk_response = await get_bulk_ticker_prices(bulk_request)
            
            # Convert to format needed for performance calculations
            market_data = {}
            for symbol, ticker_data in bulk_response.tickers.items():
                market_data[symbol] = {
                    "price": ticker_data.price,
                    "change": ticker_data.change,
                    "change_percent": ticker_data.change_percent,
                    "timestamp": ticker_data.timestamp,
                    "source": ticker_data.source
                }
            
            return market_data
            
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Unable to fetch market data for performance analysis: {str(e)}"
            )
    
    def _calculate_position_performance(self, symbol: str, position_data: Dict, 
                                      current_price: float, market_data: Dict) -> PositionPerformance:
        """Calculate performance metrics for individual position"""
        
        current_shares = position_data["shares"]
        total_cost = position_data["total_cost"]
        average_cost_basis = total_cost / current_shares if current_shares > 0 else 0
        
        current_value = current_shares * current_price
        unrealized_pnl = current_value - total_cost
        unrealized_pnl_percent = (unrealized_pnl / total_cost * 100) if total_cost > 0 else 0
        
        return PositionPerformance(
            symbol=symbol,
            current_shares=current_shares,
            average_cost_basis=average_cost_basis,
            current_price=current_price,
            current_value=current_value,
            total_cost=total_cost,
            unrealized_pnl=unrealized_pnl,
            unrealized_pnl_percent=unrealized_pnl_percent,
            realized_pnl=position_data.get("realized_pnl", 0.0),
            last_updated=market_data.get("timestamp", datetime.now().isoformat()),
            weight_in_portfolio=0  # Will be calculated at portfolio level
        )
    
    async def _generate_time_series(self, trades: List[TradeExecution], 
                                  start_date: str, end_date: str, market_data: Dict) -> List[TimeSeriesDataPoint]:
        """Generate simplified time series data for charts"""
        # For now, create a simplified version with key milestones
        time_series = []
        
        if not trades:
            return time_series
        
        # Start with first trade date
        first_trade_date = trades[0].execution_date
        current_date = datetime.fromisoformat(first_trade_date.replace('Z', '+00:00'))
        end_dt = datetime.now()
        
        if end_date:
            try:
                end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            except:
                pass
        
        # Calculate portfolio value at key points
        portfolio_value = 0
        total_invested = 0
        
        # Add data point for each trade
        for trade in trades:
            if trade.action.upper() == "BUY":
                total_invested += trade.total_amount + trade.fees
            
            # Calculate current portfolio value using current prices
            current_positions = self._calculate_current_positions(trades[:trades.index(trade)+1])
            portfolio_value = sum(
                pos_data["shares"] * market_data.get(symbol, {}).get("price", 0)
                for symbol, pos_data in current_positions.items()
            )
            
            total_return = portfolio_value - total_invested
            total_return_percent = (total_return / total_invested * 100) if total_invested > 0 else 0
            
            time_series.append(TimeSeriesDataPoint(
                date=trade.execution_date,
                portfolio_value=portfolio_value,
                total_return=total_return,
                total_return_percent=total_return_percent,
                day_change=0,  # Would need historical data for accurate calculation
                day_change_percent=0
            ))
        
        return time_series
    
    def _calculate_risk_metrics(self, time_series: List[TimeSeriesDataPoint]) -> Dict[str, float]:
        """Calculate portfolio risk metrics"""
        if len(time_series) < 2:
            return {"volatility": 0, "sharpe_ratio": 0, "max_drawdown": 0}
        
        # Calculate returns
        returns = []
        for i in range(1, len(time_series)):
            if time_series[i-1].portfolio_value > 0:
                daily_return = (time_series[i].portfolio_value - time_series[i-1].portfolio_value) / time_series[i-1].portfolio_value
                returns.append(daily_return)
        
        if not returns:
            return {"volatility": 0, "sharpe_ratio": 0, "max_drawdown": 0}
        
        # Calculate volatility (annualized)
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return)**2 for r in returns) / len(returns)
        volatility = math.sqrt(variance) * math.sqrt(252)  # Annualized
        
        # Calculate Sharpe ratio
        risk_free_rate = 0.02  # 2% annual risk-free rate
        sharpe_ratio = ((mean_return * 252) - risk_free_rate) / volatility if volatility > 0 else 0
        
        # Calculate max drawdown
        max_drawdown = 0
        peak = time_series[0].portfolio_value
        for point in time_series:
            if point.portfolio_value > peak:
                peak = point.portfolio_value
            drawdown = (peak - point.portfolio_value) / peak if peak > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)
        
        return {
            "volatility": volatility * 100,  # Convert to percentage
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown * 100  # Convert to percentage
        }
    
    async def _calculate_benchmark_performance(self, benchmark_symbol: str, total_invested: float, 
                                             start_date: str, end_date: str, market_data: Dict) -> Dict[str, Any]:
        """Calculate benchmark performance for comparison"""
        if not benchmark_symbol or benchmark_symbol not in market_data:
            return {
                "symbol": benchmark_symbol or "N/A",
                "available": False,
                "error": "Benchmark data not available"
            }
        
        benchmark_data = market_data[benchmark_symbol]
        
        # Simplified benchmark calculation
        # In a full implementation, this would use historical data
        return {
            "symbol": benchmark_symbol,
            "available": True,
            "current_price": benchmark_data["price"],
            "change_percent": benchmark_data["change_percent"],
            "comparison": {
                "benchmark_equivalent_value": total_invested,  # Simplified
                "outperformance": "Requires historical data for accurate calculation"
            },
            "last_updated": benchmark_data["timestamp"]
        }
    
    def _calculate_annualized_return(self, total_return_percent: float, trades: List[TradeExecution]) -> float:
        """Calculate annualized return based on holding period"""
        if not trades:
            return 0
        
        # Calculate holding period in years
        first_trade = datetime.fromisoformat(trades[0].execution_date.replace('Z', '+00:00'))
        now = datetime.now()
        holding_period_years = (now - first_trade).days / 365.25
        
        if holding_period_years <= 0:
            return total_return_percent
        
        # Annualized return formula: (1 + total_return)^(1/years) - 1
        if total_return_percent == 0:
            return 0
        
        total_return_ratio = total_return_percent / 100
        annualized = ((1 + total_return_ratio) ** (1 / holding_period_years)) - 1
        return annualized * 100
