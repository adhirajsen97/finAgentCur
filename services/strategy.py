"""
Investment Strategy Service - Implements the Straight Arrow strategy
"""
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from services.market_data import MarketDataService, MarketData

logger = logging.getLogger(__name__)

@dataclass
class StraightArrowStrategy:
    """Configuration for Straight Arrow investment strategy"""
    name: str = "Straight Arrow"
    description: str = "Diversified Three-Fund Style"
    
    # Base allocation from investment_strategy.py
    base_allocation: Dict[str, float] = None
    
    # Risk controls
    target_sharpe: float = 0.5
    volatility_cap: float = 0.12  # 12%
    max_drawdown: float = 0.15   # 15%
    rebalance_threshold: float = 0.05  # 5% drift
    
    # Tax optimization
    tax_loss_harvesting: bool = True
    
    def __post_init__(self):
        if self.base_allocation is None:
            self.base_allocation = {
                "VTI": 0.60,   # US Total Stock Market
                "BNDX": 0.30,  # International Bonds
                "GSG": 0.10    # Commodities
            }

@dataclass
class PortfolioPosition:
    """Individual portfolio position"""
    symbol: str
    target_weight: float
    current_weight: float
    current_value: float
    target_value: float
    drift: float
    action: str  # "BUY", "SELL", "HOLD"
    quantity_change: float = 0

@dataclass
class RebalanceRecommendation:
    """Portfolio rebalancing recommendation"""
    timestamp: datetime
    total_value: float
    positions: List[PortfolioPosition]
    rebalance_needed: bool
    estimated_cost: float
    tax_impact: float
    rationale: str

class StraightArrowStrategyService:
    """Service implementing the Straight Arrow investment strategy"""
    
    def __init__(self, market_data_service: MarketDataService):
        self.market_data = market_data_service
        self.strategy = StraightArrowStrategy()
        self.logger = logging.getLogger(__name__)
    
    async def analyze_portfolio(self, current_portfolio: Dict[str, float], 
                              portfolio_value: float) -> Dict[str, Any]:
        """Analyze current portfolio against Straight Arrow strategy"""
        try:
            # Get current market data
            symbols = list(self.strategy.base_allocation.keys())
            market_data = await self.market_data.get_portfolio_data(symbols)
            
            # Calculate portfolio metrics
            current_weights = self._calculate_current_weights(current_portfolio, portfolio_value)
            
            # Calculate drift from target allocation
            drift_analysis = self._calculate_drift(current_weights)
            
            # Get portfolio performance metrics
            portfolio_metrics = await self.market_data.calculate_portfolio_metrics(
                symbols, 
                list(self.strategy.base_allocation.values())
            )
            
            # Risk assessment
            risk_assessment = self._assess_risk(portfolio_metrics)
            
            # Generate rebalancing recommendation
            rebalance_rec = await self._generate_rebalance_recommendation(
                current_portfolio, portfolio_value, market_data
            )
            
            return {
                "strategy": asdict(self.strategy),
                "current_weights": current_weights,
                "drift_analysis": drift_analysis,
                "portfolio_metrics": portfolio_metrics,
                "risk_assessment": risk_assessment,
                "rebalance_recommendation": asdict(rebalance_rec),
                "market_data": {k: asdict(v) for k, v in market_data.items()},
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing portfolio: {e}")
            return {"error": str(e)}
    
    def _calculate_current_weights(self, portfolio: Dict[str, float], 
                                 total_value: float) -> Dict[str, float]:
        """Calculate current portfolio weights"""
        if total_value <= 0:
            return {symbol: 0 for symbol in self.strategy.base_allocation.keys()}
        
        return {symbol: value / total_value for symbol, value in portfolio.items()}
    
    def _calculate_drift(self, current_weights: Dict[str, float]) -> Dict[str, Any]:
        """Calculate drift from target allocation"""
        drift = {}
        max_drift = 0
        needs_rebalancing = False
        
        for symbol, target_weight in self.strategy.base_allocation.items():
            current_weight = current_weights.get(symbol, 0)
            symbol_drift = abs(current_weight - target_weight)
            drift[symbol] = {
                "current": current_weight,
                "target": target_weight,
                "drift": symbol_drift,
                "drift_percent": (symbol_drift / target_weight) * 100 if target_weight > 0 else 0
            }
            
            if symbol_drift > self.strategy.rebalance_threshold:
                needs_rebalancing = True
            
            max_drift = max(max_drift, symbol_drift)
        
        return {
            "positions": drift,
            "max_drift": max_drift,
            "needs_rebalancing": needs_rebalancing,
            "rebalance_threshold": self.strategy.rebalance_threshold
        }
    
    def _assess_risk(self, portfolio_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Assess portfolio risk against strategy constraints"""
        risk_assessment = {
            "overall_risk": "LOW",
            "alerts": [],
            "metrics_check": {}
        }
        
        # Check Sharpe ratio
        sharpe = portfolio_metrics.get('sharpe_ratio', 0)
        risk_assessment["metrics_check"]["sharpe_ratio"] = {
            "current": sharpe,
            "target": self.strategy.target_sharpe,
            "status": "PASS" if sharpe >= self.strategy.target_sharpe else "FAIL"
        }
        
        if sharpe < self.strategy.target_sharpe:
            risk_assessment["alerts"].append(
                f"Sharpe ratio ({sharpe:.2f}) below target ({self.strategy.target_sharpe})"
            )
            risk_assessment["overall_risk"] = "MEDIUM"
        
        # Check volatility
        volatility = portfolio_metrics.get('annual_volatility', 0)
        risk_assessment["metrics_check"]["volatility"] = {
            "current": volatility,
            "cap": self.strategy.volatility_cap,
            "status": "PASS" if volatility <= self.strategy.volatility_cap else "FAIL"
        }
        
        if volatility > self.strategy.volatility_cap:
            risk_assessment["alerts"].append(
                f"Volatility ({volatility:.1%}) exceeds cap ({self.strategy.volatility_cap:.1%})"
            )
            risk_assessment["overall_risk"] = "HIGH"
        
        # Check max drawdown
        max_drawdown = abs(portfolio_metrics.get('max_drawdown', 0))
        risk_assessment["metrics_check"]["max_drawdown"] = {
            "current": max_drawdown,
            "limit": self.strategy.max_drawdown,
            "status": "PASS" if max_drawdown <= self.strategy.max_drawdown else "FAIL"
        }
        
        if max_drawdown > self.strategy.max_drawdown:
            risk_assessment["alerts"].append(
                f"Max drawdown ({max_drawdown:.1%}) exceeds limit ({self.strategy.max_drawdown:.1%})"
            )
            risk_assessment["overall_risk"] = "HIGH"
        
        return risk_assessment
    
    async def _generate_rebalance_recommendation(self, current_portfolio: Dict[str, float],
                                               portfolio_value: float,
                                               market_data: Dict[str, MarketData]) -> RebalanceRecommendation:
        """Generate portfolio rebalancing recommendation"""
        try:
            positions = []
            total_drift = 0
            estimated_cost = 0
            
            for symbol, target_weight in self.strategy.base_allocation.items():
                current_value = current_portfolio.get(symbol, 0)
                current_weight = current_value / portfolio_value if portfolio_value > 0 else 0
                target_value = portfolio_value * target_weight
                drift = abs(current_weight - target_weight)
                
                # Determine action
                if drift > self.strategy.rebalance_threshold:
                    if current_value < target_value:
                        action = "BUY"
                        quantity_change = target_value - current_value
                    else:
                        action = "SELL"
                        quantity_change = current_value - target_value
                else:
                    action = "HOLD"
                    quantity_change = 0
                
                # Estimate transaction cost (assume 0.1% per trade)
                if action != "HOLD":
                    estimated_cost += abs(quantity_change) * 0.001
                
                positions.append(PortfolioPosition(
                    symbol=symbol,
                    target_weight=target_weight,
                    current_weight=current_weight,
                    current_value=current_value,
                    target_value=target_value,
                    drift=drift,
                    action=action,
                    quantity_change=quantity_change
                ))
                
                total_drift += drift
            
            # Determine if rebalancing is needed
            needs_rebalancing = any(pos.action != "HOLD" for pos in positions)
            
            # Generate rationale
            if needs_rebalancing:
                rationale = f"Portfolio drift of {total_drift:.1%} exceeds {self.strategy.rebalance_threshold:.1%} threshold. "
                rationale += "Rebalancing recommended to maintain target allocation and risk profile."
            else:
                rationale = "Portfolio is within acceptable drift limits. No rebalancing needed."
            
            return RebalanceRecommendation(
                timestamp=datetime.now(),
                total_value=portfolio_value,
                positions=positions,
                rebalance_needed=needs_rebalancing,
                estimated_cost=estimated_cost,
                tax_impact=0,  # Simplified - would need tax lot analysis
                rationale=rationale
            )
            
        except Exception as e:
            self.logger.error(f"Error generating rebalance recommendation: {e}")
            return RebalanceRecommendation(
                timestamp=datetime.now(),
                total_value=portfolio_value,
                positions=[],
                rebalance_needed=False,
                estimated_cost=0,
                tax_impact=0,
                rationale=f"Error in analysis: {str(e)}"
            )
    
    async def get_strategy_performance(self, period: str = "1y") -> Dict[str, Any]:
        """Get historical performance of the Straight Arrow strategy"""
        try:
            symbols = list(self.strategy.base_allocation.keys())
            weights = list(self.strategy.base_allocation.values())
            
            # Calculate portfolio metrics
            portfolio_metrics = await self.market_data.calculate_portfolio_metrics(symbols, weights)
            
            # Get individual ETF performance
            etf_performance = {}
            for symbol in symbols:
                hist_data = await self.market_data.get_historical_data(symbol, period)
                if hist_data is not None:
                    returns = hist_data['Close'].pct_change().dropna()
                    etf_performance[symbol] = {
                        "total_return": (hist_data['Close'].iloc[-1] / hist_data['Close'].iloc[0]) - 1,
                        "annual_volatility": returns.std() * (252 ** 0.5),
                        "sharpe_ratio": (returns.mean() * 252) / (returns.std() * (252 ** 0.5)) if returns.std() > 0 else 0
                    }
            
            return {
                "strategy_name": self.strategy.name,
                "period": period,
                "portfolio_metrics": portfolio_metrics,
                "etf_performance": etf_performance,
                "allocation": self.strategy.base_allocation,
                "risk_controls": {
                    "target_sharpe": self.strategy.target_sharpe,
                    "volatility_cap": self.strategy.volatility_cap,
                    "max_drawdown_limit": self.strategy.max_drawdown
                },
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting strategy performance: {e}")
            return {"error": str(e)}

# Factory function
def create_strategy_service(market_data_service: MarketDataService) -> StraightArrowStrategyService:
    """Create a strategy service instance"""
    return StraightArrowStrategyService(market_data_service)