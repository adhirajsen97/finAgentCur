"""
Unit Tests for FinAgent Services
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from services.market_data import MarketDataService, MarketData
from services.strategy import StraightArrowStrategyService, StraightArrowStrategy

class TestMarketDataService:
    """Test MarketDataService"""
    
    @pytest.fixture
    def market_service(self):
        """Create market data service instance"""
        return MarketDataService()
    
    @pytest.mark.asyncio
    async def test_get_current_price(self, market_service):
        """Test getting current price for a symbol"""
        with patch('yfinance.Ticker') as mock_ticker:
            mock_info = {
                'regularMarketPrice': 225.50,
                'regularMarketChangePercent': 0.025,
                'regularMarketVolume': 1000000,
                'marketCap': 1500000000000
            }
            mock_ticker.return_value.info = mock_info
            
            result = await market_service.get_current_price("VTI")
            
            assert result is not None
            assert result.current_price == 225.50
            assert result.change_percent == 0.025
    
    @pytest.mark.asyncio
    async def test_get_historical_data(self, market_service):
        """Test getting historical data"""
        with patch('yfinance.Ticker') as mock_ticker:
            # Create mock historical data
            dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
            mock_hist = pd.DataFrame({
                'Open': np.random.uniform(200, 250, len(dates)),
                'High': np.random.uniform(200, 250, len(dates)),
                'Low': np.random.uniform(200, 250, len(dates)),
                'Close': np.random.uniform(200, 250, len(dates)),
                'Volume': np.random.randint(1000000, 5000000, len(dates))
            }, index=dates)
            
            mock_ticker.return_value.history.return_value = mock_hist
            
            result = await market_service.get_historical_data("VTI", "1y")
            
            assert result is not None
            assert len(result) > 0
            assert 'Close' in result.columns
    
    @pytest.mark.asyncio
    async def test_calculate_portfolio_metrics(self, market_service):
        """Test portfolio metrics calculation"""
        symbols = ["VTI", "BNDX", "GSG"]
        weights = [0.6, 0.3, 0.1]
        
        with patch.object(market_service, 'get_historical_data') as mock_hist:
            # Mock historical data for each symbol
            dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
            
            def mock_hist_data(symbol, period):
                prices = np.random.uniform(100, 200, len(dates))
                return pd.DataFrame({
                    'Close': prices,
                    'Volume': np.random.randint(1000000, 5000000, len(dates))
                }, index=dates)
            
            mock_hist.side_effect = mock_hist_data
            
            result = await market_service.calculate_portfolio_metrics(symbols, weights)
            
            assert result is not None
            assert 'total_return' in result
            assert 'annual_volatility' in result
            assert 'sharpe_ratio' in result
            assert 'max_drawdown' in result

class TestStraightArrowStrategy:
    """Test StraightArrowStrategy"""
    
    @pytest.fixture
    def strategy_config(self):
        """Create strategy configuration"""
        return StraightArrowStrategy()
    
    def test_strategy_initialization(self, strategy_config):
        """Test strategy initialization"""
        assert strategy_config.name == "Straight Arrow"
        assert strategy_config.base_allocation["VTI"] == 0.60
        assert strategy_config.base_allocation["BNDX"] == 0.30
        assert strategy_config.base_allocation["GSG"] == 0.10
        assert strategy_config.target_sharpe == 0.5
        assert strategy_config.volatility_cap == 0.12

class TestStraightArrowStrategyService:
    """Test StraightArrowStrategyService"""
    
    @pytest.fixture
    def mock_market_service(self):
        """Create mock market data service"""
        mock_service = Mock(spec=MarketDataService)
        return mock_service
    
    @pytest.fixture
    def strategy_service(self, mock_market_service):
        """Create strategy service instance"""
        return StraightArrowStrategyService(mock_market_service)
    
    @pytest.mark.asyncio
    async def test_analyze_portfolio(self, strategy_service, mock_market_service):
        """Test portfolio analysis"""
        # Mock portfolio data
        current_portfolio = {
            "VTI": 22500.0,
            "BNDX": 10800.0,
            "GSG": 950.0
        }
        portfolio_value = 34250.0
        
        # Mock market data responses
        mock_market_data = {
            "VTI": MarketData(
                symbol="VTI",
                current_price=225.0,
                change_percent=0.02,
                volume=1000000,
                market_cap=1500000000000,
                timestamp=datetime.now()
            ),
            "BNDX": MarketData(
                symbol="BNDX",
                current_price=54.0,
                change_percent=-0.01,
                volume=500000,
                market_cap=50000000000,
                timestamp=datetime.now()
            ),
            "GSG": MarketData(
                symbol="GSG",
                current_price=19.0,
                change_percent=0.03,
                volume=200000,
                market_cap=2000000000,
                timestamp=datetime.now()
            )
        }
        
        mock_portfolio_metrics = {
            'total_return': 0.08,
            'annual_volatility': 0.10,
            'sharpe_ratio': 0.65,
            'max_drawdown': -0.08
        }
        
        mock_market_service.get_portfolio_data = AsyncMock(return_value=mock_market_data)
        mock_market_service.calculate_portfolio_metrics = AsyncMock(return_value=mock_portfolio_metrics)
        
        result = await strategy_service.analyze_portfolio(current_portfolio, portfolio_value)
        
        assert result is not None
        assert "strategy" in result
        assert "current_weights" in result
        assert "drift_analysis" in result
        assert "risk_assessment" in result
        assert "rebalance_recommendation" in result
    
    def test_calculate_current_weights(self, strategy_service):
        """Test current weights calculation"""
        portfolio = {"VTI": 22500.0, "BNDX": 10800.0, "GSG": 950.0}
        total_value = 34250.0
        
        weights = strategy_service._calculate_current_weights(portfolio, total_value)
        
        assert abs(weights["VTI"] - 0.657) < 0.001
        assert abs(weights["BNDX"] - 0.315) < 0.001
        assert abs(weights["GSG"] - 0.028) < 0.001
    
    def test_calculate_drift(self, strategy_service):
        """Test drift calculation"""
        current_weights = {"VTI": 0.657, "BNDX": 0.315, "GSG": 0.028}
        
        drift_analysis = strategy_service._calculate_drift(current_weights)
        
        assert "positions" in drift_analysis
        assert "max_drift" in drift_analysis
        assert "needs_rebalancing" in drift_analysis
        
        # VTI should show drift (0.657 vs target 0.60)
        vti_drift = drift_analysis["positions"]["VTI"]["drift"]
        assert abs(vti_drift - 0.057) < 0.001
    
    def test_assess_risk(self, strategy_service):
        """Test risk assessment"""
        portfolio_metrics = {
            'sharpe_ratio': 0.65,
            'annual_volatility': 0.10,
            'max_drawdown': -0.08
        }
        
        risk_assessment = strategy_service._assess_risk(portfolio_metrics)
        
        assert "overall_risk" in risk_assessment
        assert "alerts" in risk_assessment
        assert "metrics_check" in risk_assessment
        
        # Should pass all checks with these metrics
        assert risk_assessment["metrics_check"]["sharpe_ratio"]["status"] == "PASS"
        assert risk_assessment["metrics_check"]["volatility"]["status"] == "PASS"
        assert risk_assessment["metrics_check"]["max_drawdown"]["status"] == "PASS"
    
    @pytest.mark.asyncio
    async def test_generate_rebalance_recommendation(self, strategy_service):
        """Test rebalance recommendation generation"""
        current_portfolio = {"VTI": 22500.0, "BNDX": 10800.0, "GSG": 950.0}
        portfolio_value = 34250.0
        market_data = {}  # Simplified for test
        
        recommendation = await strategy_service._generate_rebalance_recommendation(
            current_portfolio, portfolio_value, market_data
        )
        
        assert recommendation is not None
        assert recommendation.total_value == portfolio_value
        assert len(recommendation.positions) == 3
        assert recommendation.timestamp is not None
        
        # Check if rebalancing is needed (VTI is overweight)
        vti_position = next(pos for pos in recommendation.positions if pos.symbol == "VTI")
        assert vti_position.action in ["BUY", "SELL", "HOLD"]
    
    @pytest.mark.asyncio
    async def test_get_strategy_performance(self, strategy_service, mock_market_service):
        """Test strategy performance retrieval"""
        mock_portfolio_metrics = {
            'total_return': 0.08,
            'annual_volatility': 0.10,
            'sharpe_ratio': 0.65,
            'max_drawdown': -0.08
        }
        
        # Mock historical data for ETF performance
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        mock_hist_data = pd.DataFrame({
            'Close': np.random.uniform(200, 250, len(dates))
        }, index=dates)
        
        mock_market_service.calculate_portfolio_metrics = AsyncMock(return_value=mock_portfolio_metrics)
        mock_market_service.get_historical_data = AsyncMock(return_value=mock_hist_data)
        
        result = await strategy_service.get_strategy_performance("1y")
        
        assert result is not None
        assert "strategy_name" in result
        assert "portfolio_metrics" in result
        assert "etf_performance" in result
        assert "allocation" in result
        assert result["strategy_name"] == "Straight Arrow"

class TestErrorHandling:
    """Test error handling in services"""
    
    @pytest.mark.asyncio
    async def test_market_data_api_failure(self):
        """Test handling of market data API failures"""
        service = MarketDataService()
        
        with patch('yfinance.Ticker') as mock_ticker:
            mock_ticker.side_effect = Exception("API Error")
            
            result = await service.get_current_price("INVALID")
            assert result is None
    
    @pytest.mark.asyncio
    async def test_strategy_analysis_error(self):
        """Test handling of strategy analysis errors"""
        mock_market_service = Mock(spec=MarketDataService)
        mock_market_service.get_portfolio_data = AsyncMock(side_effect=Exception("Market data error"))
        
        service = StraightArrowStrategyService(mock_market_service)
        
        result = await service.analyze_portfolio({}, 0)
        assert "error" in result

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 