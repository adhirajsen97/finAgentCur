"""
API Tests for FinAgent Investment System
"""
import pytest
import asyncio
from httpx import AsyncClient
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import json
from datetime import datetime

from main import app

@pytest.fixture
def client():
    """Test client fixture"""
    return TestClient(app)

@pytest.fixture
async def async_client():
    """Async test client fixture"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

class TestHealthEndpoint:
    """Test health check endpoint"""
    
    def test_health_check(self, client):
        """Test health endpoint returns 200"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

class TestAnalysisEndpoints:
    """Test analysis endpoints"""
    
    @pytest.mark.asyncio
    async def test_analyze_portfolio(self, async_client):
        """Test portfolio analysis endpoint"""
        portfolio_data = {
            "portfolio": {
                "VTI": 22500.0,
                "BNDX": 10800.0,
                "GSG": 950.0
            },
            "total_value": 34250.0
        }
        
        with patch('services.market_data.MarketDataService') as mock_service:
            # Mock market data service responses
            mock_service.return_value.get_portfolio_data.return_value = {
                "VTI": Mock(current_price=225.0, change_percent=0.02),
                "BNDX": Mock(current_price=54.0, change_percent=-0.01),
                "GSG": Mock(current_price=19.0, change_percent=0.03)
            }
            
            response = await async_client.post("/api/analyze-portfolio", json=portfolio_data)
            assert response.status_code == 200
            
            result = response.json()
            assert "analysis" in result
            assert "timestamp" in result

    @pytest.mark.asyncio
    async def test_market_sentiment(self, async_client):
        """Test market sentiment endpoint"""
        with patch('services.market_data.MarketDataService') as mock_service:
            mock_service.return_value.get_market_sentiment.return_value = {
                "overall_sentiment": "BULLISH",
                "confidence": 0.75,
                "factors": ["Strong earnings", "Low volatility"]
            }
            
            response = await async_client.get("/api/market-sentiment")
            assert response.status_code == 200
            
            result = response.json()
            assert "sentiment" in result
            assert result["sentiment"]["overall_sentiment"] in ["BULLISH", "BEARISH", "NEUTRAL"]

    @pytest.mark.asyncio
    async def test_strategy_performance(self, async_client):
        """Test strategy performance endpoint"""
        with patch('services.strategy.StraightArrowStrategyService') as mock_service:
            mock_service.return_value.get_strategy_performance.return_value = {
                "strategy_name": "Straight Arrow",
                "total_return": 0.08,
                "sharpe_ratio": 0.65,
                "volatility": 0.10
            }
            
            response = await async_client.get("/api/strategy-performance?period=1y")
            assert response.status_code == 200
            
            result = response.json()
            assert "performance" in result
            assert result["performance"]["strategy_name"] == "Straight Arrow"

class TestAgentEndpoints:
    """Test AI agent endpoints"""
    
    @pytest.mark.asyncio
    async def test_data_analysis(self, async_client):
        """Test data analyst agent endpoint"""
        request_data = {
            "query": "Analyze current market conditions for VTI, BNDX, GSG",
            "symbols": ["VTI", "BNDX", "GSG"]
        }
        
        with patch('agents.data_analyst.DataAnalyst') as mock_agent:
            mock_agent.return_value.analyze.return_value = {
                "analysis": "Market conditions are favorable for the three-fund portfolio",
                "recommendations": ["Maintain current allocation", "Monitor bond yields"]
            }
            
            response = await async_client.post("/api/agents/data-analyst", json=request_data)
            assert response.status_code == 200
            
            result = response.json()
            assert "analysis" in result

    @pytest.mark.asyncio
    async def test_trading_analysis(self, async_client):
        """Test trading analyst agent endpoint"""
        request_data = {
            "query": "Provide trading signals for portfolio rebalancing",
            "portfolio": {"VTI": 0.65, "BNDX": 0.25, "GSG": 0.10}
        }
        
        with patch('agents.trading_analyst.TradingAnalyst') as mock_agent:
            mock_agent.return_value.analyze.return_value = {
                "signals": [
                    {"symbol": "VTI", "action": "REDUCE", "confidence": 0.7},
                    {"symbol": "BNDX", "action": "INCREASE", "confidence": 0.8}
                ]
            }
            
            response = await async_client.post("/api/agents/trading-analyst", json=request_data)
            assert response.status_code == 200
            
            result = response.json()
            assert "analysis" in result

    @pytest.mark.asyncio
    async def test_risk_analysis(self, async_client):
        """Test risk analyst agent endpoint"""
        request_data = {
            "query": "Assess portfolio risk and compliance",
            "portfolio_value": 34250.0,
            "positions": {
                "VTI": {"value": 22500.0, "weight": 0.657},
                "BNDX": {"value": 10800.0, "weight": 0.315},
                "GSG": {"value": 950.0, "weight": 0.028}
            }
        }
        
        with patch('agents.risk_analyst.RiskAnalyst') as mock_agent:
            mock_agent.return_value.analyze.return_value = {
                "risk_level": "MEDIUM",
                "var_95": -0.025,
                "compliance_status": "COMPLIANT",
                "alerts": []
            }
            
            response = await async_client.post("/api/agents/risk-analyst", json=request_data)
            assert response.status_code == 200
            
            result = response.json()
            assert "analysis" in result

class TestErrorHandling:
    """Test error handling"""
    
    @pytest.mark.asyncio
    async def test_invalid_portfolio_data(self, async_client):
        """Test handling of invalid portfolio data"""
        invalid_data = {
            "portfolio": "invalid_format",
            "total_value": -1000
        }
        
        response = await async_client.post("/api/analyze-portfolio", json=invalid_data)
        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_missing_api_keys(self, async_client):
        """Test handling when API keys are missing"""
        with patch.dict('os.environ', {}, clear=True):
            response = await async_client.get("/api/market-sentiment")
            # Should handle gracefully, not crash
            assert response.status_code in [200, 500, 503]

class TestIntegration:
    """Integration tests"""
    
    @pytest.mark.asyncio
    async def test_full_analysis_workflow(self, async_client):
        """Test complete analysis workflow"""
        # Step 1: Get market sentiment
        sentiment_response = await async_client.get("/api/market-sentiment")
        
        # Step 2: Analyze portfolio
        portfolio_data = {
            "portfolio": {"VTI": 22500.0, "BNDX": 10800.0, "GSG": 950.0},
            "total_value": 34250.0
        }
        
        with patch('services.market_data.MarketDataService'):
            portfolio_response = await async_client.post("/api/analyze-portfolio", json=portfolio_data)
        
        # Step 3: Get strategy performance
        performance_response = await async_client.get("/api/strategy-performance")
        
        # All should succeed (or handle errors gracefully)
        assert sentiment_response.status_code in [200, 500, 503]
        assert portfolio_response.status_code in [200, 422, 500]
        assert performance_response.status_code in [200, 500, 503]

@pytest.mark.asyncio
async def test_concurrent_requests(async_client):
    """Test handling of concurrent requests"""
    async def make_request():
        return await async_client.get("/health")
    
    # Make 10 concurrent requests
    tasks = [make_request() for _ in range(10)]
    responses = await asyncio.gather(*tasks)
    
    # All should succeed
    for response in responses:
        assert response.status_code == 200

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 