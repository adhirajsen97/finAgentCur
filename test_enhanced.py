"""
Test suite for Enhanced FinAgent API
Tests all AI agents, risk metrics, compliance, and market data features
"""

import asyncio
import json
from typing import Dict, Any
import httpx
import pytest

# Test configuration
BASE_URL = "http://localhost:8000"
TEST_PORTFOLIO = {
    "VTI": 60000.0,
    "BNDX": 25000.0,
    "GSG": 15000.0
}
TEST_TOTAL_VALUE = 100000.0

class TestEnhancedFinAgent:
    """Test suite for enhanced FinAgent features"""
    
    def __init__(self):
        self.client = httpx.AsyncClient(base_url=BASE_URL)
    
    async def test_health_check(self):
        """Test enhanced health check endpoint"""
        print("\n🔍 Testing Enhanced Health Check...")
        
        try:
            response = await self.client.get("/health")
            assert response.status_code == 200
            
            data = response.json()
            print(f"✅ Status: {data['status']}")
            print(f"✅ Version: {data['version']}")
            print(f"✅ Strategy: {data['strategy']}")
            print(f"✅ Features: {data['features']}")
            
            # Check for enhanced features
            features = data['features']
            assert 'agents' in features
            assert 'compliance' in features
            assert 'risk_metrics' in features
            
            print("✅ Enhanced health check passed!")
            return True
            
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return False
    
    async def test_enhanced_portfolio_analysis(self):
        """Test enhanced portfolio analysis with risk metrics and compliance"""
        print("\n📊 Testing Enhanced Portfolio Analysis...")
        
        try:
            payload = {
                "portfolio": TEST_PORTFOLIO,
                "total_value": TEST_TOTAL_VALUE
            }
            
            response = await self.client.post("/api/analyze-portfolio", json=payload)
            assert response.status_code == 200
            
            data = response.json()
            analysis = data["analysis"]
            
            # Check basic analysis
            assert "strategy" in analysis
            assert "current_weights" in analysis
            assert "target_allocation" in analysis
            assert "drift_analysis" in analysis
            
            # Check enhanced features
            assert "portfolio_metrics" in analysis
            assert "compliance" in analysis
            assert "risk_assessment" in analysis
            
            # Check risk metrics
            portfolio_metrics = analysis["portfolio_metrics"]
            assert "expected_return" in portfolio_metrics
            assert "volatility" in portfolio_metrics
            assert "sharpe_ratio" in portfolio_metrics
            
            # Check compliance
            compliance = analysis["compliance"]
            assert "status" in compliance
            assert "disclosures" in compliance
            
            print(f"✅ Strategy: {analysis['strategy']}")
            print(f"✅ Expected Return: {portfolio_metrics['expected_return']:.1%}")
            print(f"✅ Volatility: {portfolio_metrics['volatility']:.1%}")
            print(f"✅ Sharpe Ratio: {portfolio_metrics['sharpe_ratio']:.2f}")
            print(f"✅ Compliance Status: {compliance['status']}")
            print("✅ Enhanced portfolio analysis passed!")
            return True
            
        except Exception as e:
            print(f"❌ Enhanced portfolio analysis failed: {e}")
            return False
    
    async def test_market_data_with_technical_analysis(self):
        """Test market data with technical analysis"""
        print("\n📈 Testing Market Data with Technical Analysis...")
        
        try:
            payload = {
                "symbols": ["VTI", "BNDX", "GSG"]
            }
            
            response = await self.client.post("/api/market-data", json=payload)
            assert response.status_code == 200
            
            data = response.json()
            quotes = data["quotes"]
            
            for symbol in ["VTI", "BNDX", "GSG"]:
                assert symbol in quotes
                quote = quotes[symbol]
                
                # Check basic quote data
                assert "symbol" in quote
                assert "price" in quote
                assert "change" in quote
                
                # Check technical analysis
                if "technical_analysis" in quote:
                    tech = quote["technical_analysis"]
                    assert "trend" in tech
                    assert "recommendation" in tech
                    print(f"✅ {symbol}: ${quote['price']:.2f} - {tech['trend']} - {tech['recommendation']}")
                else:
                    print(f"✅ {symbol}: ${quote['price']:.2f}")
            
            print("✅ Market data with technical analysis passed!")
            return True
            
        except Exception as e:
            print(f"❌ Market data test failed: {e}")
            return False
    
    async def test_market_sentiment(self):
        """Test market sentiment endpoint"""
        print("\n🎭 Testing Market Sentiment...")
        
        try:
            response = await self.client.get("/api/market-sentiment")
            assert response.status_code == 200
            
            data = response.json()
            sentiment = data["sentiment"]
            
            assert "overall_sentiment" in sentiment
            assert "fear_greed_index" in sentiment
            assert "market_trend" in sentiment
            assert "volatility_index" in sentiment
            
            print(f"✅ Overall Sentiment: {sentiment['overall_sentiment']}")
            print(f"✅ Fear & Greed Index: {sentiment['fear_greed_index']}")
            print(f"✅ Market Trend: {sentiment['market_trend']}")
            print("✅ Market sentiment test passed!")
            return True
            
        except Exception as e:
            print(f"❌ Market sentiment test failed: {e}")
            return False
    
    async def test_data_analyst_agent(self):
        """Test data analyst AI agent"""
        print("\n🤖 Testing Data Analyst AI Agent...")
        
        try:
            payload = {
                "query": "What are the current market conditions for the Straight Arrow strategy?",
                "symbols": ["VTI", "BNDX", "GSG"]
            }
            
            response = await self.client.post("/api/agents/data-analyst", json=payload)
            assert response.status_code == 200
            
            data = response.json()
            analysis = data["analysis"]
            
            assert "agent" in analysis
            assert analysis["agent"] == "data_analyst"
            assert "query" in analysis
            assert "analysis" in analysis
            assert "confidence" in analysis
            
            print(f"✅ Agent: {analysis['agent']}")
            print(f"✅ Confidence: {analysis['confidence']}")
            print(f"✅ Analysis Preview: {analysis['analysis'][:100]}...")
            print("✅ Data analyst agent test passed!")
            return True
            
        except Exception as e:
            print(f"❌ Data analyst agent test failed: {e}")
            return False
    
    async def test_risk_analyst_agent(self):
        """Test risk analyst AI agent"""
        print("\n⚖️ Testing Risk Analyst AI Agent...")
        
        try:
            payload = {
                "portfolio": TEST_PORTFOLIO,
                "total_value": TEST_TOTAL_VALUE,
                "time_horizon": "Long Term"
            }
            
            response = await self.client.post("/api/agents/risk-analyst", json=payload)
            assert response.status_code == 200
            
            data = response.json()
            analysis = data["analysis"]
            
            assert "agent" in analysis
            assert analysis["agent"] == "risk_analyst"
            assert "portfolio" in analysis
            assert "analysis" in analysis
            assert "confidence" in analysis
            
            print(f"✅ Agent: {analysis['agent']}")
            print(f"✅ Time Horizon: {analysis['time_horizon']}")
            print(f"✅ Confidence: {analysis['confidence']}")
            print(f"✅ Analysis Preview: {analysis['analysis'][:100]}...")
            print("✅ Risk analyst agent test passed!")
            return True
            
        except Exception as e:
            print(f"❌ Risk analyst agent test failed: {e}")
            return False
    
    async def test_trading_analyst_agent(self):
        """Test trading analyst AI agent"""
        print("\n📊 Testing Trading Analyst AI Agent...")
        
        try:
            payload = {
                "symbols": ["VTI", "BNDX", "GSG"],
                "analysis_type": "technical"
            }
            
            response = await self.client.post("/api/agents/trading-analyst", json=payload)
            assert response.status_code == 200
            
            data = response.json()
            analysis = data["analysis"]
            
            assert "agent" in analysis
            assert analysis["agent"] == "trading_analyst"
            assert "symbols" in analysis
            assert "analysis_type" in analysis
            assert "analysis" in analysis
            assert "confidence" in analysis
            
            print(f"✅ Agent: {analysis['agent']}")
            print(f"✅ Analysis Type: {analysis['analysis_type']}")
            print(f"✅ Symbols: {', '.join(analysis['symbols'])}")
            print(f"✅ Confidence: {analysis['confidence']}")
            print(f"✅ Analysis Preview: {analysis['analysis'][:100]}...")
            print("✅ Trading analyst agent test passed!")
            return True
            
        except Exception as e:
            print(f"❌ Trading analyst agent test failed: {e}")
            return False
    
    async def test_strategy_performance(self):
        """Test strategy performance endpoint"""
        print("\n🎯 Testing Strategy Performance...")
        
        try:
            response = await self.client.get("/api/strategy-performance")
            assert response.status_code == 200
            
            data = response.json()
            
            assert "strategy" in data
            assert "target_allocation" in data
            assert "expected_metrics" in data
            assert "performance_summary" in data
            
            performance = data["performance_summary"]
            assert "expected_annual_return" in performance
            assert "expected_volatility" in performance
            assert "sharpe_ratio" in performance
            assert "risk_level" in performance
            
            print(f"✅ Strategy: {data['strategy']}")
            print(f"✅ Expected Return: {performance['expected_annual_return']}")
            print(f"✅ Expected Volatility: {performance['expected_volatility']}")
            print(f"✅ Sharpe Ratio: {performance['sharpe_ratio']}")
            print(f"✅ Risk Level: {performance['risk_level']}")
            print("✅ Strategy performance test passed!")
            return True
            
        except Exception as e:
            print(f"❌ Strategy performance test failed: {e}")
            return False
    
    async def test_compliance_disclosures(self):
        """Test compliance disclosures endpoint"""
        print("\n⚖️ Testing Compliance Disclosures...")
        
        try:
            response = await self.client.get("/api/compliance/disclosures")
            assert response.status_code == 200
            
            data = response.json()
            
            assert "disclosures" in data
            assert "last_updated" in data
            assert "version" in data
            
            disclosures = data["disclosures"]
            assert len(disclosures) > 0
            assert any("educational purposes" in d.lower() for d in disclosures)
            assert any("investment advisor" in d.lower() for d in disclosures)
            
            print(f"✅ Number of disclosures: {len(disclosures)}")
            print(f"✅ Version: {data['version']}")
            print("✅ Sample disclosures:")
            for i, disclosure in enumerate(disclosures[:3]):
                print(f"   {i+1}. {disclosure}")
            print("✅ Compliance disclosures test passed!")
            return True
            
        except Exception as e:
            print(f"❌ Compliance disclosures test failed: {e}")
            return False
    
    async def test_portfolio_history(self):
        """Test portfolio history endpoint"""
        print("\n📚 Testing Portfolio History...")
        
        try:
            response = await self.client.get("/api/portfolio-history")
            assert response.status_code == 200
            
            data = response.json()
            assert "history" in data
            
            # History might be empty if no previous analyses
            history = data["history"]
            print(f"✅ History entries: {len(history)}")
            
            if history:
                # Check first entry structure
                entry = history[0]
                expected_fields = ["id", "analysis_date", "total_value", "allocation"]
                for field in expected_fields:
                    if field in entry:
                        print(f"✅ Found field: {field}")
            
            print("✅ Portfolio history test passed!")
            return True
            
        except Exception as e:
            print(f"❌ Portfolio history test failed: {e}")
            return False

    async def test_bulk_ticker_api(self):
        """Test the bulk ticker API with multiple symbols"""
        print("\n🚀 Testing Bulk Ticker API...")
        
        try:
            payload = {
                "symbols": ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
            }
            
            response = await self.client.post("/api/ticker/bulk", json=payload)
            print(f"Bulk Ticker Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Successfully fetched {data['success_count']}/{data['total_requested']} symbols")
                print(f"   Errors: {data['error_count']}")
                
                # Display some ticker data
                for symbol, ticker_data in list(data['tickers'].items())[:3]:  # Show first 3
                    print(f"   {symbol}: ${ticker_data['price']:.2f} ({ticker_data['change_percent']})")
                
                if data['errors']:
                    print(f"   Errors encountered: {data['errors']}")
                    
                assert data['success_count'] > 0, "Should have at least some successful fetches"
                assert data['total_requested'] == 5, "Should show 5 total requested"
                assert data['source'] == "finnhub", "Should use Finnhub as source"
                
                print("✅ Bulk ticker API test passed!")
                return True
            else:
                print(f"❌ Bulk ticker request failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Bulk ticker API test failed: {e}")
            return False

    async def test_bulk_ticker_api_edge_cases(self):
        """Test bulk ticker API edge cases"""
        print("\n🔧 Testing Bulk Ticker API Edge Cases...")
        
        try:
            # Test with duplicate symbols
            payload = {
                "symbols": ["AAPL", "AAPL", "MSFT", "MSFT", "GOOGL"]
            }
            
            response = await self.client.post("/api/ticker/bulk", json=payload)
            print(f"Duplicate Symbols Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Handled duplicates: requested 5, processed {data['total_requested']}")
                assert data['total_requested'] == 3, "Should remove duplicates"
            
            # Test with empty list (should fail validation)
            payload = {"symbols": []}
            response = await self.client.post("/api/ticker/bulk", json=payload)
            print(f"Empty List Status: {response.status_code}")
            assert response.status_code == 422, "Should reject empty symbol list"
            
            # Test with too many symbols (over limit)
            payload = {
                "symbols": [f"SYM{i:02d}" for i in range(25)]  # 25 symbols (over 20 limit)
            }
            response = await self.client.post("/api/ticker/bulk", json=payload)
            print(f"Too Many Symbols Status: {response.status_code}")
            assert response.status_code == 422, "Should reject too many symbols"
            
            print("✅ Bulk ticker edge cases test passed!")
            return True
            
        except Exception as e:
            print(f"❌ Bulk ticker edge cases test failed: {e}")
            return False
    
    async def run_all_tests(self):
        """Run all enhanced tests"""
        print("🚀 Starting Enhanced FinAgent Test Suite...")
        print("=" * 60)
        
        tests = [
            self.test_health_check,
            self.test_enhanced_portfolio_analysis,
            self.test_market_data_with_technical_analysis,
            self.test_market_sentiment,
            self.test_data_analyst_agent,
            self.test_risk_analyst_agent,
            self.test_trading_analyst_agent,
            self.test_strategy_performance,
            self.test_compliance_disclosures,
            self.test_portfolio_history,
            self.test_bulk_ticker_api,
            self.test_bulk_ticker_api_edge_cases
        ]
        
        passed = 0
        failed = 0
        
        for test in tests:
            try:
                result = await test()
                if result:
                    passed += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"❌ Test {test.__name__} failed with exception: {e}")
                failed += 1
        
        print("\n" + "=" * 60)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 60)
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"📈 Success Rate: {passed/(passed+failed)*100:.1f}%")
        
        if failed == 0:
            print("\n🎉 ALL ENHANCED TESTS PASSED! 🎉")
            print("The enhanced FinAgent API is working correctly with:")
            print("• Multiple AI Agents (Data, Risk, Trading)")
            print("• Advanced Risk Metrics (Sharpe ratio, volatility)")
            print("• Compliance Framework (disclosures, validation)")
            print("• Enhanced Market Data (technical analysis)")
            print("• Strategy Performance Tracking")
        else:
            print(f"\n⚠️  {failed} test(s) failed. Please check the API server.")
        
        await self.client.aclose()
        return failed == 0

async def main():
    """Main test runner"""
    tester = TestEnhancedFinAgent()
    success = await tester.run_all_tests()
    return success



if __name__ == "__main__":
    # Run the tests
    success = asyncio.run(main())
    exit(0 if success else 1) 