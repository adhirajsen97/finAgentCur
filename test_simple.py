#!/usr/bin/env python3
"""
Simple FinAgent Test Script
Tests the simplified Straight Arrow strategy implementation
"""

import asyncio
import aiohttp
import json
from datetime import datetime

class SimpleFinAgentTest:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def print_section(self, title: str):
        """Print formatted section header"""
        print(f"\n{'='*50}")
        print(f"🚀 {title}")
        print('='*50)
    
    async def test_health_check(self):
        """Test health check"""
        self.print_section("HEALTH CHECK")
        
        async with self.session.get(f"{self.base_url}/health") as resp:
            result = await resp.json()
            print(json.dumps(result, indent=2))
            
            print(f"\n✅ Status: {result['status']}")
            print(f"📊 Strategy: {result['strategy']}")
            print(f"🗄️  Database: {result['database']}")
            print(f"📈 Market Data: {result['market_data']}")
            print(f"🤖 AI Service: {result['ai_service']}")
    
    async def test_portfolio_analysis(self):
        """Test portfolio analysis"""
        self.print_section("PORTFOLIO ANALYSIS")
        
        # Sample portfolio
        portfolio_data = {
            "portfolio": {
                "VTI": 25000,   # 50% (target: 60%)
                "BNDX": 20000,  # 40% (target: 30%)
                "GSG": 5000     # 10% (target: 10%)
            },
            "total_value": 50000
        }
        
        print("📊 Analyzing portfolio:")
        for symbol, value in portfolio_data['portfolio'].items():
            weight = value / portfolio_data['total_value']
            print(f"   {symbol}: ${value:,} ({weight:.1%})")
        
        async with self.session.post(
            f"{self.base_url}/api/analyze-portfolio",
            json=portfolio_data
        ) as resp:
            result = await resp.json()
            analysis = result.get('analysis', {})
            
            print(f"\n📈 Analysis Results:")
            print(f"   Strategy: {analysis.get('strategy')}")
            print(f"   Total Value: ${analysis.get('total_value'):,}")
            
            # Show drift analysis
            print(f"\n📊 Drift Analysis:")
            drift_analysis = analysis.get('drift_analysis', {})
            for symbol, data in drift_analysis.items():
                current = data.get('current_weight', 0)
                target = data.get('target_weight', 0)
                drift = data.get('drift', 0)
                print(f"   {symbol}: {current:.1%} → {target:.1%} (drift: {drift:+.1%})")
            
            # Show recommendations
            recommendations = analysis.get('recommendations', [])
            if recommendations:
                print(f"\n🎯 Recommendations:")
                for rec in recommendations:
                    print(f"   {rec['action']} {rec['symbol']}: {rec['current_percent']:.1f}% → {rec['target_percent']:.1f}%")
            else:
                print(f"\n✅ Portfolio is well balanced!")
    
    async def test_market_data(self):
        """Test market data"""
        self.print_section("MARKET DATA")
        
        symbols = ["VTI", "BNDX", "GSG"]
        market_request = {"symbols": symbols}
        
        print(f"📈 Requesting market data for: {', '.join(symbols)}")
        
        async with self.session.post(
            f"{self.base_url}/api/market-data",
            json=market_request
        ) as resp:
            result = await resp.json()
            
            print(f"\n💰 Current Quotes:")
            quotes = result.get('quotes', {})
            for symbol, quote in quotes.items():
                if 'error' not in quote:
                    source = quote.get('source', 'unknown')
                    print(f"   {symbol}: ${quote.get('price', 'N/A')} ({quote.get('change_percent', 'N/A')}) [{source}]")
                else:
                    print(f"   {symbol}: {quote['error']}")
    
    async def test_ai_analysis(self):
        """Test AI analysis"""
        self.print_section("AI ANALYSIS")
        
        ai_request = {
            "query": "What is the Straight Arrow strategy and why is it good for beginners?",
            "symbols": ["VTI", "BNDX", "GSG"]
        }
        
        print(f"🤖 AI Query: {ai_request['query']}")
        
        async with self.session.post(
            f"{self.base_url}/api/agents/data-analyst",
            json=ai_request
        ) as resp:
            result = await resp.json()
            analysis = result.get('analysis', {})
            
            print(f"\n🧠 AI Response:")
            print(f"   Confidence: {analysis.get('confidence', 0):.1%}")
            print(f"\n📝 Analysis:")
            print(analysis.get('analysis', 'No response'))
    
    async def test_portfolio_history(self):
        """Test portfolio history"""
        self.print_section("PORTFOLIO HISTORY")
        
        async with self.session.get(f"{self.base_url}/api/portfolio-history") as resp:
            result = await resp.json()
            
            history = result.get('history', [])
            if history:
                print(f"📊 Found {len(history)} historical analyses:")
                for i, record in enumerate(history[:3], 1):
                    date = record.get('analysis_date', 'Unknown')
                    value = record.get('total_value', 0)
                    print(f"   {i}. {date[:10]}: ${value:,.2f}")
            else:
                print("📊 No historical data found")
                if 'message' in result:
                    print(f"   {result['message']}")
    
    async def run_all_tests(self):
        """Run all tests"""
        print("🚀 FinAgent Simplified Test Suite")
        print("=" * 50)
        print("Testing Straight Arrow Strategy Implementation")
        
        try:
            await self.test_health_check()
            await self.test_portfolio_analysis()
            await self.test_market_data()
            await self.test_ai_analysis()
            await self.test_portfolio_history()
            
            print(f"\n✅ All tests completed successfully!")
            print("🎉 Simplified FinAgent is working!")
            
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            raise

async def main():
    """Main test function"""
    print("Starting FinAgent Simplified Tests...")
    
    # Test local server
    base_url = "http://localhost:8000"
    
    try:
        async with SimpleFinAgentTest(base_url) as test:
            await test.run_all_tests()
    except Exception as e:
        print(f"❌ Failed to connect to {base_url}: {e}")
        print("💡 Make sure the server is running: python main_simplified.py")
    
    print("\n🏁 Test Suite Complete!")

if __name__ == "__main__":
    asyncio.run(main()) 