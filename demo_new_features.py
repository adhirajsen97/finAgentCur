"""
Demo: New Portfolio Performance API and Analyst Recommendations Integration
Showcases both features working together
"""

import asyncio
import httpx
import json
from datetime import datetime, timedelta

async def demo_portfolio_performance_concept():
    """
    Demo how the Portfolio Performance API would work
    This shows the concept and data structure without requiring the full implementation
    """
    print("🎯 DEMO: Portfolio Performance API Concept")
    print("=" * 50)
    
    # Sample trade data that would be sent to the API
    sample_trades = [
        {
            "symbol": "AAPL",
            "action": "BUY",
            "quantity": 50,
            "execution_price": 150.00,
            "execution_date": "2024-01-15T10:30:00Z",
            "total_amount": 7500.00,
            "fees": 1.00,
            "strategy_source": "unified_strategy"
        },
        {
            "symbol": "MSFT", 
            "action": "BUY",
            "quantity": 25,
            "execution_price": 300.00,
            "execution_date": "2024-01-16T11:00:00Z",
            "total_amount": 7500.00,
            "fees": 1.00,
            "strategy_source": "unified_strategy"
        },
        {
            "symbol": "AAPL",
            "action": "SELL",
            "quantity": 10,
            "execution_price": 170.00,
            "execution_date": "2024-02-15T14:00:00Z",
            "total_amount": 1700.00,
            "fees": 1.00,
            "strategy_source": "profit_taking"
        }
    ]
    
    print("📊 Sample Trade History:")
    total_invested = 0
    for trade in sample_trades:
        action_icon = "🟢" if trade["action"] == "BUY" else "🔴"
        print(f"   {action_icon} {trade['symbol']}: {trade['action']} {trade['quantity']} @ ${trade['execution_price']:.2f}")
        if trade["action"] == "BUY":
            total_invested += trade["total_amount"] + trade["fees"]
    
    print(f"\n💰 Total Invested: ${total_invested:,.2f}")
    
    # Show what the API response would look like using current prices
    print("\n📈 Expected API Response Structure:")
    print("""
    {
      "total_invested": 15002.00,
      "current_value": 16850.00,
      "total_return": 1848.00,
      "total_return_percent": 12.32,
      "annualized_return": 18.5,
      "volatility": 14.2,
      "sharpe_ratio": 0.78,
      "max_drawdown": 3.1,
      "positions": [
        {
          "symbol": "AAPL",
          "current_shares": 40.0,
          "average_cost_basis": 150.10,
          "current_price": 165.00,
          "unrealized_pnl": 596.00,
          "unrealized_pnl_percent": 9.94,
          "weight_in_portfolio": 39.1
        },
        {
          "symbol": "MSFT", 
          "current_shares": 25.0,
          "average_cost_basis": 300.04,
          "current_price": 325.00,
          "unrealized_pnl": 624.00,
          "unrealized_pnl_percent": 8.31,
          "weight_in_portfolio": 48.2
        }
      ],
      "time_series": [...], // Chart data for frontend
      "best_performer": {"symbol": "AAPL", "unrealized_pnl_percent": 9.94},
      "worst_performer": {"symbol": "MSFT", "unrealized_pnl_percent": 8.31}
    }
    """)

async def demo_analyst_integration_concept():
    """
    Demo how Analyst Recommendations would enhance the unified strategy
    """
    print("\n\n🏢 DEMO: Analyst Recommendations Integration")
    print("=" * 50)
    
    # Show how analyst data would be integrated
    print("📊 Sample Analyst Data Integration:")
    
    sample_symbols = ["AAPL", "MSFT", "GOOGL", "TSLA"]
    
    # Simulate analyst recommendation data
    analyst_data = {
        "AAPL": {
            "weighted_score": 4.2,
            "overall_recommendation": "Buy",
            "confidence": 85.0,
            "consensus_strength": "Strong Consensus",
            "rationale": "Buy based on strong analyst coverage (12 analysts) showing positive sentiment with most analysts recommending buying."
        },
        "MSFT": {
            "weighted_score": 3.8,
            "overall_recommendation": "Buy", 
            "confidence": 78.0,
            "consensus_strength": "Moderate Consensus",
            "rationale": "Buy based on moderate analyst coverage (8 analysts) showing positive sentiment."
        },
        "GOOGL": {
            "weighted_score": 3.5,
            "overall_recommendation": "Hold",
            "confidence": 65.0,
            "consensus_strength": "Weak Consensus",
            "rationale": "Hold based on mixed analyst sentiment with moderate coverage."
        },
        "TSLA": {
            "weighted_score": 2.8,
            "overall_recommendation": "Hold",
            "confidence": 55.0,
            "consensus_strength": "Neutral/Mixed",
            "rationale": "Hold based on neutral analyst sentiment with mixed recommendations."
        }
    }
    
    print("🎯 Enhanced Investment Recommendations:")
    for symbol in sample_symbols:
        data = analyst_data[symbol]
        score_color = "🟢" if data["weighted_score"] >= 3.5 else "🟡" if data["weighted_score"] >= 2.5 else "🔴"
        confidence_bar = "█" * int(data["confidence"] / 10)
        
        print(f"\n   {score_color} {symbol}:")
        print(f"      Analyst Score: {data['weighted_score']}/5.0")
        print(f"      Recommendation: {data['overall_recommendation']}")
        print(f"      Confidence: {confidence_bar} {data['confidence']:.0f}%")
        print(f"      Consensus: {data['consensus_strength']}")
        
        # Show how this affects allocation
        if data["weighted_score"] >= 4.0:
            allocation_adjustment = "↗️ Increase allocation by 15-30%"
        elif data["weighted_score"] >= 3.5:
            allocation_adjustment = "↗️ Slight increase (5-15%)"
        elif data["weighted_score"] <= 2.5:
            allocation_adjustment = "↘️ Consider reducing allocation"
        else:
            allocation_adjustment = "➡️ Maintain current allocation"
        
        print(f"      Strategy Impact: {allocation_adjustment}")

async def demo_enhanced_unified_strategy():
    """
    Demo how the enhanced unified strategy would work with real API call
    """
    print("\n\n🚀 DEMO: Enhanced Unified Strategy API")
    print("=" * 50)
    
    # Test the existing unified strategy API to show current capabilities
    strategy_request = {
        "risk_score": 3,
        "risk_level": "Moderate",
        "portfolio_strategy_name": "Moderate Growth with Value Focus",
        "investment_amount": 50000.00,
        "investment_restrictions": [],
        "sector_preferences": ["Technology", "Healthcare"],
        "time_horizon": "5-10 years",
        "experience_level": "Some experience",
        "liquidity_needs": "20-40% accessible",
        "current_portfolio": {},
        "current_portfolio_value": 0.0
    }
    
    print("📝 Strategy Request:")
    print(f"   Risk Level: {strategy_request['risk_level']} (Score: {strategy_request['risk_score']}/5)")
    print(f"   Investment: ${strategy_request['investment_amount']:,.2f}")
    print(f"   Sectors: {', '.join(strategy_request['sector_preferences'])}")
    print(f"   Time Horizon: {strategy_request['time_horizon']}")
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "http://localhost:8000/api/unified-strategy",
                json=strategy_request
            )
            
            if response.status_code == 200:
                result = response.json()
                strategy = result['strategy']
                
                print(f"\n✅ Strategy Created Successfully!")
                print(f"   Strategy ID: {strategy['strategy_id']}")
                print(f"   Confidence Score: {strategy['confidence_score']:.1f}/100")
                print(f"   Created: {strategy['created_at']}")
                
                # Show investment allocations
                allocations = strategy.get('investment_allocations', [])
                print(f"\n💰 Investment Allocations ({len(allocations)} positions):")
                
                total_allocated = 0
                for i, allocation in enumerate(allocations[:5]):  # Show first 5
                    amount = allocation['dollar_amount']
                    total_allocated += amount
                    percentage = (amount / strategy_request['investment_amount']) * 100
                    
                    print(f"   {i+1}. {allocation['symbol']}: ${amount:,.0f} ({percentage:.1f}%)")
                    print(f"      Current Price: ${allocation['current_price']:.2f}")
                    print(f"      Action: {allocation['action']} | Priority: {allocation['priority']}")
                    print(f"      Reason: {allocation['reason'][:60]}...")
                
                print(f"\n📊 Total Allocated: ${total_allocated:,.2f} ({total_allocated/strategy_request['investment_amount']*100:.1f}%)")
                print(f"📅 Next Review: {strategy['next_review_date']}")
                
                # Show what analyst enhancement would add
                print(f"\n🏢 ANALYST ENHANCEMENT PREVIEW:")
                print(f"   With analyst integration, each allocation would include:")
                print(f"   • Analyst recommendation (Buy/Hold/Sell)")
                print(f"   • Analyst confidence score (0-100%)")
                print(f"   • Consensus strength (Strong/Moderate/Weak)")
                print(f"   • Allocation adjustment factor (0.5x to 2.0x)")
                print(f"   • Risk adjustment based on analyst sentiment")
                
            else:
                print(f"❌ API Error: {response.status_code}")
                print(f"   Response: {response.text}")
                
    except Exception as e:
        print(f"❌ Request failed: {e}")

async def demo_real_time_integration():
    """
    Demo how the bulk ticker API supports real-time portfolio tracking
    """
    print("\n\n⚡ DEMO: Real-Time Portfolio Tracking")
    print("=" * 50)
    
    # Test symbols from a typical portfolio
    portfolio_symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
    
    print(f"📊 Fetching real-time data for portfolio symbols...")
    
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            start_time = datetime.now()
            
            response = await client.post(
                "http://localhost:8000/api/ticker/bulk",
                json={"symbols": portfolio_symbols}
            )
            
            elapsed = (datetime.now() - start_time).total_seconds()
            
            if response.status_code == 200:
                data = response.json()
                
                print(f"✅ Fetched {data['success_count']}/{len(portfolio_symbols)} symbols in {elapsed:.2f}s")
                print(f"📈 Real-Time Portfolio Data:")
                
                total_value = 0
                for symbol, ticker_data in data['tickers'].items():
                    # Simulate 10 shares of each for demo
                    shares = 10
                    position_value = ticker_data['price'] * shares
                    total_value += position_value
                    
                    change_color = "🟢" if ticker_data['change'] >= 0 else "🔴"
                    print(f"   {change_color} {symbol}: ${ticker_data['price']:.2f} "
                          f"({ticker_data['change_percent']}) | "
                          f"Position: ${position_value:.0f}")
                
                print(f"\n💼 Total Portfolio Value (10 shares each): ${total_value:,.0f}")
                print(f"⚡ Data Freshness: Real-time from Finnhub")
                print(f"🔄 Update Frequency: On-demand (60 calls/minute limit)")
                
                print(f"\n🎯 This real-time data enables:")
                print(f"   • Live P&L calculations")
                print(f"   • Instant portfolio rebalancing alerts") 
                print(f"   • Real-time risk monitoring")
                print(f"   • Accurate performance tracking")
                
            else:
                print(f"❌ Bulk ticker API error: {response.status_code}")
                
    except Exception as e:
        print(f"❌ Real-time data fetch failed: {e}")

async def main():
    """Run all demos"""
    print("🎉 FINAGENT ENHANCED FEATURES DEMO")
    print("🚀 Portfolio Performance API + Analyst Recommendations")
    print("=" * 60)
    
    await demo_portfolio_performance_concept()
    await demo_analyst_integration_concept()
    await demo_enhanced_unified_strategy()
    await demo_real_time_integration()
    
    print("\n" + "=" * 60)
    print("✅ DEMO COMPLETE!")
    print("\n🎯 Key Takeaways:")
    print("   1. Portfolio Performance API provides complete P&L tracking")
    print("   2. Analyst Recommendations enhance investment intelligence")
    print("   3. Real-time market data enables accurate valuations")
    print("   4. Enhanced unified strategy combines AI + human expertise")
    print("   5. All features integrate seamlessly with existing system")
    print("\n🚀 Ready for full implementation!")

if __name__ == "__main__":
    asyncio.run(main())
