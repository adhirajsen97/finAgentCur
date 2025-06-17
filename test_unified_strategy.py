#!/usr/bin/env python3
"""
Test script for the Unified Investment Strategy API

This script tests the unified strategy API both locally and on the deployed service.
"""

import asyncio
import httpx
import json
from datetime import datetime

# Test configuration
LOCAL_URL = "http://localhost:8000"
DEPLOYED_URL = "https://finagentcur.onrender.com"

# Sample test data
TEST_PORTFOLIO = {
    "portfolio": {
        "VTI": 50000.0,
        "BNDX": 30000.0, 
        "GSG": 20000.0
    },
    "total_value": 100000.0,
    "available_cash": 10000.0,
    "time_horizon": "3 weeks",
    "risk_tolerance": "moderate",
    "investment_goals": ["rebalancing", "growth"]
}

async def test_unified_strategy_api(base_url: str):
    """Test the unified strategy API"""
    print(f"\n🚀 Testing Unified Strategy API at {base_url}")
    print("=" * 60)
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Test the unified strategy endpoint
            print("📊 Sending strategy request...")
            response = await client.post(
                f"{base_url}/api/unified-strategy",
                json=TEST_PORTFOLIO
            )
            
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                
                # Validate response structure
                assert data["status"] == "success", "Response status should be success"
                assert "strategy" in data, "Response should contain strategy"
                assert "trade_orders" in data["strategy"], "Strategy should contain trade_orders"
                
                strategy = data["strategy"]
                trade_orders = strategy["trade_orders"]
                
                print(f"✅ API Response Valid")
                print(f"📈 Strategy ID: {strategy['strategy_id']}")
                print(f"⏰ Created: {strategy['created_at']}")
                print(f"🎯 Strategy Type: {strategy['strategy_type']}")
                print(f"💰 Trade Orders: {len(trade_orders)}")
                
                # Display trade orders
                if trade_orders:
                    print(f"\n📋 TRADE ORDERS:")
                    print("-" * 40)
                    total_buy = 0
                    total_sell = 0
                    
                    for i, order in enumerate(trade_orders, 1):
                        print(f"{i}. {order['action']} {order['symbol']}")
                        print(f"   Quantity: {order['quantity']} shares")
                        print(f"   Amount: ${order['dollar_amount']:,.2f}")
                        print(f"   Price: ${order['current_price']:.2f}")
                        print(f"   Priority: {order['priority']}")
                        print(f"   Reason: {order['reason']}")
                        print()
                        
                        if order['action'] == 'BUY':
                            total_buy += order['dollar_amount']
                        else:
                            total_sell += order['dollar_amount']
                    
                    print(f"💵 Total Buys: ${total_buy:,.2f}")
                    print(f"💸 Total Sells: ${total_sell:,.2f}")
                    print(f"📊 Net Cash Flow: ${total_sell - total_buy:,.2f}")
                
                # Display strategy summary
                summary = strategy.get("strategy_summary", {})
                print(f"\n📊 STRATEGY SUMMARY:")
                print("-" * 40)
                print(f"Overview: {summary.get('overview', 'N/A')}")
                print(f"Rebalancing Needed: {summary.get('rebalancing_needed', 'N/A')}")
                print(f"Market Conditions: {summary.get('market_conditions', 'N/A')}")
                
                # Display execution guidelines
                guidelines = strategy.get("execution_guidelines", {})
                print(f"\n⚡ EXECUTION GUIDELINES:")
                print("-" * 40)
                print(f"Order: {guidelines.get('execution_order', 'N/A')}")
                print(f"Timing: {guidelines.get('timing', 'N/A')}")
                print(f"High Priority Trades: {guidelines.get('high_priority_count', 0)}")
                
                # Display risk warnings
                warnings = strategy.get("risk_warnings", [])
                if warnings:
                    print(f"\n⚠️  RISK WARNINGS:")
                    print("-" * 40)
                    for warning in warnings:
                        print(f"• {warning}")
                
                print(f"\n✅ Test completed successfully!")
                return True
                
            else:
                print(f"❌ API Error: {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

async def test_existing_endpoints(base_url: str):
    """Test that existing endpoints still work"""
    print(f"\n🔍 Testing existing endpoints at {base_url}")
    print("=" * 60)
    
    endpoints_to_test = [
        ("/health", "GET", None),
        ("/api/market-data", "POST", {"symbols": ["VTI", "BNDX", "GSG"]}),
        ("/api/analyze-portfolio", "POST", {
            "portfolio": {"VTI": 50000.0, "BNDX": 30000.0, "GSG": 20000.0},
            "total_value": 100000.0
        })
    ]
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        for endpoint, method, payload in endpoints_to_test:
            try:
                print(f"Testing {method} {endpoint}...")
                
                if method == "GET":
                    response = await client.get(f"{base_url}{endpoint}")
                else:
                    response = await client.post(f"{base_url}{endpoint}", json=payload)
                
                if response.status_code == 200:
                    print(f"✅ {endpoint} - OK")
                else:
                    print(f"❌ {endpoint} - Status: {response.status_code}")
                    
            except Exception as e:
                print(f"❌ {endpoint} - Error: {e}")

def print_integration_summary():
    """Print integration summary"""
    print(f"\n" + "="*80)
    print(f"🎉 UNIFIED STRATEGY API INTEGRATION SUMMARY")
    print(f"="*80)
    
    print(f"""
✅ **What You Now Have:**
   • Unified API that orchestrates all your existing services
   • Actionable trade orders with specific quantities and dollar amounts
   • Priority-based execution system (HIGH/MEDIUM/LOW)
   • Market context and AI insights
   • Risk warnings and compliance information
   • Execution guidelines and timing recommendations

📋 **Your frontend can now:**
   • Call one API endpoint to get a complete investment strategy
   • Receive specific buy/sell orders ready for execution
   • Sort orders by priority for optimal execution
   • Get reasoning for each trade recommendation
   • Monitor execution with provided guidelines

🚀 **Next Steps:**
   1. Add the unified_strategy_api.py to your project
   2. Import and integrate into main_enhanced_complete.py
   3. Deploy the updated API
   4. Update your frontend to consume trade orders
   5. Implement trade execution logic in your frontend

📞 **API Endpoint:**
   POST /api/unified-strategy
   
📖 **Documentation:**
   See UNIFIED_STRATEGY_INTEGRATION.md for complete integration guide
""")
    
    print(f"="*80)

async def main():
    """Run all tests"""
    print(f"🧪 Unified Strategy API Test Suite")
    print(f"Started at: {datetime.now().isoformat()}")
    
    # Test deployed service first
    print(f"\n🌐 Testing DEPLOYED service...")
    deployed_success = await test_unified_strategy_api(DEPLOYED_URL)
    
    if deployed_success:
        print(f"\n✅ Deployed API works! Testing existing endpoints...")
        await test_existing_endpoints(DEPLOYED_URL)
    
    # Uncomment to test local service if running
    # print(f"\n💻 Testing LOCAL service...")
    # local_success = await test_unified_strategy_api(LOCAL_URL)
    
    print_integration_summary()

if __name__ == "__main__":
    asyncio.run(main()) 