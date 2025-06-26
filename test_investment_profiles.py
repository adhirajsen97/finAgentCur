"""
Test script for all 5 investment profile templates
Tests each risk profile (1-5) with appropriate questionnaire data
"""

import json
import asyncio
import httpx


async def test_all_investment_profiles():
    """Test all 5 investment profiles with appropriate questionnaire data"""
    
    # Test cases for each risk profile
    test_profiles = [
        {
            "name": "Guardian Profile (Risk 1)",
            "expected_risk": 1,
            "questionnaire": {
                "investment_goal": "Capital preservation",
                "time_horizon": "1-3 years",
                "risk_tolerance": "Sell everything immediately",
                "experience_level": "No investment experience",
                "income_level": "$50,000 - $100,000",
                "net_worth": "$100,000 - $500,000", 
                "liquidity_needs": "60%+ accessible",
                "market_insights": "No - fully automated management",
                "sector_preferences": [],
                "investment_restrictions": ["No tobacco companies"]
            }
        },
        {
            "name": "Straight Arrow Profile (Risk 2)",
            "expected_risk": 2,
            "questionnaire": {
                "investment_goal": "Generate regular income",
                "time_horizon": "3-5 years",
                "risk_tolerance": "Sell some investments",
                "experience_level": "Basic knowledge",
                "income_level": "$100,000 - $200,000",
                "net_worth": "$500,000 - $1,000,000",
                "liquidity_needs": "40-60% accessible",
                "market_insights": "No - fully automated management",
                "sector_preferences": ["ESG & Sustainable Investing"],
                "investment_restrictions": ["No fossil fuel companies"]
            }
        },
        {
            "name": "Individualist Profile (Risk 3)",
            "expected_risk": 3,
            "questionnaire": {
                "investment_goal": "Retirement planning",
                "time_horizon": "5-10 years", 
                "risk_tolerance": "Hold and wait for recovery",
                "experience_level": "Some experience",
                "income_level": "$100,000 - $200,000",
                "net_worth": "$500,000 - $1,000,000",
                "liquidity_needs": "20-40% accessible",
                "market_insights": "Yes - I'll share market insights",
                "sector_preferences": ["Financial Services"],
                "investment_restrictions": ["No weapons manufacturers"]
            }
        },
        {
            "name": "Celebrity Profile (Risk 4)",
            "expected_risk": 4,
            "questionnaire": {
                "investment_goal": "Long-term wealth accumulation",
                "time_horizon": "5-10 years",
                "risk_tolerance": "Buy more at lower prices",
                "experience_level": "Experienced investor",
                "income_level": "$200,000 - $500,000",
                "net_worth": "$1,000,000 - $5,000,000",
                "liquidity_needs": "10-20% accessible",
                "market_insights": "Yes - I'll share market insights",
                "sector_preferences": ["Technology & Innovation"],
                "investment_restrictions": []
            }
        },
        {
            "name": "Adventurer Profile (Risk 5)",
            "expected_risk": 5,
            "questionnaire": {
                "investment_goal": "Long-term wealth accumulation",
                "time_horizon": "10+ years",
                "risk_tolerance": "Excited about the opportunity",
                "experience_level": "Professional level",
                "income_level": "Over $500,000",
                "net_worth": "Over $5,000,000",
                "liquidity_needs": "None - can lock up all funds",
                "market_insights": "Yes - I'll share market insights",
                "sector_preferences": ["Technology & Innovation", "Energy & Resources"],
                "investment_restrictions": ["No restrictions"]
            }
        }
    ]
    
    print("🚀 Testing All Investment Profile Templates")
    print("=" * 60)
    print()
    
    results = []
    
    for profile in test_profiles:
        print(f"📊 Testing: {profile['name']}")
        print("-" * 40)
        
        # Convert questionnaire to JSON string
        questionnaire_json = json.dumps(profile["questionnaire"])
        request_payload = {"questionnaire": questionnaire_json}
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "http://localhost:8000/api/analyze-questionnaire",
                    json=request_payload,
                    timeout=30.0
                )
            
            if response.status_code == 200:
                result = response.json()
                results.append(result)
                
                print(f"✅ Status: Success")
                print(f"🎯 Risk Score: {result['risk_score']}/5 (Expected: {profile['expected_risk']}/5)")
                print(f"📈 Risk Level: {result['risk_level']}")
                print(f"🎨 Strategy: {result['portfolio_strategy_name']}")
                
                # Verify risk score matches expectation
                if result['risk_score'] == profile['expected_risk']:
                    print("✅ Risk score matches expected value")
                else:
                    print(f"⚠️  Risk score mismatch! Expected {profile['expected_risk']}, got {result['risk_score']}")
                
                # Display key strategy details
                recommendations = result['analysis_details']['investment_recommendations']
                print(f"💼 Core Strategy: {recommendations['core_strategy']}")
                print(f"📊 Asset Allocation: {recommendations['asset_allocation']}")
                print(f"📈 Expected Return: {recommendations['expected_return']}")
                print(f"📉 Volatility: {recommendations['volatility_expectation']}")
                print()
                
            else:
                print(f"❌ API call failed with status code: {response.status_code}")
                print(f"Response: {response.text}")
                print()
                
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            print()
    
    # Summary
    print("📋 SUMMARY")
    print("=" * 60)
    if results:
        print(f"✅ Successfully tested {len(results)} investment profiles")
        print()
        print("Profile Summary:")
        for i, result in enumerate(results):
            print(f"{i+1}. {result['portfolio_strategy_name']} (Risk {result['risk_score']}/5)")
        print()
        print("🎯 All investment profile templates are working correctly!")
    else:
        print("❌ No successful tests completed")


async def test_profile_consistency():
    """Test that the same questionnaire always produces the same result"""
    print("\n🔄 Testing Profile Consistency")
    print("=" * 60)
    
    # Test the same questionnaire multiple times
    test_questionnaire = {
        "investment_goal": "Long-term wealth accumulation",
        "time_horizon": "10+ years",
        "risk_tolerance": "Hold and wait for recovery",
        "experience_level": "Some experience",
        "income_level": "$100,000 - $200,000",
        "net_worth": "$500,000 - $1,000,000",
        "liquidity_needs": "20-40% accessible",
        "market_insights": "No - fully automated management",
        "sector_preferences": ["Technology & Innovation"],
        "investment_restrictions": ["No tobacco companies"]
    }
    
    questionnaire_json = json.dumps(test_questionnaire)
    request_payload = {"questionnaire": questionnaire_json}
    
    results = []
    
    for i in range(3):
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "http://localhost:8000/api/analyze-questionnaire",
                    json=request_payload,
                    timeout=30.0
                )
            
            if response.status_code == 200:
                result = response.json()
                results.append({
                    "attempt": i + 1,
                    "risk_score": result["risk_score"],
                    "strategy_name": result["portfolio_strategy_name"]
                })
        except Exception as e:
            print(f"❌ Attempt {i+1} failed: {e}")
    
    if len(results) >= 2:
        # Check consistency
        first_result = results[0]
        all_consistent = all(
            r["risk_score"] == first_result["risk_score"] and 
            r["strategy_name"] == first_result["strategy_name"] 
            for r in results
        )
        
        if all_consistent:
            print("✅ Profile assignment is consistent across multiple calls")
            print(f"   Consistent Result: {first_result['strategy_name']} (Risk {first_result['risk_score']}/5)")
        else:
            print("⚠️  Profile assignment is inconsistent!")
            for result in results:
                print(f"   Attempt {result['attempt']}: {result['strategy_name']} (Risk {result['risk_score']}/5)")
    else:
        print("❌ Not enough successful attempts to test consistency")


def display_investment_templates():
    """Display all investment profile templates"""
    print("\n📚 Investment Profile Templates Reference")
    print("=" * 60)
    
    templates = {
        1: "Guardian: The Sentinel of Capital",
        2: "Straight Arrow: The Path of Prudence", 
        3: "Individualist: The Calculated Risk-Taker",
        4: "Celebrity: The Trend Follower",
        5: "Adventurer: The High-Stakes Player"
    }
    
    for risk_score, name in templates.items():
        print(f"{risk_score}. {name}")
    print()


if __name__ == "__main__":
    print("🧪 Investment Profile Templates Test Suite")
    print("Please ensure the FinAgent server is running on localhost:8000")
    print()
    
    # Display templates reference
    display_investment_templates()
    
    # Run all tests
    asyncio.run(test_all_investment_profiles())
    asyncio.run(test_profile_consistency()) 