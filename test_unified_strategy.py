#!/usr/bin/env python3
"""
Test suite for the Enhanced Unified Strategy API
Tests the complete workflow: Questionnaire -> Risk Assessment -> Investment Strategy -> Market Data Integration
"""

import pytest
import httpx
import json
from datetime import datetime

# Test configuration
BASE_URL = "http://localhost:8000"

# Test data for different risk profiles
RISK_PROFILES_TEST_DATA = {
    1: {  # Ultra Conservative
        "questionnaire": {
            "investment_goal": "preserve_capital",
            "time_horizon": "less_than_1_year",
            "risk_tolerance": "sell_everything",
            "experience_level": "no_experience",
            "age": 65,
            "liquidity_needs": "more_than_60_percent",
            "sector_preferences": ["utilities", "consumer_staples"],
            "investment_restrictions": ["no_crypto", "no_options"]
        },
        "investment_amount": 50000,
        "expected_risk_score": 1
    },
    2: {  # Conservative
        "questionnaire": {
            "investment_goal": "income_generation",
            "time_horizon": "1_to_3_years",
            "risk_tolerance": "accept_small_losses",
            "experience_level": "some_experience",
            "age": 45,
            "liquidity_needs": "40_to_60_percent",
            "sector_preferences": ["bonds", "dividend_stocks"],
            "investment_restrictions": ["no_leveraged_products"]
        },
        "investment_amount": 100000,
        "expected_risk_score": 2
    },
    3: {  # Moderate
        "questionnaire": {
            "investment_goal": "balanced_growth",
            "time_horizon": "5_to_10_years",
            "risk_tolerance": "moderate_losses",
            "experience_level": "experienced",
            "age": 40,
            "liquidity_needs": "20_to_40_percent",
            "sector_preferences": ["technology", "healthcare"],
            "investment_restrictions": []
        },
        "investment_amount": 200000,
        "expected_risk_score": 3
    },
    4: {  # Aggressive
        "questionnaire": {
            "investment_goal": "wealth_accumulation",
            "time_horizon": "more_than_10_years",
            "risk_tolerance": "accept_significant_losses",
            "experience_level": "professional",
            "age": 30,
            "liquidity_needs": "less_than_20_percent",
            "sector_preferences": ["technology", "growth_stocks"],
            "investment_restrictions": []
        },
        "investment_amount": 300000,
        "expected_risk_score": 4
    },
    5: {  # Maximum Risk
        "questionnaire": {
            "investment_goal": "wealth_accumulation",
            "time_horizon": "more_than_10_years",
            "risk_tolerance": "excited_about_opportunity",
            "experience_level": "professional",
            "age": 25,
            "liquidity_needs": "none_accessible",
            "sector_preferences": ["technology", "crypto", "leveraged_products"],
            "investment_restrictions": []
        },
        "investment_amount": 500000,
        "expected_risk_score": 5
    }
}

class TestEnhancedUnifiedStrategy:
    """Test suite for Enhanced Unified Strategy API"""
    
    @pytest.fixture(scope="class")
    def client(self):
        """HTTP client for API testing"""
        return httpx.Client(base_url=BASE_URL, timeout=30.0)
    
    def test_health_check(self, client):
        """Test that the API is running and features are available"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "questionnaire_analyst" in data["features"]["agents"]
        print(f"✅ Health check passed - API version {data['version']}")
    
    @pytest.mark.parametrize("risk_score", [1, 2, 3, 4, 5])
    def test_questionnaire_analysis(self, client, risk_score):
        """Test questionnaire analysis for each risk profile"""
        test_data = RISK_PROFILES_TEST_DATA[risk_score]
        
        # Test questionnaire analysis
        questionnaire_request = {
            "questionnaire": json.dumps(test_data["questionnaire"])
        }
        
        response = client.post("/api/analyze-questionnaire", json=questionnaire_request)
        assert response.status_code == 200
        
        data = response.json()
        assert data["risk_score"] == test_data["expected_risk_score"]
        assert data["risk_level"] in ["Very Low", "Low", "Moderate", "High", "Very High"]
        assert data["portfolio_strategy_name"] is not None
        assert "analysis_details" in data
        
        print(f"✅ Risk {risk_score}/5: {data['portfolio_strategy_name']}")
        print(f"   Risk Level: {data['risk_level']}")
        
        # Store for next test
        test_data["questionnaire_result"] = data
    
    @pytest.mark.parametrize("risk_score", [1, 2, 3, 4, 5])
    def test_enhanced_unified_strategy(self, client, risk_score):
        """Test the enhanced unified strategy API for each risk profile"""
        test_data = RISK_PROFILES_TEST_DATA[risk_score]
        
        # First get questionnaire results
        questionnaire_request = {
            "questionnaire": json.dumps(test_data["questionnaire"])
        }
        
        q_response = client.post("/api/analyze-questionnaire", json=questionnaire_request)
        assert q_response.status_code == 200
        q_data = q_response.json()
        
        # Now test unified strategy with questionnaire results
        strategy_request = {
            "risk_score": q_data["risk_score"],
            "risk_level": q_data["risk_level"],
            "portfolio_strategy_name": q_data["portfolio_strategy_name"],
            "investment_amount": test_data["investment_amount"],
            "investment_restrictions": test_data["questionnaire"].get("investment_restrictions", []),
            "sector_preferences": test_data["questionnaire"].get("sector_preferences", []),
            "time_horizon": "5-10 years",
            "experience_level": test_data["questionnaire"]["experience_level"],
            "liquidity_needs": test_data["questionnaire"]["liquidity_needs"]
        }
        
        response = client.post("/api/unified-strategy", json=strategy_request)
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert "strategy" in data
        
        strategy = data["strategy"]
        
        # Validate strategy structure
        assert "risk_profile" in strategy
        assert "investment_allocations" in strategy
        assert "ai_strategy_analysis" in strategy
        assert "market_context" in strategy
        assert "next_review_date" in strategy
        
        # Validate risk profile
        risk_profile = strategy["risk_profile"]
        assert risk_profile["risk_score"] == risk_score
        assert "confidence_score" in risk_profile
        assert 0.5 <= risk_profile["confidence_score"] <= 1.0
        
        # Validate investment allocations
        allocations = strategy["investment_allocations"]
        assert len(allocations) > 0
        
        total_allocation = sum(alloc["dollar_amount"] for alloc in allocations)
        assert abs(total_allocation - test_data["investment_amount"]) < 1.0  # Allow for rounding
        
        # Validate confidence score meets minimum thresholds (higher is always better)
        min_thresholds = {1: 0.85, 2: 0.75, 3: 0.65, 4: 0.55, 5: 0.45}
        min_threshold = min_thresholds.get(risk_score, 0.65)
        assert risk_profile["confidence_score"] >= min_threshold, \
            f"Risk {risk_score} confidence {risk_profile['confidence_score']:.2f} below minimum {min_threshold}"
        
        # Validate re-evaluation date
        review_date = strategy["next_review_date"]
        assert review_date is not None
        
        # Validate AI analysis
        ai_analysis = strategy["ai_strategy_analysis"]
        assert "confidence_score" in ai_analysis
        assert "recommendation" in ai_analysis
        assert ai_analysis["recommendation"] in ["PROCEED", "PROCEED_WITH_CAUTION"]
        
        print(f"✅ Enhanced Strategy {risk_score}/5:")
        print(f"   Investment: ${test_data['investment_amount']:,}")
        print(f"   Allocations: {len(allocations)} positions")
        print(f"   Confidence: {risk_profile['confidence_score']:.0%}")
        print(f"   Recommendation: {ai_analysis['recommendation']}")
        print(f"   Review Date: {review_date[:10]}")
    
    def test_confidence_score_logic(self, client):
        """Test that confidence scores meet minimum thresholds and reflect strategy quality"""
        confidence_scores = {}
        min_thresholds = {1: 0.85, 2: 0.75, 3: 0.65, 4: 0.55, 5: 0.45}
        
        for risk_score in [1, 2, 3, 4, 5]:
            test_data = RISK_PROFILES_TEST_DATA[risk_score]
            
            # Get questionnaire results
            questionnaire_request = {
                "questionnaire": json.dumps(test_data["questionnaire"])
            }
            
            q_response = client.post("/api/analyze-questionnaire", json=questionnaire_request)
            q_data = q_response.json()
            
            # Get strategy
            strategy_request = {
                "risk_score": q_data["risk_score"],
                "risk_level": q_data["risk_level"],
                "portfolio_strategy_name": q_data["portfolio_strategy_name"],
                "investment_amount": test_data["investment_amount"],
                "investment_restrictions": test_data["questionnaire"].get("investment_restrictions", []),
                "sector_preferences": test_data["questionnaire"].get("sector_preferences", [])
            }
            
            response = client.post("/api/unified-strategy", json=strategy_request)
            data = response.json()
            
            confidence_scores[risk_score] = data["strategy"]["risk_profile"]["confidence_score"]
        
        # Validate minimum thresholds are met
        for risk_score, confidence in confidence_scores.items():
            min_threshold = min_thresholds[risk_score]
            assert confidence >= min_threshold, \
                f"Risk {risk_score} confidence {confidence:.2f} below minimum threshold {min_threshold}"
        
        print("✅ Confidence score logic validated:")
        for risk_score, confidence in confidence_scores.items():
            min_threshold = min_thresholds[risk_score]
            status = "✓" if confidence >= min_threshold else "✗"
            print(f"   Risk {risk_score}/5: {confidence:.0%} confidence (min: {min_threshold:.0%}) {status}")
    
    def test_market_data_integration(self, client):
        """Test that market data is properly integrated into strategy"""
        # Use moderate risk profile for this test
        test_data = RISK_PROFILES_TEST_DATA[3]
        
        # Get questionnaire results
        questionnaire_request = {
            "questionnaire": json.dumps(test_data["questionnaire"])
        }
        
        q_response = client.post("/api/analyze-questionnaire", json=questionnaire_request)
        q_data = q_response.json()
        
        # Get strategy
        strategy_request = {
            "risk_score": q_data["risk_score"],
            "risk_level": q_data["risk_level"],
            "portfolio_strategy_name": q_data["portfolio_strategy_name"],
            "investment_amount": test_data["investment_amount"],
            "investment_restrictions": [],
            "sector_preferences": ["technology"]
        }
        
        response = client.post("/api/unified-strategy", json=strategy_request)
        data = response.json()
        
        strategy = data["strategy"]
        
        # Validate market data integration
        assert "market_context" in strategy
        market_context = strategy["market_context"]
        assert "market_data" in market_context
        assert "market_sentiment" in market_context
        
        # Validate allocation comparison
        assert "allocation_comparison" in strategy
        comparison = strategy["allocation_comparison"]
        assert "theoretical_allocations" in comparison
        assert "recommended_stocks" in comparison
        assert "current_market_prices" in comparison
        
        # Check that actual prices are included
        prices = comparison["current_market_prices"]
        assert len(prices) > 0
        
        # Validate that allocations have actual market prices
        allocations = strategy["investment_allocations"]
        market_value_allocations = [a for a in allocations if a.get("current_price", 0) > 0]
        assert len(market_value_allocations) > 0, "At least some allocations should have market prices"
        
        print("✅ Market data integration validated:")
        print(f"   Market prices: {len(prices)} symbols")
        print(f"   Allocations with prices: {len(market_value_allocations)}/{len(allocations)}")
    
    def test_investment_profile_templates(self, client):
        """Test that all investment profile templates are properly used"""
        expected_profiles = [
            "Ultra Conservative Capital Preservation",
            "Conservative Balanced Growth", 
            "Moderate Growth with Value Focus",
            "Aggressive Growth with Trend Following",
            "Maximum Growth High-Risk Portfolio"
        ]
        
        used_profiles = []
        
        for risk_score in [1, 2, 3, 4, 5]:
            test_data = RISK_PROFILES_TEST_DATA[risk_score]
            
            questionnaire_request = {
                "questionnaire": json.dumps(test_data["questionnaire"])
            }
            
            response = client.post("/api/analyze-questionnaire", json=questionnaire_request)
            data = response.json()
            
            used_profiles.append(data["portfolio_strategy_name"])
        
        # Validate that the correct profile names are used
        for i, expected_profile in enumerate(expected_profiles):
            assert expected_profile in used_profiles[i], \
                f"Risk {i+1} should use profile '{expected_profile}', but got '{used_profiles[i]}'"
        
        print("✅ Investment profile templates validated:")
        for i, profile in enumerate(used_profiles):
            print(f"   Risk {i+1}/5: {profile}")
    
    def test_review_date_logic(self, client):
        """Test that review dates are appropriately set based on risk scores"""
        review_intervals = {}
        
        for risk_score in [1, 2, 3, 4, 5]:
            test_data = RISK_PROFILES_TEST_DATA[risk_score]
            
            # Get questionnaire results
            questionnaire_request = {
                "questionnaire": json.dumps(test_data["questionnaire"])
            }
            
            q_response = client.post("/api/analyze-questionnaire", json=questionnaire_request)
            q_data = q_response.json()
            
            # Get strategy
            strategy_request = {
                "risk_score": q_data["risk_score"],
                "risk_level": q_data["risk_level"],
                "portfolio_strategy_name": q_data["portfolio_strategy_name"],
                "investment_amount": test_data["investment_amount"],
                "investment_restrictions": [],
                "sector_preferences": []
            }
            
            response = client.post("/api/unified-strategy", json=strategy_request)
            data = response.json()
            
            review_date = data["strategy"]["next_review_date"]
            
            # Calculate days until review
            from datetime import datetime
            review_dt = datetime.fromisoformat(review_date.replace('Z', '+00:00'))
            now = datetime.now(review_dt.tzinfo)
            days_until_review = (review_dt - now).days
            
            review_intervals[risk_score] = days_until_review
        
        # Validate that higher risk scores have shorter review intervals
        assert review_intervals[5] < review_intervals[1], \
            "Highest risk should have shortest review interval"
        
        print("✅ Review date logic validated:")
        for risk_score, days in review_intervals.items():
            print(f"   Risk {risk_score}/5: Review in {days} days")

def run_comprehensive_test():
    """Run all tests and provide a summary"""
    print("🚀 Starting Enhanced Unified Strategy API Tests...")
    print("=" * 60)
    
    client = httpx.Client(base_url=BASE_URL, timeout=30.0)
    test_suite = TestEnhancedUnifiedStrategy()
    
    try:
        # Run all tests
        test_suite.test_health_check(client)
        print()
        
        for risk_score in [1, 2, 3, 4, 5]:
            test_suite.test_questionnaire_analysis(client, risk_score)
        print()
        
        for risk_score in [1, 2, 3, 4, 5]:
            test_suite.test_enhanced_unified_strategy(client, risk_score)
        print()
        
        test_suite.test_confidence_score_logic(client)
        print()
        
        test_suite.test_market_data_integration(client)
        print()
        
        test_suite.test_investment_profile_templates(client)
        print()
        
        test_suite.test_review_date_logic(client)
        print()
        
        print("🎉 ALL TESTS PASSED!")
        print("Enhanced Unified Strategy API is working correctly!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise
    finally:
        client.close()

if __name__ == "__main__":
    run_comprehensive_test() 