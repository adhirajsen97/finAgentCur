# Questionnaire Analysis API Guide

## Overview

The `/api/analyze-questionnaire` endpoint analyzes user investment questionnaires to determine risk profiles and recommend appropriate investment strategies from 5 predefined templates.

## Investment Profile Templates

The API uses 5 investment profiles, ordered from least to most risky:

## 1. **Ultra Conservative Capital Preservation** (Risk 1/5)
- **Strategy**: Capital Preservation + Inflation Hedge
- **Allocation**: 50% SGOV, 30% VPU, 20% TIPS
- **Expected Return**: 3-5% annually (inflation-adjusted)
- **Volatility**: <3% annual volatility
- **Best For**: Short to medium term with capital preservation priority

## 2. **Conservative Balanced Growth** (Risk 2/5)
- **Strategy**: Diversified Three-Fund Style
- **Allocation**: 60% VTI, 30% BNDX, 10% GSG
- **Expected Return**: 6-8% annually over long term
- **Volatility**: 10-12% annual volatility
- **Best For**: Medium to long term with moderate risk tolerance

## 3. **Moderate Growth with Value Focus** (Risk 3/5)
- **Strategy**: Graham-Buffett Value + Momentum Tilt
- **Allocation**: 70% VTI, 15% VTV, 10% MTUM, 5% VMOT
- **Expected Return**: 7-10% annually with value tilt
- **Volatility**: 12-15% annual volatility
- **Best For**: Long term with active management overlay

## 4. **Aggressive Growth with Trend Following** (Risk 4/5)
- **Strategy**: Growth-oriented with trend following and innovation exposure
- **Allocation**: 40% QQQ, 30% VUG, 20% ARKK, 10% VMOT
- **Expected Return**: 8-12% annually with higher variance
- **Volatility**: 15-20% annual volatility
- **Best For**: Medium to long term with active trend monitoring

## 5. **Maximum Growth High-Risk Portfolio** (Risk 5/5)
- **Strategy**: Barbell: Stability + High Volatility Growth
- **Allocation**: 30% BND, 25% TQQQ, 20% SOXL, 15% ARKK, 10% SPXL
- **Expected Return**: 10-15% annually with high risk/reward
- **Volatility**: 25-40% annual volatility
- **Best For**: Long term with high risk tolerance and volatility acceptance

## API Usage

### Endpoint
```
POST /api/analyze-questionnaire
```

### Request Format
```json
{
  "questionnaire": "{\"investment_goal\": \"Long-term wealth accumulation\", \"time_horizon\": \"10+ years\", \"risk_tolerance\": \"Hold and wait for recovery\", \"experience_level\": \"Some experience\", \"income_level\": \"$100,000 - $200,000\", \"net_worth\": \"$500,000 - $1,000,000\", \"liquidity_needs\": \"20-40% accessible\", \"market_insights\": \"No - fully automated management\", \"sector_preferences\": [\"Technology & Innovation\"], \"investment_restrictions\": [\"No tobacco companies\"]}"
}
```

### Questionnaire Fields

#### Required Fields
- **investment_goal**: Primary investment objective
  - Options: "Capital preservation", "Generate regular income", "Retirement planning", "Long-term wealth accumulation", "Save for major purchase"
  
- **time_horizon**: Investment time frame
  - Options: "Less than 1 year", "1-3 years", "3-5 years", "5-10 years", "10+ years"
  
- **risk_tolerance**: Reaction to 20% portfolio decline
  - Options: "Sell everything immediately", "Sell some investments", "Hold and wait for recovery", "Buy more at lower prices", "Excited about the opportunity"
  
- **experience_level**: Investment experience
  - Options: "No investment experience", "Basic knowledge", "Some experience", "Experienced investor", "Professional level"

#### Optional Fields
- **income_level**: Annual household income
- **net_worth**: Approximate net worth
- **liquidity_needs**: Required accessibility percentage
- **market_insights**: Whether user wants to share market insights
- **sector_preferences**: Preferred investment sectors (array)
- **investment_restrictions**: Investment restrictions (array)

### Response Format
```json
{
  "risk_score": 4,
  "risk_level": "High",
  "portfolio_strategy_name": "Celebrity: The Trend Follower",
  "analysis_details": {
    "detailed_analysis": "Based on your questionnaire responses...",
    "questionnaire_breakdown": {
      "primary_factors": {...},
      "supporting_factors": {...},
      "preferences": {...}
    },
    "investment_recommendations": {
      "profile_name": "Celebrity: The Trend Follower",
      "core_strategy": "Social Sentiment Mirroring + Robo Buffer",
      "asset_allocation": "80% Robo-Advisor Portfolio, 20% Social Sentiment Indices",
      "base_allocation": {"RoboAdvisor": 80, "CrowdIndex": 20},
      "investment_focus": "Trend following with social sentiment analysis",
      "recommended_products": [...],
      "time_horizon_fit": "Medium term with active trend monitoring",
      "volatility_expectation": "15-20% annual volatility",
      "expected_return": "8-12% annually with higher variance",
      "risk_controls": {...},
      "description": "Trend-following approach using social sentiment..."
    },
    "strategy_rationale": "Based on your high risk profile..."
  }
}
```

## Risk Scoring Algorithm

The API calculates risk scores (1-5) based on weighted factors:

1. **Investment Goal** (20% weight): Capital preservation (1) to wealth accumulation (4)
2. **Time Horizon** (25% weight): <1 year (1) to 10+ years (5)
3. **Risk Tolerance** (30% weight): Sell everything (1) to excited about opportunity (5)
4. **Experience Level** (15% weight): No experience (1) to professional (5)
5. **Liquidity Needs** (10% weight): 60%+ accessible (1) to none accessible (5)

## Example Usage

### Python Example
```python
import httpx
import json

# Sample questionnaire data
questionnaire_data = {
    "investment_goal": "Long-term wealth accumulation",
    "time_horizon": "10+ years",
    "risk_tolerance": "Buy more at lower prices",
    "experience_level": "Experienced investor",
    "income_level": "$200,000 - $500,000",
    "net_worth": "$1,000,000 - $5,000,000",
    "liquidity_needs": "10-20% accessible",
    "market_insights": "Yes - I'll share market insights",
    "sector_preferences": ["Technology & Innovation"],
    "investment_restrictions": []
}

# Make API request
response = httpx.post(
    "https://finagentcur.onrender.com/api/analyze-questionnaire",
    json={"questionnaire": json.dumps(questionnaire_data)}
)

result = response.json()
print(f"Risk Score: {result['risk_score']}/5")
print(f"Strategy: {result['portfolio_strategy_name']}")
```

### JavaScript/Frontend Example
```javascript
const analyzeQuestionnaire = async (questionnaireData) => {
  const response = await fetch('/api/analyze-questionnaire', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      questionnaire: JSON.stringify(questionnaireData)
    })
  });
  
  const result = await response.json();
  return {
    riskScore: result.risk_score,
    riskLevel: result.risk_level,
    strategyName: result.portfolio_strategy_name,
    recommendations: result.analysis_details.investment_recommendations
  };
};
```

### cURL Example
```bash
curl -X POST "https://finagentcur.onrender.com/api/analyze-questionnaire" \
  -H "Content-Type: application/json" \
  -d '{
    "questionnaire": "{\"investment_goal\": \"Long-term wealth accumulation\", \"time_horizon\": \"10+ years\", \"risk_tolerance\": \"Hold and wait for recovery\", \"experience_level\": \"Some experience\"}"
  }'
```

## Error Handling

### 400 Bad Request
- Invalid JSON format in questionnaire string
- Missing required fields

### 422 Unprocessable Entity
- Invalid request structure
- Missing questionnaire field

### 500 Internal Server Error
- Server error during analysis
- Falls back to default "Individualist" profile (Risk 3/5)

## Integration Notes

1. **Questionnaire Validation**: Validate questionnaire data on frontend before sending
2. **Error Handling**: Always handle potential API errors gracefully
3. **Persistence**: Analysis results are automatically saved to database if available
4. **Consistency**: Same questionnaire always produces same risk score
5. **Fallback**: System defaults to moderate risk profile on errors

## Testing

Run the comprehensive test suite:
```bash
python test_investment_profiles.py
```

This tests all 5 investment profiles with appropriate questionnaire data and verifies consistency across multiple calls. 