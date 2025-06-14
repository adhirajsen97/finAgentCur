# 🚀 FinAgent API Usage Guide

Welcome to the FinAgent API! This guide will help you interact with your deployed AI-powered investment analysis system.

## 🌐 Base URL

```
https://finagentcur.onrender.com
```

Your FinAgent API is live and ready to use!

## 📋 Quick Start

### 1. Health Check
First, verify your API is running:

```bash
curl https://finagentcur.onrender.com/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:45.123456",
  "version": "1.0.0",
  "environment": "production",
  "database": "supabase",
  "cache": "redis"
}
```

## 🔗 API Endpoints

### Core Endpoints

#### 🏠 Root Endpoint
- **GET** `/`
- **Description:** Basic API information and navigation
- **Authentication:** None required

```bash
curl https://finagentcur.onrender.com/
```

**Response:**
```json
{
  "message": "FinAgent API with Supabase",
  "docs": "/docs",
  "health": "/health"
}
```

#### 💚 Health Check
- **GET** `/health`
- **Description:** System health status
- **Authentication:** None required

```bash
curl https://finagentcur.onrender.com/health
```

### Portfolio Analysis

#### 📊 Analyze Portfolio
- **POST** `/api/analyze-portfolio`
- **Description:** Analyze your portfolio against the Straight Arrow strategy
- **Content-Type:** `application/json`

**Request Body:**
```json
{
  "portfolio": {
    "VTI": 22500.0,
    "BNDX": 10800.0,
    "GSG": 950.0
  },
  "total_value": 34250.0
}
```

**Example Request:**
```bash
curl -X POST "https://finagentcur.onrender.com/api/analyze-portfolio" \
  -H "Content-Type: application/json" \
  -d '{
    "portfolio": {
      "VTI": 22500.0,
      "BNDX": 10800.0,
      "GSG": 950.0
    },
    "total_value": 34250.0
  }'
```

**Response:**
```json
{
  "analysis": {
    "strategy": "Straight Arrow",
    "total_value": 34250.0,
    "current_weights": {
      "VTI": 0.657,
      "BNDX": 0.315,
      "GSG": 0.028
    },
    "target_allocation": {
      "VTI": 0.70,
      "BNDX": 0.25,
      "GSG": 0.05
    },
    "drift_analysis": {
      "VTI": {
        "current_weight": 0.657,
        "target_weight": 0.70,
        "drift": -0.043,
        "drift_percent": -6.14
      }
    },
    "risk_assessment": {
      "risk_level": "MODERATE",
      "diversification_score": 0.85,
      "volatility_estimate": 0.12
    },
    "recommendations": [
      {
        "symbol": "VTI",
        "action": "INCREASE",
        "current_percent": 65.7,
        "target_percent": 70.0
      }
    ],
    "timestamp": "2024-01-15T10:30:45.123456",
    "analysis_id": "uuid-string"
  }
}
```

#### 📈 Portfolio History
- **GET** `/api/portfolio-history`
- **Description:** Get your portfolio analysis history
- **Query Parameters:** 
  - `limit` (optional): Number of records to return (default: 10)

```bash
curl "https://finagentcur.onrender.com/api/portfolio-history?limit=5"
```

**Response:**
```json
{
  "history": [
    {
      "id": "uuid-string",
      "analysis_date": "2024-01-15T10:30:45.123456",
      "total_value": 34250.0,
      "allocation": {
        "VTI": 0.657,
        "BNDX": 0.315,
        "GSG": 0.028
      },
      "drift_analysis": {},
      "risk_assessment": {},
      "rebalance_recommendation": []
    }
  ]
}
```

### AI Agents

#### 🤖 Data Analyst Agent
- **POST** `/api/agents/data-analyst`
- **Description:** Get AI-powered market data analysis
- **Content-Type:** `application/json`

**Request Body:**
```json
{
  "query": "Analyze current market conditions for my portfolio",
  "symbols": ["VTI", "BNDX", "GSG"],
  "portfolio": {
    "VTI": 22500.0,
    "BNDX": 10800.0,
    "GSG": 950.0
  },
  "portfolio_value": 34250.0
}
```

**Example Request:**
```bash
curl -X POST "https://finagentcur.onrender.com/api/agents/data-analyst" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the current market outlook for my three-fund portfolio?",
    "symbols": ["VTI", "BNDX", "GSG"]
  }'
```

**Response:**
```json
{
  "analysis": {
    "analysis": "Market analysis for VTI, BNDX, GSG: Current market conditions show moderate volatility with strong fundamentals in equity markets.",
    "recommendations": [
      "Current market conditions show moderate volatility",
      "Three-fund portfolio allocation remains appropriate",
      "Monitor bond yields for potential rebalancing opportunities"
    ],
    "confidence": 0.75,
    "symbols_analyzed": ["VTI", "BNDX", "GSG"],
    "timestamp": "2024-01-15T10:30:45.123456"
  }
}
```

## 📚 Interactive Documentation

Your API includes built-in interactive documentation:

- **Swagger UI:** `https://finagentcur.onrender.com/docs`
- **ReDoc:** `https://finagentcur.onrender.com/redoc`

## 🔧 Request/Response Format

### Content Type
All POST requests should use:
```
Content-Type: application/json
```

### Standard Response Format
All successful responses follow this pattern:
```json
{
  "status": "success",           // or "error"
  "data": {},                   // response data
  "timestamp": "ISO8601 string" // response timestamp
}
```

### Error Response Format
```json
{
  "detail": "Error message describing what went wrong",
  "status_code": 400
}
```

## 🚨 Error Handling

### Common HTTP Status Codes

- **200 OK:** Request successful
- **400 Bad Request:** Invalid request format
- **422 Unprocessable Entity:** Validation error
- **500 Internal Server Error:** Server error
- **503 Service Unavailable:** Service temporarily unavailable

### Example Error Response
```json
{
  "detail": "Database service not available",
  "status_code": 503
}
```

## 💡 Usage Examples

### Complete Portfolio Analysis Workflow

```bash
# 1. Check API health
curl https://finagentcur.onrender.com/health

# 2. Analyze your current portfolio
curl -X POST "https://finagentcur.onrender.com/api/analyze-portfolio" \
  -H "Content-Type: application/json" \
  -d '{
    "portfolio": {
      "VTI": 25000.0,
      "BNDX": 12000.0,
      "GSG": 2000.0
    },
    "total_value": 39000.0
  }'

# 3. Get AI analysis of market conditions
curl -X POST "https://finagentcur.onrender.com/api/agents/data-analyst" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Should I rebalance my portfolio given current market conditions?",
    "symbols": ["VTI", "BNDX", "GSG"],
    "portfolio": {
      "VTI": 25000.0,
      "BNDX": 12000.0,
      "GSG": 2000.0
    },
    "portfolio_value": 39000.0
  }'

# 4. Check your analysis history
curl "https://finagentcur.onrender.com/api/portfolio-history?limit=5"
```

### Using Python

```python
import requests
import json

# Base URL for your deployed API
BASE_URL = "https://finagentcur.onrender.com"

# Example: Analyze portfolio
def analyze_portfolio():
    url = f"{BASE_URL}/api/analyze-portfolio"
    
    payload = {
        "portfolio": {
            "VTI": 22500.0,
            "BNDX": 10800.0,
            "GSG": 950.0
        },
        "total_value": 34250.0
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

# Example: Get AI analysis
def get_ai_analysis(query, symbols):
    url = f"{BASE_URL}/api/agents/data-analyst"
    
    payload = {
        "query": query,
        "symbols": symbols
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

# Usage
if __name__ == "__main__":
    # Analyze portfolio
    analysis = analyze_portfolio()
    if analysis:
        print("Portfolio Analysis:", json.dumps(analysis, indent=2))
    
    # Get AI insights
    ai_analysis = get_ai_analysis(
        "What's the market outlook for my portfolio?",
        ["VTI", "BNDX", "GSG"]
    )
    if ai_analysis:
        print("AI Analysis:", json.dumps(ai_analysis, indent=2))
```

### Using JavaScript/Node.js

```javascript
const axios = require('axios');

const BASE_URL = 'https://finagentcur.onrender.com';

// Analyze portfolio
async function analyzePortfolio() {
  try {
    const response = await axios.post(`${BASE_URL}/api/analyze-portfolio`, {
      portfolio: {
        VTI: 22500.0,
        BNDX: 10800.0,
        GSG: 950.0
      },
      total_value: 34250.0
    });
    
    return response.data;
  } catch (error) {
    console.error('Error:', error.response?.data || error.message);
    return null;
  }
}

// Get AI analysis
async function getAIAnalysis(query, symbols) {
  try {
    const response = await axios.post(`${BASE_URL}/api/agents/data-analyst`, {
      query: query,
      symbols: symbols
    });
    
    return response.data;
  } catch (error) {
    console.error('Error:', error.response?.data || error.message);
    return null;
  }
}

// Usage
(async () => {
  const analysis = await analyzePortfolio();
  if (analysis) {
    console.log('Portfolio Analysis:', JSON.stringify(analysis, null, 2));
  }
  
  const aiAnalysis = await getAIAnalysis(
    "What's the current market sentiment?",
    ["VTI", "BNDX", "GSG"]
  );
  if (aiAnalysis) {
    console.log('AI Analysis:', JSON.stringify(aiAnalysis, null, 2));
  }
})();
```

## 🔒 Security & Rate Limiting

- All requests should be made over HTTPS
- No authentication required for public endpoints
- Rate limiting may apply (check response headers)
- CORS is enabled for cross-origin requests

## 🆘 Troubleshooting

### Common Issues

1. **503 Service Unavailable**
   - The service might be starting up (wait 1-2 minutes)
   - Database connection issues

2. **422 Validation Error**
   - Check your request body format
   - Ensure all required fields are present
   - Verify data types match the expected format

3. **500 Internal Server Error**
   - Server-side issue
   - Check if the issue persists
   - Review your request format

### Getting Help

- **Interactive Docs:** Visit `/docs` for live API testing
- **API Status:** Check `/health` endpoint
- **Logs:** Monitor Render dashboard for service logs

## 🎯 Next Steps

1. **Try the Interactive Docs:** Visit `https://finagentcur.onrender.com/docs`
2. **Test with Your Data:** Start with the `/health` endpoint
3. **Integrate:** Use the Python/JavaScript examples above
4. **Monitor:** Check your portfolio analysis history regularly

---

**📧 Need Help?** This API is part of your FinAgent AI Investment System. Check the interactive documentation at `/docs` for real-time testing and examples. 