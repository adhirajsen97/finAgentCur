# Market Insight Parser & Portfolio Rebalancer APIs

## 🚀 New APIs Overview

This document provides comprehensive examples for the two new powerful APIs:

1. **URL Financial Content Parser** - Extracts and validates financial content from any URL
2. **AI Portfolio Rebalancer** - Uses market insights to intelligently rebalance portfolios
3. **Combined Workflow API** - One-step URL-to-portfolio-action automation

---

## 📖 API 1: URL Financial Content Parser

### Purpose
Parse any URL to extract financial/market-related content and validate its relevance.

**Supported Content Types:**
- HTML web pages (news articles, blogs, reports)
- Text-based content with UTF-8 or Latin-1 encoding

**Unsupported Content Types:**
- PDF files (will return error message)
- Images (JPG, PNG, GIF, WebP)
- Videos (MP4, AVI, MOV)
- Binary files

### Endpoint
```
POST /api/parse-url
```

### Basic Example
```bash
curl -X POST "https://finagentcur.onrender.com/api/parse-url" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://www.cnbc.com/2024/12/01/tech-stocks-rally-continues.html",
    "max_content_length": 50000
  }'
```

### Expected Response
```json
{
  "url": "https://www.cnbc.com/2024/12/01/tech-stocks-rally-continues.html",
  "is_financial": true,
  "financial_content": {
    "title": "Tech Stocks Rally Continues as AI Investment Surges",
    "content": "Technology stocks continued their upward trajectory...",
    "summary": "Tech sector showing strong momentum driven by AI investments and positive earnings outlook.",
    "financial_keywords": ["stock", "investment", "earnings", "tech", "AI", "rally"],
    "market_sentiment": "BULLISH",
    "sectors_mentioned": ["Technology"],
    "tickers_mentioned": ["AAPL", "MSFT", "NVDA"],
    "key_insights": [
      "AI investments driving tech sector growth",
      "Positive earnings outlook for Q4",
      "Continued institutional buying in tech stocks"
    ],
    "confidence_score": 92.5
  },
  "metadata": {
    "status": "success",
    "content_length": 3547,
    "processing_time": 2.3
  },
  "timestamp": "2024-12-01T15:30:45Z"
}
```

### Real-World URL Examples

#### Parse Bloomberg Article
```bash
curl -X POST "https://finagentcur.onrender.com/api/parse-url" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://www.bloomberg.com/news/articles/2024-12-01/fed-signals-potential-rate-cuts",
    "max_content_length": 30000
  }'
```

#### Parse MarketWatch Analysis
```bash
curl -X POST "https://finagentcur.onrender.com/api/parse-url" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://www.marketwatch.com/story/energy-sector-outlook-2024",
    "max_content_length": 40000
  }'
```

#### Parse Non-Financial Content (Will Return Not Financial)
```bash
curl -X POST "https://finagentcur.onrender.com/api/parse-url" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://www.wikipedia.org/wiki/Python_programming",
    "max_content_length": 20000
  }'
```

#### Parse Unsupported Content Type (PDF Example)
```bash
curl -X POST "https://finagentcur.onrender.com/api/parse-url" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/financial-report.pdf",
    "max_content_length": 50000
  }'
```

**Expected Response for PDF:**
```json
{
  "url": "https://example.com/financial-report.pdf",
  "is_financial": false,
  "error_message": "Failed to fetch content from URL. This may be due to unsupported content type (PDF, image, video) or network issues.",
  "metadata": {
    "status": "failed",
    "reason": "content_fetch_error"
  },
  "timestamp": "2024-12-01T15:30:45Z"
}
```

---

## 🤖 API 2: AI Portfolio Rebalancer

### Purpose
Use market insights from parsed URLs to intelligently rebalance investment portfolios.

**Rebalance Types Available:**
- `market_insight` (default): Fast response to market news/analysis (4-12 hour implementation)
- `time_based`: Scheduled periodic rebalancing (1-2 week implementation)
- `risk_adjustment`: Risk management-focused changes (2-3 day implementation)  
- `manual`: User-directed manual rebalancing (standard timeline)

**Key Features:**
- Sector-based reallocation based on insights
- Risk profile adjustments for market conditions
- Ticker-specific actions for mentioned stocks
- AI-powered rationale tailored to rebalance type
- Dynamic implementation timelines

### Endpoint
```
POST /api/rebalance-portfolio
```

### Complete Example
```bash
curl -X POST "https://finagentcur.onrender.com/api/rebalance-portfolio" \
  -H "Content-Type: application/json" \
  -d '{
    "portfolio_holdings": [
      {
        "symbol": "AAPL",
        "current_shares": 50,
        "current_value": 9500.0,
        "sector": "Technology",
        "purchase_price": 180.00
      },
      {
        "symbol": "MSFT", 
        "current_shares": 30,
        "current_value": 11250.0,
        "sector": "Technology",
        "purchase_price": 350.00
      },
      {
        "symbol": "JNJ",
        "current_shares": 40,
        "current_value": 6400.0,
        "sector": "Healthcare",
        "purchase_price": 160.00
      },
      {
        "symbol": "XOM",
        "current_shares": 60,
        "current_value": 6600.0,
        "sector": "Energy",
        "purchase_price": 110.00
      }
    ],
    "total_portfolio_value": 33750.0,
    "available_cash": 5000.0,
    "risk_profile": {
      "risk_score": 3,
      "risk_tolerance": "moderate",
      "time_horizon": "5-10 years",
      "liquidity_needs": "20-40% accessible",
      "sector_preferences": ["Technology", "Healthcare"],
      "sector_restrictions": []
    },
    "market_insights": {
      "title": "Tech Stocks Rally Continues as AI Investment Surges",
      "content": "Technology sector showing unprecedented growth...",
      "summary": "Strong bullish sentiment in tech sector with AI driving growth",
      "financial_keywords": ["tech", "AI", "growth", "investment"],
      "market_sentiment": "BULLISH",
      "sectors_mentioned": ["Technology"],
      "tickers_mentioned": ["AAPL", "MSFT", "NVDA"],
      "key_insights": [
        "AI investments driving tech sector growth",
        "Positive earnings outlook for tech companies",
        "Institutional buying increasing in tech stocks"
      ],
      "confidence_score": 85.0
    },
    "rebalance_type": "market_insight"
  }'
```

### Expected Response
```json
{
  "rebalancing_actions": [
    {
      "symbol": "AAPL",
      "action": "BUY",
      "current_shares": 50,
      "target_shares": 65,
      "shares_to_trade": 15,
      "dollar_amount": 2850.0,
      "current_allocation": 28.1,
      "target_allocation": 32.5,
      "rationale": "BUY AAPL based on market insights: AAPL directly mentioned in market analysis; Technology sector highlighted in insights; Bullish sentiment supports increased allocation",
      "priority": "HIGH"
    },
    {
      "symbol": "MSFT",
      "action": "BUY", 
      "current_shares": 30,
      "target_shares": 35,
      "shares_to_trade": 5,
      "dollar_amount": 1875.0,
      "current_allocation": 33.3,
      "target_allocation": 36.8,
      "rationale": "BUY MSFT based on market insights: Technology sector highlighted in insights; Bullish sentiment supports increased allocation",
      "priority": "HIGH"
    },
    {
      "symbol": "XOM",
      "action": "SELL",
      "current_shares": 60,
      "target_shares": 45,
      "shares_to_trade": 15,
      "dollar_amount": 1650.0,
      "current_allocation": 19.6,
      "target_allocation": 15.0,
      "rationale": "SELL XOM based on market insights: Reduce energy exposure to increase tech allocation",
      "priority": "MEDIUM"
    }
  ],
  "adjusted_risk_profile": {
    "original_risk_score": 3,
    "adjusted_risk_score": 4,
    "adjustment_reason": "Increased risk tolerance due to strong bullish sentiment",
    "sector_allocation_changes": {
      "Technology": 5.0
    },
    "time_horizon_impact": "Maintained 5-10 years with bullish outlook"
  },
  "market_insight_summary": "Market sentiment: BULLISH; Confidence: 85%; Key sectors: Technology; Mentioned tickers: AAPL, MSFT, NVDA",
  "projected_portfolio_value": 38750.0,
  "risk_metrics": {
    "portfolio_volatility": 20.0,
    "rebalancing_risk": 6.0,
    "implementation_risk": 4.8,
    "diversification_score": 85
  },
  "implementation_timeline": "Implement high priority actions within 24 hours, others within 1 week",
  "monitoring_recommendations": [
    "Monitor portfolio performance daily for first week after rebalancing",
    "Track bullish sentiment indicators",
    "Review sector allocation weekly for first month",
    "Monitor AAPL, MSFT, NVDA closely"
  ],
  "confidence_score": 68.0,
  "timestamp": "2024-12-01T15:35:22Z"
}
```

---

## 🔄 API 3: Combined Workflow (URL-to-Portfolio-Action)

### Purpose
One-step API that combines URL parsing and portfolio rebalancing for automated workflows.

### Endpoint
```
POST /api/url-to-portfolio-action
```

### Complete Workflow Example
```bash
curl -X POST "https://finagentcur.onrender.com/api/url-to-portfolio-action" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://www.reuters.com/business/energy/oil-prices-surge-geopolitical-tensions-2024-12-01/",
    "portfolio_holdings": [
      {
        "symbol": "XOM",
        "current_shares": 100,
        "current_value": 11000.0,
        "sector": "Energy",
        "purchase_price": 105.00
      },
      {
        "symbol": "CVX",
        "current_shares": 75,
        "current_value": 10500.0,
        "sector": "Energy",
        "purchase_price": 140.00
      },
      {
        "symbol": "AAPL",
        "current_shares": 50,
        "current_value": 9500.0,
        "sector": "Technology",
        "purchase_price": 180.00
      }
    ],
    "total_portfolio_value": 31000.0,
    "available_cash": 3000.0,
    "risk_profile": {
      "risk_score": 3,
      "risk_tolerance": "moderate",
      "time_horizon": "3-5 years",
      "liquidity_needs": "moderate",
      "sector_preferences": ["Energy", "Technology"],
      "sector_restrictions": []
    }
  }'
```

### Expected Response
```json
{
  "status": "success",
  "url_analysis": {
    "url": "https://www.reuters.com/business/energy/oil-prices-surge-geopolitical-tensions-2024-12-01/",
    "is_financial": true,
    "financial_content": {
      "title": "Oil Prices Surge on Geopolitical Tensions",
      "market_sentiment": "BULLISH",
      "sectors_mentioned": ["Energy"],
      "tickers_mentioned": ["XOM", "CVX", "COP"],
      "confidence_score": 88.5
    }
  },
  "portfolio_action": {
    "rebalancing_actions": [
      {
        "symbol": "XOM",
        "action": "BUY",
        "shares_to_trade": 15,
        "dollar_amount": 1650.0,
        "priority": "HIGH",
        "rationale": "XOM directly mentioned; Energy sector bullish outlook"
      },
      {
        "symbol": "CVX", 
        "action": "BUY",
        "shares_to_trade": 8,
        "dollar_amount": 1120.0,
        "priority": "HIGH",
        "rationale": "CVX mentioned; Strong energy sector momentum"
      }
    ],
    "confidence_score": 70.8
  },
  "summary": {
    "market_sentiment": "BULLISH",
    "confidence": 88.5,
    "actions_recommended": 2,
    "high_priority_actions": 2,
    "implementation_timeline": "Implement within 1-2 trading days"
  },
  "timestamp": "2024-12-01T15:40:15Z"
}
```

---

## 🔧 Python SDK Examples

### Python URL Parser Example
```python
import httpx
import asyncio

async def parse_financial_url(url: str):
    """Parse a URL for financial content"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://finagentcur.onrender.com/api/parse-url",
            json={"url": url, "max_content_length": 50000}
        )
        
        if response.status_code == 200:
            result = response.json()
            if result["is_financial"]:
                content = result["financial_content"]
                print(f"✅ Financial Content Found")
                print(f"Title: {content['title']}")
                print(f"Sentiment: {content['market_sentiment']}")
                print(f"Confidence: {content['confidence_score']:.1f}%")
                print(f"Sectors: {', '.join(content['sectors_mentioned'])}")
                print(f"Tickers: {', '.join(content['tickers_mentioned'])}")
                return result
            else:
                print(f"❌ Not financial content: {result['error_message']}")
                return None
        else:
            print(f"Error: {response.status_code}")
            return None

# Usage
url = "https://www.marketwatch.com/story/tech-earnings-outlook-strong"
result = asyncio.run(parse_financial_url(url))
```

### Python Portfolio Rebalancer Example
```python
async def rebalance_with_insights(portfolio, insights):
    """Rebalance portfolio using market insights"""
    rebalance_request = {
        "portfolio_holdings": portfolio,
        "total_portfolio_value": sum(h["current_value"] for h in portfolio),
        "available_cash": 5000.0,
        "risk_profile": {
            "risk_score": 3,
            "risk_tolerance": "moderate",
            "time_horizon": "5-10 years",
            "liquidity_needs": "20-40% accessible",
            "sector_preferences": ["Technology", "Healthcare"],
            "sector_restrictions": []
        },
        "market_insights": insights["financial_content"]
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://finagentcur.onrender.com/api/rebalance-portfolio",
            json=rebalance_request
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"🤖 AI Portfolio Rebalancing Complete")
            print(f"Actions Recommended: {len(result['rebalancing_actions'])}")
            
            for action in result["rebalancing_actions"]:
                print(f"  {action['action']} {action['symbol']}: "
                      f"{action['shares_to_trade']} shares "
                      f"(${action['dollar_amount']:,.0f}) - {action['priority']}")
                print(f"    Reason: {action['rationale']}")
            
            return result
        else:
            print(f"Error: {response.status_code}")
            return None

# Usage with parsed insights
portfolio = [
    {"symbol": "AAPL", "current_shares": 50, "current_value": 9500.0, "sector": "Technology"},
    {"symbol": "JNJ", "current_shares": 40, "current_value": 6400.0, "sector": "Healthcare"}
]

if result:  # From previous URL parsing
    rebalance_result = asyncio.run(rebalance_with_insights(portfolio, result))
```

### Complete Automated Workflow
```python
async def automated_url_to_action(url: str, portfolio: list):
    """Complete automated workflow from URL to portfolio action"""
    async with httpx.AsyncClient() as client:
        # One-step API call
        response = await client.post(
            "https://finagentcur.onrender.com/api/url-to-portfolio-action",
            json={
                "url": url,
                "portfolio_holdings": portfolio,
                "total_portfolio_value": sum(h["current_value"] for h in portfolio),
                "available_cash": 5000.0
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            if result["status"] == "success":
                summary = result["summary"]
                print(f"🚀 Automated Analysis Complete")
                print(f"Market Sentiment: {summary['market_sentiment']}")
                print(f"Confidence: {summary['confidence']:.1f}%")
                print(f"Actions: {summary['actions_recommended']} "
                      f"({summary['high_priority_actions']} high priority)")
                print(f"Timeline: {summary['implementation_timeline']}")
                
                return result
            else:
                print(f"❌ {result['message']}")
                return None

# Usage
url = "https://finance.yahoo.com/news/fed-rate-decision-impacts-markets"
portfolio = [
    {"symbol": "SPY", "current_shares": 100, "current_value": 45000.0, "sector": "Broad Market"},
    {"symbol": "QQQ", "current_shares": 50, "current_value": 18000.0, "sector": "Technology"}
]

result = asyncio.run(automated_url_to_action(url, portfolio))
```

---

## 🛠️ Error Handling

### Common Error Responses

#### Invalid URL
```json
{
  "detail": "Invalid URL provided"
}
```

#### Non-Financial Content
```json
{
  "url": "https://example.com/cooking-recipe",
  "is_financial": false,
  "error_message": "Content is not financial/market related",
  "metadata": {
    "status": "not_financial",
    "confidence": 25,
    "detected_keywords": ["market", "price"]
  }
}
```

#### Missing Market Insights
```json
{
  "detail": "Market insights are required. Please use /api/parse-url first to get financial content."
}
```

#### Unsupported Content Type
```json
{
  "url": "https://example.com/document.pdf",
  "is_financial": false,
  "error_message": "Failed to fetch content from URL. This may be due to unsupported content type (PDF, image, video) or network issues.",
  "metadata": {
    "status": "failed",
    "reason": "content_fetch_error"
  }
}
```

---

## 📊 Use Cases

### 1. News-Driven Trading
```bash
# Parse breaking financial news
curl -X POST "https://finagentcur.onrender.com/api/parse-url" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.cnbc.com/breaking-market-news"}'

# Use insights to rebalance immediately
# (Use financial_content from above response in portfolio rebalancer)
```

### 2. Earnings Report Analysis
```bash
# Parse company earnings report
curl -X POST "https://finagentcur.onrender.com/api/parse-url" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://investor.apple.com/quarterly-results"}'
```

### 3. Automated Portfolio Management
```bash
# Daily market analysis workflow
curl -X POST "https://finagentcur.onrender.com/api/url-to-portfolio-action" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.bloomberg.com/markets/daily-market-wrap", ...}'
```

---

## 🎯 Best Practices

1. **URL Selection**: Use reputable financial news sources (Bloomberg, Reuters, CNBC, MarketWatch)
2. **Confidence Thresholds**: Only act on insights with confidence > 70%
3. **Portfolio Validation**: Always validate portfolio data before rebalancing
4. **Risk Management**: Review risk profile adjustments carefully
5. **Implementation**: Follow recommended implementation timelines
6. **Monitoring**: Use provided monitoring recommendations after rebalancing

---

## 🔗 Integration with Existing APIs

These new APIs work seamlessly with existing FinAgent endpoints:

```bash
# 1. Parse market news
curl -X POST ".../api/parse-url" -d '{"url": "..."}'

# 2. Get current market data for portfolio
curl -X POST ".../api/market-data" -d '{"symbols": ["AAPL", "MSFT"]}'

# 3. Rebalance with insights
curl -X POST ".../api/rebalance-portfolio" -d '{...}'

# 4. Analyze final portfolio
curl -X POST ".../api/analyze-portfolio" -d '{...}'
```

This creates a complete end-to-end workflow from market news to portfolio action! 