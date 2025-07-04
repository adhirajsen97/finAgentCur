# FinAgent - Enhanced AI Investment System

🚀 **Enhanced Simple Version** - All the power, none of the complexity!

## 🎯 What's Included

### ✅ **RESTORED FEATURES**
- **🤖 Multiple AI Agents**: Data Analyst, Risk Analyst, Trading Analyst
- **📊 Advanced Risk Metrics**: Sharpe ratio, volatility, expected returns
- **⚖️ Compliance Framework**: SEC/FINRA-style validation and disclosures
- **📈 Enhanced Market Data**: Technical analysis and sentiment
- **🎯 Strategy Performance**: Performance tracking and metrics
- **💾 Database Integration**: Supabase with enhanced schema

### 🎯 **STRAIGHT ARROW STRATEGY**
- **60% VTI** - Total Stock Market (U.S. Equities)
- **30% BNDX** - International Bonds (Fixed Income)
- **10% GSG** - Commodities (Inflation Protection)

## 🏗️ **Architecture**

```
Enhanced FinAgent API v1.6
├── 🤖 AI Agents
│   ├── Data Analyst (Market analysis)
│   ├── Risk Analyst (Risk assessment)
│   └── Trading Analyst (Technical analysis)
├── 📊 Portfolio Engine
│   ├── Drift Analysis
│   ├── Risk Metrics (Sharpe, volatility)
│   └── Rebalancing Recommendations
├── 📈 Market Data Service
│   ├── Alpha Vantage Integration
│   ├── Technical Analysis
│   └── Market Sentiment
├── ⚖️ Compliance Framework
│   ├── Risk Validation
│   ├── Regulatory Disclosures
│   └── Audit Trail
└── 💾 Database Layer
    ├── Portfolio Analytics
    ├── Market Data Cache
    ├── AI Analysis Log
    ├── Compliance Audit
    └── Strategy Performance
```

## 🚀 **Quick Start**

### 1. **Setup Environment**
```bash
# Clone and setup
git clone <your-repo>
cd finAgentCur

# Install dependencies
pip install -r requirements_enhanced.txt

# Setup environment variables
cp .env.example .env
# Edit .env with your API keys
```

### 2. **Database Setup**
```bash
# Apply enhanced schema to Supabase
psql -h <supabase-host> -U postgres -d postgres -f database_enhanced.sql
```

### 3. **Run the API**
```bash
# Development
python main_enhanced_complete.py

# Production
uvicorn main_enhanced_complete:app --host 0.0.0.0 --port 8000
```

### 4. **Test Everything**
```bash
# Run comprehensive test suite
python test_enhanced.py
```

## 📡 **API Endpoints**

### 🏠 **Core Endpoints**
- `GET /` - API info and features
- `GET /health` - Enhanced health check with feature status
- `GET /docs` - Interactive API documentation

### 📊 **Portfolio Analysis**
- `POST /api/analyze-portfolio` - Enhanced portfolio analysis with risk metrics
- `GET /api/portfolio-history` - Portfolio analysis history
- `GET /api/strategy-performance` - Straight Arrow strategy performance

### 📈 **Market Data**
- `POST /api/market-data` - Market quotes with technical analysis
- `GET /api/market-sentiment` - Market sentiment analysis

### 🤖 **AI Agents**
- `POST /api/agents/data-analyst` - Market data analysis
- `POST /api/agents/risk-analyst` - Risk assessment and management
- `POST /api/agents/trading-analyst` - Technical analysis and signals

### 🚀 **Unified Strategy API** ⭐ NEW!
- `POST /api/unified-strategy` - **Complete investment strategy with actionable trade orders**

### ⚖️ **Compliance**
- `GET /api/compliance/disclosures` - Regulatory disclosures

## 🧪 **Example Usage**

### **Portfolio Analysis**
```python
import httpx

# Analyze portfolio
response = httpx.post("http://localhost:8000/api/analyze-portfolio", json={
    "portfolio": {
        "VTI": 60000.0,
        "BNDX": 25000.0,
        "GSG": 15000.0
    },
    "total_value": 100000.0
})

analysis = response.json()["analysis"]
print(f"Expected Return: {analysis['portfolio_metrics']['expected_return']:.1%}")
print(f"Volatility: {analysis['portfolio_metrics']['volatility']:.1%}")
print(f"Sharpe Ratio: {analysis['portfolio_metrics']['sharpe_ratio']:.2f}")
print(f"Compliance: {analysis['compliance']['status']}")
```

### **AI Risk Analysis**
```python
# Get risk analysis
response = httpx.post("http://localhost:8000/api/agents/risk-analyst", json={
    "portfolio": {"VTI": 60000.0, "BNDX": 25000.0, "GSG": 15000.0},
    "total_value": 100000.0,
    "time_horizon": "Long Term"
})

risk_analysis = response.json()["analysis"]
print(f"Risk Assessment: {risk_analysis['analysis']}")
```

### **Market Data with Technical Analysis**
```python
# Get market data
response = httpx.post("http://localhost:8000/api/market-data", json={
    "symbols": ["VTI", "BNDX", "GSG"]
})

quotes = response.json()["quotes"]
for symbol, quote in quotes.items():
    tech = quote.get("technical_analysis", {})
    print(f"{symbol}: ${quote['price']:.2f} - {tech.get('trend', 'N/A')}")
```

### **🚀 Unified Investment Strategy** ⭐ NEW!
```python
# Get complete investment strategy with actionable trade orders
response = httpx.post("http://localhost:8000/api/unified-strategy", json={
    "portfolio": {"VTI": 50000.0, "BNDX": 30000.0, "GSG": 20000.0},
    "total_value": 100000.0,
    "available_cash": 10000.0,
    "time_horizon": "3 weeks",
    "risk_tolerance": "moderate",
    "investment_goals": ["rebalancing"]
})

strategy = response.json()["strategy"]
trade_orders = strategy["trade_orders"]

# Execute trades based on priority
for order in sorted(trade_orders, key=lambda x: {"HIGH": 3, "MEDIUM": 2, "LOW": 1}[x["priority"]], reverse=True):
    print(f"{order['action']} {order['quantity']} shares of {order['symbol']}")
    print(f"Amount: ${order['dollar_amount']:,.2f} at ${order['current_price']:.2f}")
    print(f"Priority: {order['priority']} - {order['reason']}")
    # Your trade execution logic here
```

## 🔧 **Configuration**

### **Environment Variables**
```bash
# Required
SUPABASE_URL=your_supabase_url
SUPABASE_ANON_KEY=your_supabase_key

# Optional (will use mock data if not provided)
OPENAI_API_KEY=your_openai_key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key

# Server
PORT=8000
```

### **Database Schema**
The enhanced schema includes:
- `portfolio_analytics` - Enhanced portfolio analysis with risk metrics
- `market_data_cache` - Market data with technical analysis
- `ai_analysis_log` - AI agent interaction logs
- `market_sentiment` - Market sentiment tracking
- `strategy_performance` - Strategy performance metrics
- `compliance_audit` - Compliance audit trail

## 📊 **Features Comparison**

| Feature | Simple Version | Enhanced Version |
|---------|---------------|------------------|
| Portfolio Analysis | ✅ Basic | ✅ Advanced with risk metrics |
| Market Data | ✅ Basic quotes | ✅ Technical analysis included |
| AI Analysis | ✅ Single agent | ✅ Multiple specialized agents |
| Risk Metrics | ❌ | ✅ Sharpe ratio, volatility |
| Compliance | ✅ Basic | ✅ Full framework |
| Database Schema | ✅ 2 tables | ✅ 6 tables with audit trail |
| Performance Tracking | ❌ | ✅ Strategy performance |
| Market Sentiment | ❌ | ✅ Sentiment analysis |
| **🚀 Unified Strategy API** | ❌ | ✅ **Complete trade-ready strategies** |
| **Actionable Trade Orders** | ❌ | ✅ **Specific buy/sell with quantities** |
| **Priority-based Execution** | ❌ | ✅ **HIGH/MEDIUM/LOW priorities** |

## 🧪 **Testing**

### **Run Test Suite**
```bash
# Make sure API is running first
python main_enhanced_complete.py &

# Run comprehensive tests
python test_enhanced.py
```

### **Test Coverage**
- ✅ Enhanced health check
- ✅ Portfolio analysis with risk metrics
- ✅ Market data with technical analysis
- ✅ Market sentiment
- ✅ All three AI agents
- ✅ Strategy performance
- ✅ Compliance disclosures
- ✅ Portfolio history

## 🚀 **Deployment**

### **Render.com**
```bash
# Deploy using render.yaml
git push origin main
# Render will automatically deploy using requirements_enhanced.txt
```

### **Manual Deployment**
```bash
# Install dependencies
pip install -r requirements_enhanced.txt

# Run with gunicorn
gunicorn main_enhanced_complete:app -w 4 -k uvicorn.workers.UnicornWorker
```

## 🔍 **Monitoring**

### **Health Check**
```bash
curl http://localhost:8000/health
```

### **Feature Status**
The health endpoint shows:
- Database connection status
- Market data source (Alpha Vantage or mock)
- AI service status (OpenAI or mock)
- Available agents
- Compliance status
- Risk metrics status

## 📈 **Performance**

### **Expected Metrics** (Straight Arrow Strategy)
- **Expected Annual Return**: ~7.8%
- **Expected Volatility**: ~11.2%
- **Sharpe Ratio**: ~0.52
- **Risk Level**: Moderate
- **Rebalancing**: Quarterly

## ⚖️ **Compliance**

### **Regulatory Disclosures**
- Educational purposes only
- Not personalized investment advice
- Past performance doesn't guarantee future results
- Consult registered investment advisor
- AI technology disclosure
- Risk warnings

### **Risk Management**
- Portfolio volatility monitoring
- Concentration risk checks
- Diversification validation
- Risk-adjusted return analysis

## 🤝 **Contributing**

1. Fork the repository
2. Create feature branch
3. Add tests for new features
4. Ensure all tests pass
5. Submit pull request

## 📄 **License**

MIT License - see LICENSE file for details.

## 🆘 **Support**

- 📚 Documentation: `/docs` endpoint
- 🧪 Test Suite: `python test_enhanced.py`
- 🔍 Health Check: `/health` endpoint
- 📊 API Docs: `/docs` and `/redoc`

---

## 🎉 **What You Get**

✅ **All the missing features are back!**
- Multiple AI agents for specialized analysis
- Advanced risk metrics (Sharpe ratio, volatility)
- Comprehensive compliance framework
- Enhanced market data with technical analysis
- Strategy performance tracking
- Full database integration with audit trails

✅ **But still simple!**
- Single Straight Arrow strategy (no complex user profiling)
- Fixed database schema (no user_id issues)
- Clean, focused codebase
- Easy to deploy and maintain

**This is the best of both worlds - all the advanced features you need, without the complexity you don't want!** 🚀 

## 📊 Market Data & Real-Time Quotes

### Single Ticker API
```bash
# POST method with JSON body
curl -X POST "http://localhost:8000/api/ticker" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL"}'

# GET method with URL parameter  
curl "http://localhost:8000/api/ticker/AAPL"
```

### 🚀 NEW: Bulk Ticker API
**Fetch multiple stock prices simultaneously with concurrent processing!**

```bash
# Fetch multiple symbols at once
curl -X POST "http://localhost:8000/api/ticker/bulk" \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
  }'
```

**Response Format:**
```json
{
  "success_count": 5,
  "error_count": 0,
  "total_requested": 5,
  "tickers": {
    "AAPL": {
      "symbol": "AAPL",
      "price": 201.00,
      "change": -0.56,
      "change_percent": "-0.28%",
      "high": 202.64,
      "low": 199.46,
      "previous_close": 201.56,
      "timestamp": "2025-06-27T09:44:59.854545",
      "source": "finnhub"
    },
    "MSFT": {
      "symbol": "MSFT", 
      "price": 497.45,
      "change": 5.18,
      "change_percent": "1.05%",
      "high": 498.04,
      "low": 492.81,
      "previous_close": 492.27,
      "timestamp": "2025-06-27T09:44:59.857483",
      "source": "finnhub"
    }
  },
  "errors": {},
  "timestamp": "2025-06-27T09:44:59.858698",
  "source": "finnhub"
}
```

**Features:**
- ⚡ **Concurrent Processing**: Fetches all symbols simultaneously for maximum speed
- 🛡️ **Smart Validation**: Automatically removes duplicates and validates symbol format
- 🚫 **Rate Limit Aware**: Respects Finnhub's 60 calls/minute limit
- 📊 **Comprehensive Stats**: Returns success/error counts and detailed metrics
- 🔄 **Graceful Error Handling**: Continues processing other symbols if some fail
- 🎯 **Flexible Limits**: Supports 1-20 symbols per request

**Python Example:**
```python
import httpx

async def fetch_bulk_tickers():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/api/ticker/bulk",
            json={"symbols": ["AAPL", "MSFT", "GOOGL", "TSLA"]}
        )
        
        data = response.json()
        print(f"✅ Fetched {data['success_count']}/{data['total_requested']} symbols")
        
        for symbol, ticker in data['tickers'].items():
            print(f"{symbol}: ${ticker['price']:.2f} ({ticker['change_percent']})")

# Run the example
import asyncio
asyncio.run(fetch_bulk_tickers())
```

**Use Cases:**
- 📈 **Portfolio Analysis**: Get prices for all holdings at once
- 🔍 **Market Screening**: Analyze multiple stocks simultaneously  
- 📊 **Dashboard Updates**: Bulk refresh of market data
- 🎯 **Strategy Backtesting**: Efficient data collection for analysis 