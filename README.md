# 🤖 FinAgent - AI-Powered Investment System

An advanced async Python AI investing agent with 4 specialized models (data analyst, trading analyst, execution analyst, risk analyst) following the "Straight Arrow" investment strategy.

## 🚀 Features

### 🧠 AI Agent Architecture
- **Data Analyst**: Processes live market data and financial metrics
- **Trading Analyst**: Technical analysis and trading signals  
- **Risk Analyst**: Risk monitoring and compliance
- **Execution Analyst**: Trade execution coordination

### 📊 Investment Strategy
- **Straight Arrow Strategy**: Diversified Three-Fund Portfolio
  - VTI (Total Stock Market): 60%
  - BNDX (International Bonds): 30% 
  - GSG (Commodities): 10%
- **Risk Management**: Sharpe ratio > 0.5, volatility < 12%
- **Tax Loss Harvesting**: Automated optimization

### 🔧 Technical Stack
- **Framework**: FastAPI + CrewAI for multi-agent orchestration
- **Database**: PostgreSQL + Redis caching
- **Market Data**: Alpha Vantage, yfinance integration
- **AI Models**: OpenAI GPT-4, Anthropic Claude support
- **Deployment**: Docker, Render cloud-ready

## 🏃‍♂️ Quick Start

### Deploy to Render (Recommended)

1. **Fork this repository** to your GitHub account

2. **Get API Keys**:
   - [OpenAI API Key](https://platform.openai.com/api-keys)
   - [Alpha Vantage API Key](https://www.alphavantage.co/support/#api-key)

3. **Deploy to Render**:
   - Go to [Render Dashboard](https://dashboard.render.com)
   - Click "New" → "Blueprint"
   - Connect your GitHub repository
   - Select `render.yaml`
   - Add environment variables:
     ```
     OPENAI_API_KEY=your_openai_key_here
     ALPHA_VANTAGE_API_KEY=your_alphavantage_key_here
     ```
   - Click "Apply"

4. **Access your app** at `https://your-app-name.onrender.com`

### Local Development

```bash
# Clone repository
git clone <your-repo-url>
cd finAgentCur

# Set up environment
cp env.example .env
# Edit .env with your API keys

# Run with Docker Compose
docker-compose up -d

# Or run locally
pip install -r requirements.txt
uvicorn main:app --reload
```

## 📡 API Endpoints

### Core Analysis
- `GET /health` - Health check
- `POST /api/analyze-portfolio` - Portfolio analysis
- `GET /api/market-sentiment` - Market sentiment analysis
- `GET /api/strategy-performance` - Strategy performance metrics

### AI Agents
- `POST /api/agents/data-analyst` - Market data analysis
- `POST /api/agents/trading-analyst` - Trading signals
- `POST /api/agents/risk-analyst` - Risk assessment

### Documentation
- `GET /docs` - Interactive API documentation
- `GET /redoc` - Alternative API documentation

## 🧪 Testing

```bash
# Run all tests
./scripts/test.sh all

# Run specific test types
./scripts/test.sh unit        # Unit tests
./scripts/test.sh api         # API tests
./scripts/test.sh security    # Security scan
./scripts/test.sh coverage    # Coverage report
```

## 📊 Example Usage

### Portfolio Analysis
```bash
curl -X POST "https://your-app.onrender.com/api/analyze-portfolio" \
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

### Market Sentiment
```bash
curl -X GET "https://your-app.onrender.com/api/market-sentiment"
```

### AI Agent Query
```bash
curl -X POST "https://your-app.onrender.com/api/agents/data-analyst" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Analyze current market conditions for VTI, BNDX, GSG",
    "symbols": ["VTI", "BNDX", "GSG"]
  }'
```

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI App   │    │   AI Agents     │    │  Market Data    │
│                 │    │                 │    │                 │
│ • REST API      │◄──►│ • Data Analyst  │◄──►│ • Alpha Vantage │
│ • Async/Await   │    │ • Trading       │    │ • yfinance      │
│ • Pydantic      │    │ • Risk Analyst  │    │ • Real-time     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PostgreSQL    │    │     Redis       │    │   Strategy      │
│                 │    │                 │    │                 │
│ • Portfolio     │    │ • Caching       │    │ • Straight      │
│ • Trades        │    │ • Sessions      │    │   Arrow         │
│ • Analytics     │    │ • Rate Limiting │    │ • Rebalancing   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🔒 Security Features

- **API Key Management**: Secure environment variable handling
- **Rate Limiting**: Protection against abuse
- **Input Validation**: Pydantic model validation
- **Container Security**: Non-root user execution
- **SSL/TLS**: HTTPS encryption in production

## 📈 Performance

- **Response Time**: < 2 seconds for API calls
- **Throughput**: 100+ requests/minute
- **Caching**: Redis for market data optimization
- **Async Processing**: Non-blocking I/O operations

## 🛠️ Configuration

### Environment Variables
```bash
# Required
OPENAI_API_KEY=your_openai_key
ALPHA_VANTAGE_API_KEY=your_alphavantage_key

# Optional
ANTHROPIC_API_KEY=your_anthropic_key
DATABASE_URL=postgresql://...
REDIS_URL=redis://...
```

### Strategy Configuration
The Straight Arrow strategy can be customized in `services/strategy.py`:
- Asset allocation percentages
- Risk thresholds (Sharpe ratio, volatility)
- Rebalancing triggers

## 📚 Documentation

- **[Deployment Guide](deployment_guide.md)** - Complete deployment instructions
- **[API Documentation](https://your-app.onrender.com/docs)** - Interactive API docs
- **[Strategy Overview](investment_strategy.py)** - Investment strategy details

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- **Issues**: Create an issue on GitHub
- **Documentation**: Check `/docs` endpoint
- **Deployment**: See `deployment_guide.md`

---

**Built with ❤️ using FastAPI, CrewAI, and modern Python async patterns** 