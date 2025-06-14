# 🚀 FinAgent - Simplified AI Investment System

**Straight Arrow Strategy Implementation**

A clean, focused AI-powered investment analysis system using the proven Straight Arrow strategy (60% VTI, 30% BNDX, 10% GSG).

## ✨ Features

- **Straight Arrow Strategy**: Fixed allocation (60% VTI, 30% BNDX, 10% GSG)
- **Portfolio Analysis**: Drift analysis and rebalancing recommendations
- **Real-time Market Data**: Alpha Vantage integration with fallback to mock data
- **AI Analysis**: Educational investment guidance with OpenAI integration
- **Database Storage**: Portfolio history tracking with Supabase
- **Simple API**: Clean, focused endpoints

## 🏗️ Architecture

```
FinAgent Simplified
├── Straight Arrow Strategy (Fixed Allocation)
├── Market Data Service (Alpha Vantage + Mock)
├── AI Service (OpenAI + Mock)
├── Database (Supabase PostgreSQL)
└── FastAPI REST API
```

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
Supabase account (optional)
API Keys: OpenAI, Alpha Vantage (optional)
```

### 1. Install Dependencies
```bash
pip install -r requirements_simple.txt
```

### 2. Set Environment Variables (Optional)
```bash
export SUPABASE_URL="your-supabase-url"
export SUPABASE_ANON_KEY="your-supabase-key"
export OPENAI_API_KEY="your-openai-key"
export ALPHA_VANTAGE_API_KEY="your-alpha-vantage-key"
```

### 3. Set Up Database (Optional)
```bash
# Run in your Supabase SQL editor
cat database_simple.sql
```

### 4. Start Server
```bash
python main_simplified.py
# Server available at http://localhost:8000
# API docs at http://localhost:8000/docs
```

### 5. Test
```bash
python test_simple.py
```

## 📊 API Endpoints

### Health Check
```bash
GET /health
```

### Portfolio Analysis
```bash
POST /api/analyze-portfolio
{
  "portfolio": {
    "VTI": 25000,
    "BNDX": 20000,
    "GSG": 5000
  },
  "total_value": 50000
}
```

### Market Data
```bash
POST /api/market-data
{
  "symbols": ["VTI", "BNDX", "GSG"]
}
```

### AI Analysis
```bash
POST /api/agents/data-analyst
{
  "query": "What is the Straight Arrow strategy?",
  "symbols": ["VTI", "BNDX", "GSG"]
}
```

### Portfolio History
```bash
GET /api/portfolio-history
```

## 🎯 Straight Arrow Strategy

**Target Allocation:**
- **60% VTI** - Vanguard Total Stock Market ETF
- **30% BNDX** - Vanguard Total International Bond ETF  
- **10% GSG** - iShares GSCI Commodity-Indexed Trust

**Benefits:**
- Simple 3-fund portfolio
- Broad diversification
- Low costs
- Easy to rebalance
- Suitable for beginners

## 🔧 Development

### Local Testing
```bash
# Start server
python main_simplified.py

# Run tests
python test_simple.py

# Check health
curl http://localhost:8000/health
```

### Production Deployment
```bash
# Deploy to Render/Heroku
# Set environment variables
# Update database connection
```

## 📈 Example Usage

### Portfolio Analysis
```python
import requests

response = requests.post("http://localhost:8000/api/analyze-portfolio", json={
    "portfolio": {"VTI": 30000, "BNDX": 15000, "GSG": 5000},
    "total_value": 50000
})

analysis = response.json()["analysis"]
print(f"Strategy: {analysis['strategy']}")
print(f"Needs rebalancing: {analysis['risk_assessment']['needs_rebalancing']}")
```

### AI Guidance
```python
response = requests.post("http://localhost:8000/api/agents/data-analyst", json={
    "query": "Should I rebalance my portfolio?",
    "symbols": ["VTI", "BNDX", "GSG"]
})

ai_response = response.json()["analysis"]
print(ai_response["analysis"])
```

## 🛠️ Configuration

### With APIs (Recommended)
- Set `ALPHA_VANTAGE_API_KEY` for real market data
- Set `OPENAI_API_KEY` for AI analysis
- Set `SUPABASE_URL` and `SUPABASE_ANON_KEY` for database

### Without APIs (Mock Mode)
- System works with mock data
- No external dependencies
- Perfect for testing and development

## 📝 Files

- `main_simplified.py` - Main application
- `requirements_simple.txt` - Dependencies
- `database_simple.sql` - Database schema
- `test_simple.py` - Test suite
- `README.md` - This file

## 🚀 Deployment

### Render Deployment
1. Connect GitHub repository
2. Set environment variables
3. Deploy from `main_simplified.py`
4. Run database schema in Supabase

### Environment Variables
```
SUPABASE_URL=your-supabase-url
SUPABASE_ANON_KEY=your-supabase-key
OPENAI_API_KEY=your-openai-key
ALPHA_VANTAGE_API_KEY=your-alpha-vantage-key
PORT=8000
```

## 📊 Live Demo

Try the live API:
- Health: `GET https://your-app.onrender.com/health`
- Docs: `https://your-app.onrender.com/docs`

## 🤝 Contributing

1. Fork repository
2. Create feature branch
3. Make changes
4. Test thoroughly
5. Submit pull request

## 📝 License

MIT License - see LICENSE file

---

**FinAgent Simplified** - *Clean, focused investment analysis* 🎯 