# FinAgent Enhanced API - Deployment Guide

## 🚀 Render Deployment

### Fixed Issues:
1. **Missing requirements.txt** - Created from `requirements_enhanced.txt`
2. **Incorrect module reference** - Updated to use `main_enhanced_complete:app`
3. **Build configuration** - Updated `render.yaml` and `Dockerfile`

### Deployment Configuration:

#### render.yaml
```yaml
services:
  - type: web
    name: finagent-enhanced
    runtime: python
    plan: starter
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn main_enhanced_complete:app --host 0.0.0.0 --port $PORT --workers 1
    healthCheckPath: /health
    autoDeploy: true
```

#### requirements.txt
```
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
httpx==0.24.1
supabase==2.0.2
aiohttp==3.9.1
python-dotenv==1.0.0
python-multipart==0.0.6
```

### Environment Variables (Set in Render Dashboard):
- `SUPABASE_URL` (optional)
- `SUPABASE_ANON_KEY` (optional)
- `OPENAI_API_KEY` (optional)
- `ALPHA_VANTAGE_API_KEY` (optional)

### Deployment Features:
- ✅ Enhanced FinAgent API with all features
- ✅ 3 AI Agents (Data, Risk, Trading Analysts)
- ✅ Advanced risk metrics (Sharpe ratio, volatility)
- ✅ Compliance framework with audit trails
- ✅ Interactive workflow guide at `/workflow`
- ✅ JSON API guide at `/api/workflow-guide`
- ✅ Health check endpoint at `/health`
- ✅ Auto-scaling and monitoring

### Post-Deployment URLs:
- **API Root**: `https://your-app.onrender.com/`
- **Interactive Docs**: `https://your-app.onrender.com/docs`
- **Workflow Guide**: `https://your-app.onrender.com/workflow`
- **Health Check**: `https://your-app.onrender.com/health`

### Testing Deployment:
```bash
# Health check
curl https://your-app.onrender.com/health

# Get workflow guide
curl https://your-app.onrender.com/api/workflow-guide

# Test market data
curl -X POST https://your-app.onrender.com/api/market-data \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["VTI", "BNDX", "GSG"]}'
```

## 🐳 Docker Deployment

### Dockerfile
```dockerfile
FROM python:3.11-slim as builder
# ... (multi-stage build for optimization)
CMD ["sh", "-c", "uvicorn main_enhanced_complete:app --host 0.0.0.0 --port $PORT --workers 1"]
```

### Docker Commands:
```bash
# Build
docker build -t finagent-enhanced .

# Run locally
docker run -p 8000:8000 -e PORT=8000 finagent-enhanced

# Test
curl http://localhost:8000/health
```

## 📊 Monitoring

### Health Check Response:
```json
{
  "status": "healthy",
  "version": "1.6.0",
  "strategy": "Straight Arrow",
  "features": {
    "agents": ["data_analyst", "risk_analyst", "trading_analyst"],
    "compliance": "enabled",
    "risk_metrics": "enabled"
  }
}
```

### Key Endpoints:
- `/health` - System health and feature status
- `/workflow` - Interactive HTML workflow guide
- `/api/workflow-guide` - JSON workflow data
- `/docs` - OpenAPI documentation
- `/api/analyze-portfolio` - Portfolio analysis with risk metrics
- `/api/agents/*` - AI agent endpoints

## 🔧 Troubleshooting

### Common Issues:
1. **Build fails on requirements** - Check Python version compatibility
2. **Module not found** - Ensure `main_enhanced_complete.py` is in root
3. **Port binding** - Render sets `$PORT` environment variable
4. **Health check fails** - Check `/health` endpoint response

### Logs:
```bash
# Render logs
render logs --service finagent-enhanced

# Docker logs
docker logs <container-id>
```

## 🎯 Next Steps

1. **Deploy to Render** - Push changes and deploy
2. **Set Environment Variables** - Configure API keys in Render dashboard
3. **Test All Endpoints** - Use the workflow guide to test functionality
4. **Monitor Performance** - Check health endpoint and logs
5. **Scale if Needed** - Upgrade Render plan for higher traffic 