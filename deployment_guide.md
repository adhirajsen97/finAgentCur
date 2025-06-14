# FinAgent Production Deployment Guide

This guide provides comprehensive instructions for deploying and testing the FinAgent AI investing system in production environments.

## 📋 Prerequisites

### System Requirements
- **OS**: Linux (Ubuntu 20.04+ recommended) or macOS
- **Memory**: Minimum 4GB RAM (8GB+ recommended)
- **Storage**: 20GB+ available disk space
- **Network**: Stable internet connection for market data APIs

### Required Software
- Docker 20.10+
- Docker Compose 2.0+
- Git
- curl (for health checks)

### API Keys Required
- **OpenAI API Key** (required for AI agents)
- **Alpha Vantage API Key** (required for market data)
- **Anthropic API Key** (optional, for Claude models)

## 🚀 Deployment Options

### Option 1: Render Cloud Deployment (Recommended)

#### Step 1: Prepare Repository
```bash
# Clone the repository
git clone <your-repo-url>
cd finAgentCur

# Ensure all files are committed
git add .
git commit -m "Production deployment ready"
git push origin main
```

#### Step 2: Deploy to Render
1. **Connect Repository**:
   - Go to [Render Dashboard](https://dashboard.render.com)
   - Click "New" → "Blueprint"
   - Connect your GitHub repository
   - Select the `render.yaml` file

2. **Configure Environment Variables**:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key_here
   ```

3. **Deploy**:
   - Click "Apply" to start deployment
   - Monitor deployment logs
   - Wait for all services to be healthy

#### Step 3: Verify Deployment
```bash
# Check health endpoint
curl https://your-app-name.onrender.com/health

# Test API endpoints
curl https://your-app-name.onrender.com/docs
```

### Option 2: Docker Compose Deployment

#### Step 1: Environment Setup
```bash
# Copy environment template
cp env.example .env

# Edit environment variables
nano .env
```

Required environment variables:
```bash
ENVIRONMENT=production
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key_here
```

#### Step 2: Deploy with Script
```bash
# Make deployment script executable
chmod +x scripts/deploy.sh

# Run deployment
./scripts/deploy.sh
```

#### Step 3: Manual Deployment (Alternative)
```bash
# Build and start services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f app
```

### Option 3: Kubernetes Deployment

#### Step 1: Create Kubernetes Manifests
```bash
# Generate Kubernetes manifests
kubectl create deployment finagent --image=finagent:latest --dry-run=client -o yaml > k8s/deployment.yaml
kubectl create service clusterip finagent --tcp=8000:8000 --dry-run=client -o yaml > k8s/service.yaml
```

#### Step 2: Deploy to Kubernetes
```bash
# Apply manifests
kubectl apply -f k8s/

# Check deployment status
kubectl get pods
kubectl get services
```

## 🧪 Testing in Production

### Automated Testing

#### Run Full Test Suite
```bash
# Make test script executable
chmod +x scripts/test.sh

# Run all tests
./scripts/test.sh all
```

#### Run Specific Test Types
```bash
# Unit tests only
./scripts/test.sh unit

# API tests only
./scripts/test.sh api

# Load tests only
./scripts/test.sh load

# Security scan only
./scripts/test.sh security
```

### Manual Testing

#### 1. Health Check
```bash
curl -X GET "https://your-domain.com/health"
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0.0"
}
```

#### 2. Market Sentiment Analysis
```bash
curl -X GET "https://your-domain.com/api/market-sentiment"
```

#### 3. Portfolio Analysis
```bash
curl -X POST "https://your-domain.com/api/analyze-portfolio" \
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

#### 4. Strategy Performance
```bash
curl -X GET "https://your-domain.com/api/strategy-performance?period=1y"
```

#### 5. AI Agent Testing
```bash
# Data Analyst
curl -X POST "https://your-domain.com/api/agents/data-analyst" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Analyze current market conditions for VTI, BNDX, GSG",
    "symbols": ["VTI", "BNDX", "GSG"]
  }'

# Trading Analyst
curl -X POST "https://your-domain.com/api/agents/trading-analyst" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Provide trading signals for portfolio rebalancing",
    "portfolio": {"VTI": 0.65, "BNDX": 0.25, "GSG": 0.10}
  }'

# Risk Analyst
curl -X POST "https://your-domain.com/api/agents/risk-analyst" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Assess portfolio risk and compliance",
    "portfolio_value": 34250.0
  }'
```

### Load Testing

#### Using Apache Bench
```bash
# Test health endpoint
ab -n 1000 -c 10 https://your-domain.com/health

# Test API endpoint
ab -n 100 -c 5 -p portfolio.json -T application/json https://your-domain.com/api/analyze-portfolio
```

#### Using curl for Stress Testing
```bash
# Concurrent requests script
for i in {1..50}; do
  curl -X GET "https://your-domain.com/health" &
done
wait
```

## 📊 Monitoring and Observability

### Health Monitoring
```bash
# Continuous health check
watch -n 30 'curl -s https://your-domain.com/health | jq'
```

### Log Monitoring
```bash
# Docker Compose logs
docker-compose logs -f app

# Kubernetes logs
kubectl logs -f deployment/finagent
```

### Performance Metrics
- **Response Time**: < 2 seconds for API calls
- **Throughput**: 100+ requests/minute
- **Uptime**: 99.9% availability target
- **Memory Usage**: < 2GB per instance

## 🔧 Troubleshooting

### Common Issues

#### 1. API Key Errors
```bash
# Check environment variables
docker-compose exec app env | grep API_KEY

# Verify API key validity
curl "https://api.openai.com/v1/models" -H "Authorization: Bearer $OPENAI_API_KEY"
```

#### 2. Database Connection Issues
```bash
# Check database connectivity
docker-compose exec app python -c "
import asyncpg
import asyncio
import os

async def test_db():
    try:
        conn = await asyncpg.connect(os.getenv('DATABASE_URL'))
        result = await conn.fetchval('SELECT 1')
        print(f'Database connection: OK (result: {result})')
        await conn.close()
    except Exception as e:
        print(f'Database connection: FAILED ({e})')

asyncio.run(test_db())
"
```

#### 3. Redis Connection Issues
```bash
# Test Redis connectivity
docker-compose exec app python -c "
import redis
import os

try:
    r = redis.from_url(os.getenv('REDIS_URL'))
    r.ping()
    print('Redis connection: OK')
except Exception as e:
    print(f'Redis connection: FAILED ({e})')
"
```

#### 4. Market Data API Issues
```bash
# Test Alpha Vantage API
curl "https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=VTI&apikey=$ALPHA_VANTAGE_API_KEY"
```

### Performance Optimization

#### 1. Database Optimization
```sql
-- Check slow queries
SELECT query, mean_time, calls 
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 10;

-- Analyze table statistics
ANALYZE portfolio_positions;
ANALYZE trade_history;
```

#### 2. Redis Cache Optimization
```bash
# Check Redis memory usage
docker-compose exec redis redis-cli info memory

# Monitor cache hit rate
docker-compose exec redis redis-cli info stats | grep keyspace
```

#### 3. Application Scaling
```yaml
# docker-compose.yml scaling
services:
  app:
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G
```

## 🔒 Security Considerations

### 1. Environment Variables
- Never commit API keys to version control
- Use secure secret management (AWS Secrets Manager, etc.)
- Rotate API keys regularly

### 2. Network Security
- Use HTTPS in production
- Implement rate limiting
- Configure firewall rules

### 3. Database Security
- Use strong passwords
- Enable SSL connections
- Regular security updates

### 4. Container Security
- Use non-root user in containers
- Scan images for vulnerabilities
- Keep base images updated

## 📈 Scaling Considerations

### Horizontal Scaling
```bash
# Scale application instances
docker-compose up -d --scale app=3

# Load balancer configuration (nginx)
upstream finagent_backend {
    server app1:8000;
    server app2:8000;
    server app3:8000;
}
```

### Vertical Scaling
```yaml
# Increase resource limits
services:
  app:
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2'
```

### Database Scaling
- Read replicas for analytics queries
- Connection pooling
- Query optimization

## 🔄 Maintenance

### Regular Tasks
1. **Daily**: Monitor health checks and logs
2. **Weekly**: Review performance metrics
3. **Monthly**: Update dependencies and security patches
4. **Quarterly**: Review and rotate API keys

### Backup Strategy
```bash
# Database backup
docker-compose exec postgres pg_dump -U finagent finagent > backup_$(date +%Y%m%d).sql

# Redis backup
docker-compose exec redis redis-cli BGSAVE
```

### Update Procedure
```bash
# 1. Backup current state
./scripts/deploy.sh backup

# 2. Pull latest changes
git pull origin main

# 3. Deploy with rollback capability
./scripts/deploy.sh deploy

# 4. Verify deployment
./scripts/test.sh api

# 5. Rollback if needed
./scripts/deploy.sh rollback
```

## 📞 Support

For deployment issues or questions:
1. Check the troubleshooting section above
2. Review application logs
3. Consult the API documentation at `/docs`
4. Create an issue in the repository

---

**Last Updated**: January 2024  
**Version**: 1.0.0 