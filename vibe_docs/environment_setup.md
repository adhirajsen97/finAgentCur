# Environment Setup

## Chosen Tech Stack
- **Language**: Python 3.11+
- **Framework**: CrewAI (for multi-agent orchestration)
- **API Framework**: FastAPI (async support)
- **Market Data**: Alpha Vantage + yfinance (cost-effective combination)
- **Database**: Redis (for caching) + PostgreSQL (for persistence)  
- **Deployment**: Render (with gunicorn + uvicorn)
- **AI Models**: OpenAI GPT-4 (primary) + Anthropic Claude (backup)

## Installation Commands

### 1. Python Environment Setup
```bash
# Create virtual environment
python -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip
```

### 2. Core Dependencies
```bash
# Install main packages
pip install crewai
pip install fastapi uvicorn
pip install yfinance alpha-vantage
pip install redis-py asyncpg
pip install python-dotenv
pip install pydantic
pip install httpx aiohttp
pip install numpy pandas
pip install langchain openai anthropic
```

### 3. Development Dependencies
```bash
pip install pytest pytest-asyncio
pip install black flake8
pip install pre-commit
```

## Environment Variables (.env)
```env
# AI API Keys
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here

# Market Data APIs
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here

# Database URLs
REDIS_URL=redis://localhost:6379
DATABASE_URL=postgresql://user:pass@localhost/finagent

# Render Deployment
PORT=8000
RENDER_EXTERNAL_URL=your_render_url

# Agent Configuration
MAX_CONCURRENT_AGENTS=4
AGENT_TIMEOUT=300
DATA_REFRESH_INTERVAL=60
```

## Development Server Commands
```bash
# Start development server
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Run agents in background mode
python -m agents.runner --background

# Run tests
pytest tests/ -v

# Format code
black . && flake8 .
```

## Production Deployment (Render)
```bash
# Install production dependencies
pip install gunicorn

# Start production server
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT
```

## Docker Setup (Optional)
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["gunicorn", "main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
```

## Project Structure
```
finAgentCur/
├── agents/
│   ├── data_analyst.py
│   ├── trading_analyst.py
│   ├── execution_analyst.py
│   ├── risk_analyst.py
│   └── orchestrator.py
├── services/
│   ├── market_data.py
│   ├── strategy.py
│   └── portfolio.py
├── main.py
├── requirements.txt
├── .env
└── vibe_docs/
```

## Verification Steps
1. ✅ Python 3.11+ installed
2. ✅ Virtual environment activated
3. ✅ Dependencies installed
4. ✅ Environment variables configured
5. ✅ Development server starts successfully
6. ✅ API endpoints respond correctly
7. ✅ Market data feeds working
8. ✅ Agents can communicate

## How to Run
1. Activate virtual environment
2. Set environment variables
3. Start Redis and PostgreSQL
4. Run: `uvicorn main:app --reload`
5. Access: http://localhost:8000
6. API docs: http://localhost:8000/docs

## Troubleshooting
- **Port conflicts**: Change PORT in .env
- **Redis connection**: Check Redis server is running
- **API rate limits**: Implement proper caching
- **Agent timeouts**: Adjust AGENT_TIMEOUT value

Last updated: 2025-01-06 