"""
AI Investing Agent - Main Application
Async Python script for AI investing agent with 4 specialized models
"""
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Import services
from services.market_data import MarketDataService, get_market_data
from services.strategy import create_strategy_service

# Import agents
from agents.data_analyst import create_data_analyst_agent, create_data_analysis_tasks
from agents.trading_analyst import create_trading_analyst_agent, create_trading_analysis_tasks
from agents.risk_analyst import create_risk_analyst_agent, create_risk_analysis_tasks

# CrewAI imports
from crewai import Crew, Process
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Pydantic models for API requests
class AnalysisRequest(BaseModel):
    symbols: List[str] = Field(default=["VTI", "BNDX", "GSG"], description="List of symbols to analyze")
    portfolio_values: Optional[List[float]] = Field(default=None, description="Current portfolio values")
    strategy: str = Field(default="straight_arrow", description="Investment strategy to follow")
    analysis_type: str = Field(default="comprehensive", description="Type of analysis to perform")

class PortfolioRequest(BaseModel):
    portfolio: Dict[str, float] = Field(description="Portfolio positions with symbol: value pairs")
    total_value: float = Field(description="Total portfolio value")

class AgentResponse(BaseModel):
    agent: str
    analysis: str
    timestamp: datetime
    execution_time: float

class InvestmentRecommendation(BaseModel):
    symbol: str
    action: str  # BUY, SELL, HOLD
    confidence: float
    reasoning: str
    risk_level: str

class AnalysisResponse(BaseModel):
    request_id: str
    status: str
    agents_completed: List[str]
    analysis_results: List[AgentResponse]
    investment_recommendations: List[InvestmentRecommendation]
    portfolio_assessment: Optional[Dict[str, Any]]
    execution_time: float
    timestamp: datetime

# Global services
market_data_service: MarketDataService = None
strategy_service = None
llm = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global market_data_service, strategy_service, llm
    
    # Startup
    logger.info("🚀 Starting AI Investing Agent...")
    
    # Initialize services
    market_data_service = await get_market_data()
    strategy_service = create_strategy_service(market_data_service)
    
    # Initialize LLM
    llm = ChatOpenAI(
        model="gpt-4",
        temperature=0.1,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    logger.info("✅ Services initialized successfully")
    yield
    
    # Shutdown
    logger.info("🔄 Shutting down AI Investing Agent...")

# Create FastAPI app
app = FastAPI(
    title="AI Investing Agent",
    description="Async AI investing agent with 4 specialized analysts following Straight Arrow strategy",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InvestmentAgentOrchestrator:
    """Orchestrates the 4 AI analyst agents"""
    
    def __init__(self, market_data_service: MarketDataService, strategy_service, llm):
        self.market_data_service = market_data_service
        self.strategy_service = strategy_service
        self.llm = llm
        
        # Create agents
        self.data_analyst = create_data_analyst_agent(market_data_service)
        self.trading_analyst = create_trading_analyst_agent(market_data_service)
        self.risk_analyst = create_risk_analyst_agent(market_data_service, strategy_service)
        
        # Execution analyst (simplified for now)
        self.execution_analyst = None  # Would implement actual execution logic
    
    async def run_comprehensive_analysis(self, request: AnalysisRequest) -> AnalysisResponse:
        """Run comprehensive analysis with all agents"""
        start_time = datetime.now()
        request_id = f"analysis_{start_time.strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"🎯 Starting comprehensive analysis {request_id} for symbols: {request.symbols}")
        
        try:
            # Create tasks for each agent
            data_tasks = create_data_analysis_tasks(request.symbols, request.portfolio_values)
            trading_tasks = create_trading_analysis_tasks(request.symbols, request.strategy)
            risk_tasks = create_risk_analysis_tasks(request.symbols, request.portfolio_values)
            
            # Create crews for parallel execution
            data_crew = Crew(
                agents=[self.data_analyst],
                tasks=data_tasks,
                process=Process.sequential,
                verbose=True
            )
            
            trading_crew = Crew(
                agents=[self.trading_analyst],
                tasks=trading_tasks,
                process=Process.sequential,
                verbose=True
            )
            
            risk_crew = Crew(
                agents=[self.risk_analyst],
                tasks=risk_tasks,
                process=Process.sequential,
                verbose=True
            )
            
            # Execute all crews in parallel
            logger.info("🔄 Executing agent analysis in parallel...")
            
            data_result = await asyncio.to_thread(data_crew.kickoff)
            trading_result = await asyncio.to_thread(trading_crew.kickoff)
            risk_result = await asyncio.to_thread(risk_crew.kickoff)
            
            # Compile results
            analysis_results = [
                AgentResponse(
                    agent="data_analyst",
                    analysis=str(data_result),
                    timestamp=datetime.now(),
                    execution_time=0.0
                ),
                AgentResponse(
                    agent="trading_analyst", 
                    analysis=str(trading_result),
                    timestamp=datetime.now(),
                    execution_time=0.0
                ),
                AgentResponse(
                    agent="risk_analyst",
                    analysis=str(risk_result),
                    timestamp=datetime.now(),
                    execution_time=0.0
                )
            ]
            
            # Generate investment recommendations
            recommendations = await self._generate_recommendations(request.symbols, analysis_results)
            
            # Portfolio assessment
            portfolio_assessment = None
            if request.portfolio_values:
                portfolio_dict = dict(zip(request.symbols, request.portfolio_values))
                total_value = sum(request.portfolio_values)
                portfolio_assessment = await self.strategy_service.analyze_portfolio(portfolio_dict, total_value)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            response = AnalysisResponse(
                request_id=request_id,
                status="completed",
                agents_completed=["data_analyst", "trading_analyst", "risk_analyst"],
                analysis_results=analysis_results,
                investment_recommendations=recommendations,
                portfolio_assessment=portfolio_assessment,
                execution_time=execution_time,
                timestamp=datetime.now()
            )
            
            logger.info(f"✅ Analysis {request_id} completed in {execution_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"❌ Error in analysis {request_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
    async def _generate_recommendations(self, symbols: List[str], analysis_results: List[AgentResponse]) -> List[InvestmentRecommendation]:
        """Generate investment recommendations based on agent analysis"""
        recommendations = []
        
        for symbol in symbols:
            # Simplified recommendation logic - would be more sophisticated in production
            action = "HOLD"
            confidence = 0.5
            reasoning = "Analysis in progress"
            risk_level = "MEDIUM"
            
            # Parse agent outputs for signals (simplified)
            for result in analysis_results:
                if "BUY" in result.analysis.upper():
                    action = "BUY"
                    confidence = min(confidence + 0.2, 1.0)
                elif "SELL" in result.analysis.upper():
                    action = "SELL"
                    confidence = min(confidence + 0.2, 1.0)
                
                if result.agent == "risk_analyst":
                    if "HIGH" in result.analysis.upper():
                        risk_level = "HIGH"
                    elif "LOW" in result.analysis.upper():
                        risk_level = "LOW"
            
            reasoning = f"Based on comprehensive analysis from data, trading, and risk analysts. Action: {action} with {confidence:.0%} confidence."
            
            recommendations.append(InvestmentRecommendation(
                symbol=symbol,
                action=action,
                confidence=confidence,
                reasoning=reasoning,
                risk_level=risk_level
            ))
        
        return recommendations

# Initialize orchestrator
orchestrator: InvestmentAgentOrchestrator = None

# API Endpoints
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "🎯 AI Investing Agent - Straight Arrow Strategy",
        "status": "active",
        "version": "1.0.0",
        "agents": ["data_analyst", "trading_analyst", "risk_analyst", "execution_analyst"],
        "strategy": "straight_arrow",
        "timestamp": datetime.now()
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    try:
        # Test market data service
        test_data = await market_data_service.get_real_time_quote("VTI")
        
        return {
            "status": "healthy",
            "services": {
                "market_data": "active" if test_data else "degraded",
                "strategy_service": "active",
                "agents": "active"
            },
            "timestamp": datetime.now()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now()
        }

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_investments(request: AnalysisRequest):
    """Run comprehensive investment analysis"""
    global orchestrator
    
    if orchestrator is None:
        orchestrator = InvestmentAgentOrchestrator(market_data_service, strategy_service, llm)
    
    return await orchestrator.run_comprehensive_analysis(request)

@app.post("/portfolio/analyze")
async def analyze_portfolio(request: PortfolioRequest):
    """Analyze existing portfolio against Straight Arrow strategy"""
    try:
        analysis = await strategy_service.analyze_portfolio(request.portfolio, request.total_value)
        return {
            "status": "success",
            "analysis": analysis,
            "timestamp": datetime.now()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/market/sentiment")
async def get_market_sentiment():
    """Get current market sentiment indicators"""
    try:
        sentiment = await market_data_service.get_market_sentiment()
        return {
            "status": "success",
            "sentiment": sentiment,
            "timestamp": datetime.now()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/strategy/performance")
async def get_strategy_performance(period: str = "1y"):
    """Get Straight Arrow strategy performance"""
    try:
        performance = await strategy_service.get_strategy_performance(period)
        return {
            "status": "success",
            "performance": performance,
            "timestamp": datetime.now()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/market/data/{symbol}")
async def get_symbol_data(symbol: str):
    """Get market data for specific symbol"""
    try:
        data = await market_data_service.get_real_time_quote(symbol)
        metrics = await market_data_service.get_financial_metrics(symbol)
        
        return {
            "status": "success",
            "symbol": symbol,
            "market_data": data.__dict__ if data else None,
            "financial_metrics": metrics.__dict__ if metrics else None,
            "timestamp": datetime.now()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    
    logger.info(f"🚀 Starting AI Investing Agent on port {port}")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    ) 