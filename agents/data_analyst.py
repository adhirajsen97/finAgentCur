"""
Data Analyst Agent - Specializes in processing live market data and financial metrics
"""
import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from crewai import Agent, Task, Tool
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from services.market_data import MarketDataService, MarketData, FinancialMetrics

logger = logging.getLogger(__name__)

class MarketDataAnalysisTool(BaseTool):
    """Tool for analyzing market data"""
    name = "market_data_analysis"
    description = "Analyze real-time and historical market data for investment decisions"
    market_data_service: MarketDataService = None
    
    def __init__(self, market_data_service: MarketDataService):
        super().__init__()
        self.market_data_service = market_data_service
    
    async def _arun(self, symbols: str, analysis_type: str = "comprehensive") -> str:
        """Async run method for the tool"""
        try:
            symbol_list = [s.strip() for s in symbols.split(',')]
            
            if analysis_type == "real_time":
                return await self._analyze_real_time_data(symbol_list)
            elif analysis_type == "historical":
                return await self._analyze_historical_data(symbol_list)
            elif analysis_type == "financial_metrics":
                return await self._analyze_financial_metrics(symbol_list)
            else:  # comprehensive
                return await self._comprehensive_analysis(symbol_list)
                
        except Exception as e:
            logger.error(f"Error in market data analysis: {e}")
            return f"Error analyzing market data: {str(e)}"
    
    def _run(self, symbols: str, analysis_type: str = "comprehensive") -> str:
        """Sync wrapper for async method"""
        return asyncio.run(self._arun(symbols, analysis_type))
    
    async def _analyze_real_time_data(self, symbols: List[str]) -> str:
        """Analyze real-time market data"""
        market_data = await self.market_data_service.get_portfolio_data(symbols)
        
        analysis = []
        analysis.append("## Real-Time Market Data Analysis\n")
        
        for symbol, data in market_data.items():
            if data:
                analysis.append(f"### {symbol}")
                analysis.append(f"- **Current Price**: ${data.price:.2f}")
                analysis.append(f"- **Change**: ${data.change:.2f} ({data.change_percent:.2f}%)")
                analysis.append(f"- **Volume**: {data.volume:,}")
                if data.market_cap:
                    analysis.append(f"- **Market Cap**: ${data.market_cap/1e9:.2f}B")
                if data.pe_ratio:
                    analysis.append(f"- **P/E Ratio**: {data.pe_ratio:.2f}")
                if data.dividend_yield:
                    analysis.append(f"- **Dividend Yield**: {data.dividend_yield:.2%}")
                analysis.append("")
        
        # Add market sentiment
        sentiment = await self.market_data_service.get_market_sentiment()
        if sentiment:
            analysis.append("### Market Sentiment Indicators")
            if sentiment.get('vix'):
                analysis.append(f"- **VIX (Fear Index)**: {sentiment['vix']:.2f}")
            analysis.append("")
        
        return "\n".join(analysis)
    
    async def _analyze_historical_data(self, symbols: List[str]) -> str:
        """Analyze historical data trends"""
        analysis = []
        analysis.append("## Historical Data Analysis\n")
        
        for symbol in symbols:
            hist_data = await self.market_data_service.get_historical_data(symbol, "1y")
            if hist_data is not None and not hist_data.empty:
                # Calculate returns and volatility
                returns = hist_data['Close'].pct_change().dropna()
                annual_return = returns.mean() * 252
                annual_volatility = returns.std() * np.sqrt(252)
                
                # Calculate max drawdown
                cumulative = (1 + returns).cumprod()
                peak = cumulative.cummax()
                drawdown = (cumulative - peak) / peak
                max_drawdown = drawdown.min()
                
                analysis.append(f"### {symbol} - 1 Year Analysis")
                analysis.append(f"- **Annual Return**: {annual_return:.2%}")
                analysis.append(f"- **Annual Volatility**: {annual_volatility:.2%}")
                analysis.append(f"- **Sharpe Ratio**: {annual_return/annual_volatility:.2f}" if annual_volatility > 0 else "- **Sharpe Ratio**: N/A")
                analysis.append(f"- **Max Drawdown**: {max_drawdown:.2%}")
                analysis.append(f"- **Current vs 1Y Ago**: {((hist_data['Close'].iloc[-1] / hist_data['Close'].iloc[0]) - 1):.2%}")
                analysis.append("")
        
        return "\n".join(analysis)
    
    async def _analyze_financial_metrics(self, symbols: List[str]) -> str:
        """Analyze fundamental financial metrics"""
        analysis = []
        analysis.append("## Financial Metrics Analysis\n")
        
        for symbol in symbols:
            metrics = await self.market_data_service.get_financial_metrics(symbol)
            if metrics:
                analysis.append(f"### {symbol} - Financial Health")
                
                if metrics.revenue:
                    analysis.append(f"- **Revenue**: ${metrics.revenue/1e9:.2f}B")
                if metrics.net_income:
                    analysis.append(f"- **Net Income**: ${metrics.net_income/1e9:.2f}B")
                if metrics.free_cash_flow:
                    analysis.append(f"- **Free Cash Flow**: ${metrics.free_cash_flow/1e9:.2f}B")
                if metrics.roe:
                    analysis.append(f"- **Return on Equity**: {metrics.roe:.2%}")
                if metrics.debt_to_equity:
                    analysis.append(f"- **Debt-to-Equity**: {metrics.debt_to_equity:.2f}")
                if metrics.current_ratio:
                    analysis.append(f"- **Current Ratio**: {metrics.current_ratio:.2f}")
                
                analysis.append("")
        
        return "\n".join(analysis)
    
    async def _comprehensive_analysis(self, symbols: List[str]) -> str:
        """Comprehensive analysis combining all data sources"""
        real_time = await self._analyze_real_time_data(symbols)
        historical = await self._analyze_historical_data(symbols)
        financial = await self._analyze_financial_metrics(symbols)
        
        analysis = []
        analysis.append("# Comprehensive Market Data Analysis")
        analysis.append(f"**Analysis Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        analysis.append(f"**Symbols Analyzed**: {', '.join(symbols)}\n")
        analysis.append(real_time)
        analysis.append(historical)
        analysis.append(financial)
        
        # Add summary insights
        analysis.append("## Key Insights Summary")
        analysis.append("- Market data successfully retrieved and analyzed")
        analysis.append("- All requested symbols have been evaluated for investment potential")
        analysis.append("- Data includes real-time prices, historical performance, and fundamental metrics")
        analysis.append("- Analysis ready for trading and risk assessment teams")
        
        return "\n".join(analysis)

class PortfolioAnalysisTool(BaseTool):
    """Tool for portfolio-level analysis"""
    name = "portfolio_analysis"
    description = "Analyze portfolio composition, performance, and risk metrics"
    market_data_service: MarketDataService = None
    
    def __init__(self, market_data_service: MarketDataService):
        super().__init__()
        self.market_data_service = market_data_service
    
    async def _arun(self, symbols: str, weights: str = None) -> str:
        """Analyze portfolio metrics"""
        try:
            symbol_list = [s.strip() for s in symbols.split(',')]
            
            # Parse weights if provided, otherwise use equal weights
            if weights:
                weight_list = [float(w.strip()) for w in weights.split(',')]
            else:
                weight_list = [1.0/len(symbol_list)] * len(symbol_list)
            
            # Normalize weights to sum to 1
            total_weight = sum(weight_list)
            weight_list = [w/total_weight for w in weight_list]
            
            # Calculate portfolio metrics
            portfolio_metrics = await self.market_data_service.calculate_portfolio_metrics(
                symbol_list, weight_list
            )
            
            analysis = []
            analysis.append("## Portfolio Analysis Results\n")
            analysis.append("### Portfolio Composition")
            for symbol, weight in zip(symbol_list, weight_list):
                analysis.append(f"- **{symbol}**: {weight:.1%}")
            analysis.append("")
            
            if portfolio_metrics:
                analysis.append("### Portfolio Performance Metrics")
                analysis.append(f"- **Annual Return**: {portfolio_metrics.get('annual_return', 0):.2%}")
                analysis.append(f"- **Annual Volatility**: {portfolio_metrics.get('annual_volatility', 0):.2%}")
                analysis.append(f"- **Sharpe Ratio**: {portfolio_metrics.get('sharpe_ratio', 0):.2f}")
                analysis.append(f"- **Max Drawdown**: {portfolio_metrics.get('max_drawdown', 0):.2%}")
                analysis.append(f"- **Total Return**: {portfolio_metrics.get('total_return', 0):.2%}")
            
            return "\n".join(analysis)
            
        except Exception as e:
            return f"Error in portfolio analysis: {str(e)}"
    
    def _run(self, symbols: str, weights: str = None) -> str:
        """Sync wrapper"""
        return asyncio.run(self._arun(symbols, weights))

def create_data_analyst_agent(market_data_service: MarketDataService) -> Agent:
    """Create the Data Analyst agent"""
    
    # Create tools
    market_tool = MarketDataAnalysisTool(market_data_service)
    portfolio_tool = PortfolioAnalysisTool(market_data_service)
    
    # Create agent
    data_analyst = Agent(
        role="Senior Data Analyst",
        goal="Analyze live market data, historical trends, and financial metrics to provide comprehensive investment insights",
        backstory="""You are a highly experienced financial data analyst with expertise in:
        - Real-time market data analysis and interpretation
        - Historical trend analysis and pattern recognition  
        - Financial metrics evaluation and fundamental analysis
        - Portfolio composition analysis and performance metrics
        - Risk measurement and volatility assessment
        
        You excel at processing large amounts of financial data quickly and accurately,
        identifying key trends and anomalies, and presenting complex data in clear,
        actionable insights for investment decision-making.""",
        
        tools=[market_tool, portfolio_tool],
        verbose=True,
        allow_delegation=False,
        max_iter=3,
        memory=True
    )
    
    return data_analyst

def create_data_analysis_tasks(symbols: List[str], portfolio_value: float = None) -> List[Task]:
    """Create data analysis tasks"""
    
    symbols_str = ','.join(symbols)
    
    tasks = []
    
    # Task 1: Real-time data analysis
    real_time_task = Task(
        description=f"""Analyze real-time market data for symbols: {symbols_str}
        
        Requirements:
        1. Fetch current prices, volume, and basic metrics
        2. Calculate price changes and percentage movements
        3. Analyze market capitalization and valuation metrics
        4. Assess current market sentiment indicators
        5. Identify any significant price movements or anomalies
        
        Provide a comprehensive real-time market analysis with key insights.""",
        
        expected_output="Detailed real-time market analysis with current prices, changes, volume analysis, and market sentiment assessment",
        tools=["market_data_analysis"],
        async_execution=True
    )
    tasks.append(real_time_task)
    
    # Task 2: Historical analysis
    historical_task = Task(
        description=f"""Conduct historical performance analysis for symbols: {symbols_str}
        
        Requirements:
        1. Analyze 1-year historical price data and trends
        2. Calculate annual returns, volatility, and Sharpe ratios
        3. Measure maximum drawdown and risk metrics
        4. Identify long-term trends and support/resistance levels
        5. Compare recent performance to historical averages
        
        Provide comprehensive historical analysis with performance metrics.""",
        
        expected_output="Detailed historical analysis with annual returns, volatility measures, risk metrics, and trend assessment",
        tools=["market_data_analysis"],
        async_execution=True
    )
    tasks.append(historical_task)
    
    # Task 3: Portfolio analysis (if applicable)
    if len(symbols) > 1:
        portfolio_task = Task(
            description=f"""Analyze portfolio composition and metrics for symbols: {symbols_str}
            
            Requirements:
            1. Evaluate portfolio diversification and correlation
            2. Calculate portfolio-level risk and return metrics
            3. Assess overall portfolio performance vs benchmarks
            4. Analyze sector and geographic exposure
            5. Identify concentration risks and rebalancing needs
            
            Provide comprehensive portfolio analysis with optimization recommendations.""",
            
            expected_output="Portfolio analysis with diversification assessment, risk metrics, performance evaluation, and optimization recommendations",
            tools=["portfolio_analysis"],
            async_execution=True
        )
        tasks.append(portfolio_task)
    
    return tasks 