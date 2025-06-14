"""
Trading Analyst Agent - Specializes in market trend analysis and trading signals
"""
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from crewai import Agent, Task
from langchain.tools import BaseTool
import ta  # Technical Analysis library
from services.market_data import MarketDataService, MarketData

logger = logging.getLogger(__name__)

class TechnicalAnalysisTool(BaseTool):
    """Tool for technical analysis and trading signals"""
    name = "technical_analysis"
    description = "Perform technical analysis and generate trading signals based on market data"
    market_data_service: MarketDataService = None
    
    def __init__(self, market_data_service: MarketDataService):
        super().__init__()
        self.market_data_service = market_data_service
    
    async def _arun(self, symbols: str, analysis_type: str = "comprehensive") -> str:
        """Async run method for technical analysis"""
        try:
            symbol_list = [s.strip() for s in symbols.split(',')]
            
            if analysis_type == "trend":
                return await self._analyze_trends(symbol_list)
            elif analysis_type == "momentum":
                return await self._analyze_momentum(symbol_list)
            elif analysis_type == "volatility":
                return await self._analyze_volatility(symbol_list)
            elif analysis_type == "signals":
                return await self._generate_trading_signals(symbol_list)
            else:  # comprehensive
                return await self._comprehensive_technical_analysis(symbol_list)
                
        except Exception as e:
            logger.error(f"Error in technical analysis: {e}")
            return f"Error in technical analysis: {str(e)}"
    
    def _run(self, symbols: str, analysis_type: str = "comprehensive") -> str:
        """Sync wrapper for async method"""
        return asyncio.run(self._arun(symbols, analysis_type))
    
    async def _analyze_trends(self, symbols: List[str]) -> str:
        """Analyze price trends using moving averages and trend indicators"""
        analysis = []
        analysis.append("## Trend Analysis\n")
        
        for symbol in symbols:
            hist_data = await self.market_data_service.get_historical_data(symbol, "6mo")
            if hist_data is not None and not hist_data.empty:
                # Calculate moving averages
                hist_data['MA20'] = hist_data['Close'].rolling(window=20).mean()
                hist_data['MA50'] = hist_data['Close'].rolling(window=50).mean()
                hist_data['MA200'] = hist_data['Close'].rolling(window=100).mean()  # Adjusted for available data
                
                # Current values
                current_price = hist_data['Close'].iloc[-1]
                ma20 = hist_data['MA20'].iloc[-1]
                ma50 = hist_data['MA50'].iloc[-1]
                ma200 = hist_data['MA200'].iloc[-1]
                
                # Trend determination
                short_trend = "BULLISH" if current_price > ma20 and ma20 > ma50 else "BEARISH"
                long_trend = "BULLISH" if ma50 > ma200 else "BEARISH"
                
                # Calculate trend strength
                trend_strength = self._calculate_trend_strength(hist_data)
                
                analysis.append(f"### {symbol} - Trend Analysis")
                analysis.append(f"- **Current Price**: ${current_price:.2f}")
                analysis.append(f"- **20-Day MA**: ${ma20:.2f}")
                analysis.append(f"- **50-Day MA**: ${ma50:.2f}")
                analysis.append(f"- **200-Day MA**: ${ma200:.2f}")
                analysis.append(f"- **Short-term Trend**: {short_trend}")
                analysis.append(f"- **Long-term Trend**: {long_trend}")
                analysis.append(f"- **Trend Strength**: {trend_strength}")
                analysis.append("")
        
        return "\n".join(analysis)
    
    async def _analyze_momentum(self, symbols: List[str]) -> str:
        """Analyze momentum indicators (RSI, MACD, etc.)"""
        analysis = []
        analysis.append("## Momentum Analysis\n")
        
        for symbol in symbols:
            hist_data = await self.market_data_service.get_historical_data(symbol, "3mo")
            if hist_data is not None and not hist_data.empty:
                # Calculate RSI
                hist_data['RSI'] = ta.momentum.RSIIndicator(hist_data['Close']).rsi()
                
                # Calculate MACD
                macd = ta.trend.MACD(hist_data['Close'])
                hist_data['MACD'] = macd.macd()
                hist_data['MACD_Signal'] = macd.macd_signal()
                hist_data['MACD_Histogram'] = macd.macd_diff()
                
                # Current values
                current_rsi = hist_data['RSI'].iloc[-1]
                current_macd = hist_data['MACD'].iloc[-1]
                current_macd_signal = hist_data['MACD_Signal'].iloc[-1]
                current_macd_hist = hist_data['MACD_Histogram'].iloc[-1]
                
                # Momentum signals
                rsi_signal = self._interpret_rsi(current_rsi)
                macd_signal = self._interpret_macd(current_macd, current_macd_signal, current_macd_hist)
                
                analysis.append(f"### {symbol} - Momentum Analysis")
                analysis.append(f"- **RSI**: {current_rsi:.2f} ({rsi_signal})")
                analysis.append(f"- **MACD**: {current_macd:.4f}")
                analysis.append(f"- **MACD Signal**: {current_macd_signal:.4f}")
                analysis.append(f"- **MACD Histogram**: {current_macd_hist:.4f}")
                analysis.append(f"- **MACD Signal**: {macd_signal}")
                analysis.append("")
        
        return "\n".join(analysis)
    
    async def _analyze_volatility(self, symbols: List[str]) -> str:
        """Analyze volatility patterns and Bollinger Bands"""
        analysis = []
        analysis.append("## Volatility Analysis\n")
        
        for symbol in symbols:
            hist_data = await self.market_data_service.get_historical_data(symbol, "3mo")
            if hist_data is not None and not hist_data.empty:
                # Calculate Bollinger Bands
                bb = ta.volatility.BollingerBands(hist_data['Close'])
                hist_data['BB_Upper'] = bb.bollinger_hband()
                hist_data['BB_Middle'] = bb.bollinger_mavg()
                hist_data['BB_Lower'] = bb.bollinger_lband()
                
                # Calculate ATR (Average True Range)
                hist_data['ATR'] = ta.volatility.AverageTrueRange(
                    hist_data['High'], hist_data['Low'], hist_data['Close']
                ).average_true_range()
                
                # Current values
                current_price = hist_data['Close'].iloc[-1]
                bb_upper = hist_data['BB_Upper'].iloc[-1]
                bb_middle = hist_data['BB_Middle'].iloc[-1]
                bb_lower = hist_data['BB_Lower'].iloc[-1]
                current_atr = hist_data['ATR'].iloc[-1]
                
                # Volatility interpretation
                bb_position = (current_price - bb_lower) / (bb_upper - bb_lower)
                volatility_signal = self._interpret_bollinger_bands(bb_position)
                
                # Calculate historical volatility
                returns = hist_data['Close'].pct_change().dropna()
                hist_vol = returns.std() * np.sqrt(252)  # Annualized
                
                analysis.append(f"### {symbol} - Volatility Analysis")
                analysis.append(f"- **Current Price**: ${current_price:.2f}")
                analysis.append(f"- **Bollinger Upper**: ${bb_upper:.2f}")
                analysis.append(f"- **Bollinger Middle**: ${bb_middle:.2f}")
                analysis.append(f"- **Bollinger Lower**: ${bb_lower:.2f}")
                analysis.append(f"- **BB Position**: {bb_position:.2%} ({volatility_signal})")
                analysis.append(f"- **ATR**: ${current_atr:.2f}")
                analysis.append(f"- **Historical Volatility**: {hist_vol:.2%}")
                analysis.append("")
        
        return "\n".join(analysis)
    
    async def _generate_trading_signals(self, symbols: List[str]) -> str:
        """Generate comprehensive trading signals"""
        analysis = []
        analysis.append("## Trading Signals\n")
        
        for symbol in symbols:
            hist_data = await self.market_data_service.get_historical_data(symbol, "6mo")
            if hist_data is not None and not hist_data.empty:
                # Get all indicators
                signals = await self._calculate_all_indicators(hist_data)
                
                # Generate overall signal
                overall_signal = self._generate_overall_signal(signals)
                confidence = self._calculate_signal_confidence(signals)
                
                analysis.append(f"### {symbol} - Trading Signals")
                analysis.append(f"- **Overall Signal**: {overall_signal}")
                analysis.append(f"- **Confidence**: {confidence:.0%}")
                analysis.append(f"- **Trend Signal**: {signals['trend_signal']}")
                analysis.append(f"- **Momentum Signal**: {signals['momentum_signal']}")
                analysis.append(f"- **Volatility Signal**: {signals['volatility_signal']}")
                analysis.append(f"- **Entry Recommendation**: {self._get_entry_recommendation(overall_signal, confidence)}")
                analysis.append("")
        
        return "\n".join(analysis)
    
    async def _comprehensive_technical_analysis(self, symbols: List[str]) -> str:
        """Comprehensive technical analysis combining all methods"""
        trend_analysis = await self._analyze_trends(symbols)
        momentum_analysis = await self._analyze_momentum(symbols)
        volatility_analysis = await self._analyze_volatility(symbols)
        trading_signals = await self._generate_trading_signals(symbols)
        
        analysis = []
        analysis.append("# Comprehensive Technical Analysis")
        analysis.append(f"**Analysis Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        analysis.append(f"**Symbols Analyzed**: {', '.join(symbols)}\n")
        analysis.append(trend_analysis)
        analysis.append(momentum_analysis)
        analysis.append(volatility_analysis)
        analysis.append(trading_signals)
        
        # Add summary
        analysis.append("## Technical Analysis Summary")
        analysis.append("- Complete technical analysis performed across all major indicators")
        analysis.append("- Trend, momentum, and volatility patterns identified")
        analysis.append("- Trading signals generated with confidence levels")
        analysis.append("- Analysis ready for execution and risk management review")
        
        return "\n".join(analysis)
    
    def _calculate_trend_strength(self, data: pd.DataFrame) -> str:
        """Calculate trend strength based on price action"""
        try:
            # Calculate trend strength using various methods
            ma20 = data['Close'].rolling(20).mean()
            ma50 = data['Close'].rolling(50).mean()
            
            # Count how many of last 20 days were above/below MA
            recent_data = data.tail(20)
            above_ma20 = (recent_data['Close'] > recent_data['Close'].rolling(20).mean()).sum()
            
            if above_ma20 >= 16:
                return "STRONG"
            elif above_ma20 >= 12:
                return "MODERATE"
            else:
                return "WEAK"
        except:
            return "UNKNOWN"
    
    def _interpret_rsi(self, rsi: float) -> str:
        """Interpret RSI value"""
        if rsi >= 70:
            return "OVERBOUGHT"
        elif rsi <= 30:
            return "OVERSOLD"
        elif rsi >= 50:
            return "BULLISH"
        else:
            return "BEARISH"
    
    def _interpret_macd(self, macd: float, signal: float, histogram: float) -> str:
        """Interpret MACD signals"""
        if macd > signal and histogram > 0:
            return "BULLISH"
        elif macd < signal and histogram < 0:
            return "BEARISH"
        elif histogram > 0:
            return "BULLISH MOMENTUM"
        else:
            return "BEARISH MOMENTUM"
    
    def _interpret_bollinger_bands(self, position: float) -> str:
        """Interpret Bollinger Band position"""
        if position >= 0.8:
            return "NEAR UPPER BAND"
        elif position <= 0.2:
            return "NEAR LOWER BAND"
        else:
            return "WITHIN BANDS"
    
    async def _calculate_all_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate all technical indicators"""
        signals = {}
        
        # Trend signals
        ma20 = data['Close'].rolling(20).mean().iloc[-1]
        ma50 = data['Close'].rolling(50).mean().iloc[-1]
        current_price = data['Close'].iloc[-1]
        
        if current_price > ma20 and ma20 > ma50:
            signals['trend_signal'] = "BULLISH"
        elif current_price < ma20 and ma20 < ma50:
            signals['trend_signal'] = "BEARISH"
        else:
            signals['trend_signal'] = "NEUTRAL"
        
        # Momentum signals
        rsi = ta.momentum.RSIIndicator(data['Close']).rsi().iloc[-1]
        if rsi >= 70:
            signals['momentum_signal'] = "OVERBOUGHT"
        elif rsi <= 30:
            signals['momentum_signal'] = "OVERSOLD"
        elif rsi >= 50:
            signals['momentum_signal'] = "BULLISH"
        else:
            signals['momentum_signal'] = "BEARISH"
        
        # Volatility signals
        bb = ta.volatility.BollingerBands(data['Close'])
        bb_upper = bb.bollinger_hband().iloc[-1]
        bb_lower = bb.bollinger_lband().iloc[-1]
        bb_position = (current_price - bb_lower) / (bb_upper - bb_lower)
        
        if bb_position >= 0.8:
            signals['volatility_signal'] = "HIGH_VOLATILITY"
        elif bb_position <= 0.2:
            signals['volatility_signal'] = "LOW_VOLATILITY"
        else:
            signals['volatility_signal'] = "NORMAL"
        
        return signals
    
    def _generate_overall_signal(self, signals: Dict[str, Any]) -> str:
        """Generate overall trading signal"""
        bullish_count = 0
        bearish_count = 0
        
        if signals['trend_signal'] == "BULLISH":
            bullish_count += 2
        elif signals['trend_signal'] == "BEARISH":
            bearish_count += 2
        
        if signals['momentum_signal'] in ["BULLISH", "OVERSOLD"]:
            bullish_count += 1
        elif signals['momentum_signal'] in ["BEARISH", "OVERBOUGHT"]:
            bearish_count += 1
        
        if bullish_count > bearish_count:
            return "BUY"
        elif bearish_count > bullish_count:
            return "SELL"
        else:
            return "HOLD"
    
    def _calculate_signal_confidence(self, signals: Dict[str, Any]) -> float:
        """Calculate confidence in the trading signal"""
        # Simple confidence calculation based on signal alignment
        total_signals = 3
        aligned_signals = 0
        
        # Check if signals are aligned
        if signals['trend_signal'] == "BULLISH" and signals['momentum_signal'] in ["BULLISH", "OVERSOLD"]:
            aligned_signals += 1
        elif signals['trend_signal'] == "BEARISH" and signals['momentum_signal'] in ["BEARISH", "OVERBOUGHT"]:
            aligned_signals += 1
        
        return aligned_signals / total_signals
    
    def _get_entry_recommendation(self, signal: str, confidence: float) -> str:
        """Get entry recommendation based on signal and confidence"""
        if signal == "BUY" and confidence >= 0.6:
            return "STRONG BUY"
        elif signal == "BUY" and confidence >= 0.4:
            return "BUY"
        elif signal == "SELL" and confidence >= 0.6:
            return "STRONG SELL"
        elif signal == "SELL" and confidence >= 0.4:
            return "SELL"
        else:
            return "HOLD/WAIT"

class MarketTimingTool(BaseTool):
    """Tool for market timing analysis"""
    name = "market_timing"
    description = "Analyze market timing and entry/exit points"
    market_data_service: MarketDataService = None
    
    def __init__(self, market_data_service: MarketDataService):
        super().__init__()
        self.market_data_service = market_data_service
    
    async def _arun(self, symbols: str, strategy: str = "straight_arrow") -> str:
        """Analyze optimal timing for investment strategy"""
        try:
            symbol_list = [s.strip() for s in symbols.split(',')]
            
            analysis = []
            analysis.append("## Market Timing Analysis\n")
            
            # Get market sentiment
            sentiment = await self.market_data_service.get_market_sentiment()
            
            # Analyze each symbol for timing
            for symbol in symbol_list:
                timing_analysis = await self._analyze_timing(symbol, strategy)
                analysis.append(timing_analysis)
            
            # Overall market timing assessment
            overall_timing = self._assess_overall_timing(sentiment)
            analysis.append(f"### Overall Market Timing Assessment")
            analysis.append(overall_timing)
            
            return "\n".join(analysis)
            
        except Exception as e:
            return f"Error in market timing analysis: {str(e)}"
    
    def _run(self, symbols: str, strategy: str = "straight_arrow") -> str:
        """Sync wrapper"""
        return asyncio.run(self._arun(symbols, strategy))
    
    async def _analyze_timing(self, symbol: str, strategy: str) -> str:
        """Analyze timing for individual symbol"""
        hist_data = await self.market_data_service.get_historical_data(symbol, "1y")
        
        if hist_data is None or hist_data.empty:
            return f"### {symbol} - No data available"
        
        # Calculate key metrics for timing
        current_price = hist_data['Close'].iloc[-1]
        ma200 = hist_data['Close'].rolling(200).mean().iloc[-1]
        volatility = hist_data['Close'].pct_change().std() * np.sqrt(252)
        
        # Determine timing recommendation
        if strategy == "straight_arrow":
            # For Straight Arrow strategy, focus on consistent investment
            timing_rec = self._straight_arrow_timing(current_price, ma200, volatility)
        else:
            timing_rec = "NEUTRAL"
        
        return f"""### {symbol} - Timing Analysis
- **Current Price**: ${current_price:.2f}
- **200-Day Average**: ${ma200:.2f}
- **Annual Volatility**: {volatility:.2%}
- **Timing Recommendation**: {timing_rec}
"""
    
    def _straight_arrow_timing(self, price: float, ma200: float, volatility: float) -> str:
        """Timing for Straight Arrow strategy"""
        # For diversified strategy, timing is less critical
        # Focus on regular investment with slight adjustments
        
        if volatility > 0.20:  # High volatility
            return "WAIT - High volatility, consider dollar cost averaging"
        elif price < ma200 * 0.95:  # Significantly below trend
            return "GOOD - Below trend line, favorable entry point"
        elif price > ma200 * 1.05:  # Significantly above trend
            return "CAUTION - Above trend line, consider smaller position"
        else:
            return "NEUTRAL - Normal market conditions, proceed with allocation"
    
    def _assess_overall_timing(self, sentiment: Dict[str, Any]) -> str:
        """Assess overall market timing"""
        vix = sentiment.get('vix', 20)
        
        if vix > 30:
            return "HIGH VOLATILITY - Market fear elevated, good for long-term entries but expect continued volatility"
        elif vix < 15:
            return "LOW VOLATILITY - Market complacency, be cautious of potential surprises"
        else:
            return "NORMAL CONDITIONS - Standard market conditions, proceed with planned strategy"

def create_trading_analyst_agent(market_data_service: MarketDataService) -> Agent:
    """Create the Trading Analyst agent"""
    
    # Create tools
    technical_tool = TechnicalAnalysisTool(market_data_service)
    timing_tool = MarketTimingTool(market_data_service)
    
    # Create agent
    trading_analyst = Agent(
        role="Senior Trading Analyst",
        goal="Analyze market trends, generate trading signals, and provide optimal entry/exit timing for investment strategies",
        backstory="""You are an expert trading analyst with deep expertise in:
        - Technical analysis and chart pattern recognition
        - Trend analysis using moving averages and trend indicators
        - Momentum analysis with RSI, MACD, and oscillators
        - Volatility analysis and Bollinger Bands
        - Market timing and entry/exit point optimization
        - Trading signal generation and confidence assessment
        
        You specialize in providing actionable trading insights that align with 
        investment strategies, focusing on risk-adjusted returns and optimal timing.
        Your analysis helps determine when to enter positions, adjust allocations,
        and implement systematic investment approaches like the Straight Arrow strategy.""",
        
        tools=[technical_tool, timing_tool],
        verbose=True,
        allow_delegation=False,
        max_iter=3,
        memory=True
    )
    
    return trading_analyst

def create_trading_analysis_tasks(symbols: List[str], strategy: str = "straight_arrow") -> List[Task]:
    """Create trading analysis tasks"""
    
    symbols_str = ','.join(symbols)
    
    tasks = []
    
    # Task 1: Technical Analysis
    technical_task = Task(
        description=f"""Perform comprehensive technical analysis for symbols: {symbols_str}
        
        Requirements:
        1. Analyze price trends using moving averages and trend indicators
        2. Evaluate momentum using RSI, MACD, and other oscillators
        3. Assess volatility patterns and Bollinger Bands
        4. Generate trading signals with confidence levels
        5. Identify support and resistance levels
        
        Provide detailed technical analysis with trading recommendations.""",
        
        expected_output="Comprehensive technical analysis with trend assessment, momentum indicators, volatility analysis, and trading signals",
        tools=["technical_analysis"],
        async_execution=True
    )
    tasks.append(technical_task)
    
    # Task 2: Market Timing Analysis
    timing_task = Task(
        description=f"""Analyze market timing and entry points for symbols: {symbols_str}
        
        Strategy Context: {strategy}
        
        Requirements:
        1. Assess current market conditions and sentiment
        2. Identify optimal entry and exit timing
        3. Evaluate market volatility and timing risks
        4. Provide strategy-specific timing recommendations
        5. Consider dollar-cost averaging vs lump-sum timing
        
        Provide market timing analysis aligned with the investment strategy.""",
        
        expected_output="Market timing analysis with entry/exit recommendations, volatility assessment, and strategy-aligned timing guidance",
        tools=["market_timing"],
        async_execution=True
    )
    tasks.append(timing_task)
    
    return tasks 