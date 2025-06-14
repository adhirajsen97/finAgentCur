"""
Risk Analyst Agent - Specializes in risk monitoring and portfolio exposure analysis
"""
import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from crewai import Agent, Task
from langchain.tools import BaseTool
from services.market_data import MarketDataService
from services.strategy import StraightArrowStrategyService

logger = logging.getLogger(__name__)

class RiskAssessmentTool(BaseTool):
    """Tool for comprehensive risk assessment"""
    name = "risk_assessment"
    description = "Assess portfolio and individual security risk metrics"
    market_data_service: MarketDataService = None
    strategy_service: StraightArrowStrategyService = None
    
    def __init__(self, market_data_service: MarketDataService, strategy_service: StraightArrowStrategyService):
        super().__init__()
        self.market_data_service = market_data_service
        self.strategy_service = strategy_service
    
    async def _arun(self, symbols: str, portfolio_values: str = None, assessment_type: str = "comprehensive") -> str:
        """Async run method for risk assessment"""
        try:
            symbol_list = [s.strip() for s in symbols.split(',')]
            
            # Parse portfolio values if provided
            portfolio_dict = {}
            if portfolio_values:
                value_list = [float(v.strip()) for v in portfolio_values.split(',')]
                portfolio_dict = dict(zip(symbol_list, value_list))
            
            if assessment_type == "volatility":
                return await self._assess_volatility_risk(symbol_list)
            elif assessment_type == "drawdown":
                return await self._assess_drawdown_risk(symbol_list)
            elif assessment_type == "correlation":
                return await self._assess_correlation_risk(symbol_list)
            elif assessment_type == "var":
                return await self._calculate_var(symbol_list, portfolio_dict)
            elif assessment_type == "strategy_compliance":
                return await self._assess_strategy_compliance(portfolio_dict)
            else:  # comprehensive
                return await self._comprehensive_risk_assessment(symbol_list, portfolio_dict)
                
        except Exception as e:
            logger.error(f"Error in risk assessment: {e}")
            return f"Error in risk assessment: {str(e)}"
    
    def _run(self, symbols: str, portfolio_values: str = None, assessment_type: str = "comprehensive") -> str:
        """Sync wrapper for async method"""
        return asyncio.run(self._arun(symbols, portfolio_values, assessment_type))
    
    async def _assess_volatility_risk(self, symbols: List[str]) -> str:
        """Assess volatility risk for each symbol and portfolio"""
        analysis = []
        analysis.append("## Volatility Risk Assessment\n")
        
        volatilities = []
        for symbol in symbols:
            hist_data = await self.market_data_service.get_historical_data(symbol, "1y")
            if hist_data is not None and not hist_data.empty:
                returns = hist_data['Close'].pct_change().dropna()
                annual_vol = returns.std() * np.sqrt(252)
                volatilities.append(annual_vol)
                
                # Risk level classification
                if annual_vol < 0.15:
                    risk_level = "LOW"
                elif annual_vol < 0.25:
                    risk_level = "MEDIUM"
                else:
                    risk_level = "HIGH"
                
                # Calculate VaR (95% confidence)
                var_95 = returns.quantile(0.05)
                
                analysis.append(f"### {symbol} - Volatility Risk")
                analysis.append(f"- **Annual Volatility**: {annual_vol:.2%}")
                analysis.append(f"- **Risk Level**: {risk_level}")
                analysis.append(f"- **Daily VaR (95%)**: {var_95:.2%}")
                analysis.append(f"- **Volatility Trend**: {self._get_volatility_trend(returns)}")
                analysis.append("")
        
        # Portfolio volatility estimate (simplified)
        if len(volatilities) > 1:
            avg_vol = np.mean(volatilities)
            analysis.append(f"### Portfolio Volatility Estimate")
            analysis.append(f"- **Average Volatility**: {avg_vol:.2%}")
            analysis.append(f"- **Diversification Benefit**: Actual portfolio volatility likely lower due to correlation effects")
            analysis.append("")
        
        return "\n".join(analysis)
    
    async def _assess_drawdown_risk(self, symbols: List[str]) -> str:
        """Assess maximum drawdown risk"""
        analysis = []
        analysis.append("## Drawdown Risk Assessment\n")
        
        for symbol in symbols:
            hist_data = await self.market_data_service.get_historical_data(symbol, "1y")
            if hist_data is not None and not hist_data.empty:
                # Calculate rolling maximum and drawdown
                rolling_max = hist_data['Close'].expanding().max()
                drawdown = (hist_data['Close'] - rolling_max) / rolling_max
                max_drawdown = drawdown.min()
                
                # Current drawdown
                current_drawdown = drawdown.iloc[-1]
                
                # Recovery analysis
                recovery_time = self._calculate_recovery_time(hist_data['Close'], drawdown)
                
                # Risk assessment
                if abs(max_drawdown) < 0.10:
                    drawdown_risk = "LOW"
                elif abs(max_drawdown) < 0.20:
                    drawdown_risk = "MEDIUM"
                else:
                    drawdown_risk = "HIGH"
                
                analysis.append(f"### {symbol} - Drawdown Risk")
                analysis.append(f"- **Max Drawdown (1Y)**: {max_drawdown:.2%}")
                analysis.append(f"- **Current Drawdown**: {current_drawdown:.2%}")
                analysis.append(f"- **Risk Level**: {drawdown_risk}")
                analysis.append(f"- **Avg Recovery Time**: {recovery_time} days")
                analysis.append("")
        
        return "\n".join(analysis)
    
    async def _assess_correlation_risk(self, symbols: List[str]) -> str:
        """Assess correlation risk between assets"""
        analysis = []
        analysis.append("## Correlation Risk Assessment\n")
        
        if len(symbols) < 2:
            analysis.append("Need at least 2 symbols for correlation analysis")
            return "\n".join(analysis)
        
        # Get historical data for all symbols
        returns_data = {}
        for symbol in symbols:
            hist_data = await self.market_data_service.get_historical_data(symbol, "1y")
            if hist_data is not None and not hist_data.empty:
                returns_data[symbol] = hist_data['Close'].pct_change().dropna()
        
        if len(returns_data) >= 2:
            # Create correlation matrix
            returns_df = pd.DataFrame(returns_data)
            correlation_matrix = returns_df.corr()
            
            analysis.append("### Correlation Matrix")
            for i, symbol1 in enumerate(symbols):
                for j, symbol2 in enumerate(symbols):
                    if i < j and symbol1 in correlation_matrix.index and symbol2 in correlation_matrix.columns:
                        corr = correlation_matrix.loc[symbol1, symbol2]
                        analysis.append(f"- **{symbol1} vs {symbol2}**: {corr:.3f}")
            analysis.append("")
            
            # Assess diversification benefit
            avg_correlation = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()
            
            if avg_correlation < 0.3:
                diversification = "EXCELLENT"
            elif avg_correlation < 0.6:
                diversification = "GOOD"
            elif avg_correlation < 0.8:
                diversification = "MODERATE"
            else:
                diversification = "POOR"
            
            analysis.append(f"### Diversification Assessment")
            analysis.append(f"- **Average Correlation**: {avg_correlation:.3f}")
            analysis.append(f"- **Diversification Benefit**: {diversification}")
            analysis.append("")
        
        return "\n".join(analysis)
    
    async def _calculate_var(self, symbols: List[str], portfolio_dict: Dict[str, float]) -> str:
        """Calculate Value at Risk (VaR) metrics"""
        analysis = []
        analysis.append("## Value at Risk (VaR) Analysis\n")
        
        if not portfolio_dict:
            analysis.append("Portfolio values needed for VaR calculation")
            return "\n".join(analysis)
        
        total_value = sum(portfolio_dict.values())
        
        # Calculate portfolio returns
        portfolio_returns = []
        weights = {symbol: value/total_value for symbol, value in portfolio_dict.items()}
        
        # Get historical data and calculate weighted returns
        returns_data = {}
        for symbol in symbols:
            hist_data = await self.market_data_service.get_historical_data(symbol, "1y")
            if hist_data is not None and not hist_data.empty:
                returns_data[symbol] = hist_data['Close'].pct_change().dropna()
        
        if returns_data:
            # Align dates and calculate portfolio returns
            returns_df = pd.DataFrame(returns_data).dropna()
            portfolio_returns = (returns_df * list(weights.values())).sum(axis=1)
            
            # Calculate VaR at different confidence levels
            var_95 = portfolio_returns.quantile(0.05)
            var_99 = portfolio_returns.quantile(0.01)
            
            # Expected Shortfall (Conditional VaR)
            es_95 = portfolio_returns[portfolio_returns <= var_95].mean()
            es_99 = portfolio_returns[portfolio_returns <= var_99].mean()
            
            # Convert to dollar amounts
            var_95_dollar = var_95 * total_value
            var_99_dollar = var_99 * total_value
            es_95_dollar = es_95 * total_value
            es_99_dollar = es_99 * total_value
            
            analysis.append(f"### Portfolio VaR Analysis")
            analysis.append(f"- **Portfolio Value**: ${total_value:,.2f}")
            analysis.append(f"- **Daily VaR (95%)**: {var_95:.2%} (${var_95_dollar:,.2f})")
            analysis.append(f"- **Daily VaR (99%)**: {var_99:.2%} (${var_99_dollar:,.2f})")
            analysis.append(f"- **Expected Shortfall (95%)**: {es_95:.2%} (${es_95_dollar:,.2f})")
            analysis.append(f"- **Expected Shortfall (99%)**: {es_99:.2%} (${es_99_dollar:,.2f})")
            analysis.append("")
            
            # Risk interpretation
            if abs(var_95) < 0.02:
                risk_assessment = "LOW RISK - Conservative portfolio with limited downside"
            elif abs(var_95) < 0.04:
                risk_assessment = "MODERATE RISK - Balanced risk-return profile"
            else:
                risk_assessment = "HIGH RISK - Significant potential for large losses"
            
            analysis.append(f"### Risk Assessment")
            analysis.append(f"- **Overall Risk Level**: {risk_assessment}")
            analysis.append("")
        
        return "\n".join(analysis)
    
    async def _assess_strategy_compliance(self, portfolio_dict: Dict[str, float]) -> str:
        """Assess compliance with Straight Arrow strategy risk controls"""
        analysis = []
        analysis.append("## Strategy Risk Compliance Assessment\n")
        
        if not portfolio_dict:
            analysis.append("Portfolio values needed for strategy compliance assessment")
            return "\n".join(analysis)
        
        total_value = sum(portfolio_dict.values())
        
        # Analyze portfolio against Straight Arrow strategy
        strategy_analysis = await self.strategy_service.analyze_portfolio(portfolio_dict, total_value)
        
        if "error" in strategy_analysis:
            analysis.append(f"Error in strategy analysis: {strategy_analysis['error']}")
            return "\n".join(analysis)
        
        risk_assessment = strategy_analysis.get('risk_assessment', {})
        portfolio_metrics = strategy_analysis.get('portfolio_metrics', {})
        
        analysis.append(f"### Straight Arrow Strategy Compliance")
        analysis.append(f"- **Overall Risk Level**: {risk_assessment.get('overall_risk', 'UNKNOWN')}")
        
        # Check each risk control
        metrics_check = risk_assessment.get('metrics_check', {})
        
        if 'sharpe_ratio' in metrics_check:
            sharpe_check = metrics_check['sharpe_ratio']
            analysis.append(f"- **Sharpe Ratio**: {sharpe_check['current']:.2f} (Target: >{sharpe_check['target']}) - {sharpe_check['status']}")
        
        if 'volatility' in metrics_check:
            vol_check = metrics_check['volatility']
            analysis.append(f"- **Volatility**: {vol_check['current']:.2%} (Cap: <{vol_check['cap']:.2%}) - {vol_check['status']}")
        
        if 'max_drawdown' in metrics_check:
            dd_check = metrics_check['max_drawdown']
            analysis.append(f"- **Max Drawdown**: {dd_check['current']:.2%} (Limit: <{dd_check['limit']:.2%}) - {dd_check['status']}")
        
        analysis.append("")
        
        # Risk alerts
        alerts = risk_assessment.get('alerts', [])
        if alerts:
            analysis.append(f"### Risk Alerts")
            for alert in alerts:
                analysis.append(f"- ⚠️  {alert}")
            analysis.append("")
        
        # Rebalancing needs
        rebalance_rec = strategy_analysis.get('rebalance_recommendation', {})
        if rebalance_rec.get('rebalance_needed', False):
            analysis.append(f"### Rebalancing Required")
            analysis.append(f"- **Reason**: {rebalance_rec.get('rationale', 'Portfolio drift detected')}")
            analysis.append(f"- **Estimated Cost**: ${rebalance_rec.get('estimated_cost', 0):.2f}")
            analysis.append("")
        
        return "\n".join(analysis)
    
    async def _comprehensive_risk_assessment(self, symbols: List[str], portfolio_dict: Dict[str, float]) -> str:
        """Comprehensive risk assessment combining all methods"""
        volatility_assessment = await self._assess_volatility_risk(symbols)
        drawdown_assessment = await self._assess_drawdown_risk(symbols)
        correlation_assessment = await self._assess_correlation_risk(symbols)
        
        analysis = []
        analysis.append("# Comprehensive Risk Assessment")
        analysis.append(f"**Assessment Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        analysis.append(f"**Symbols Analyzed**: {', '.join(symbols)}\n")
        analysis.append(volatility_assessment)
        analysis.append(drawdown_assessment)
        analysis.append(correlation_assessment)
        
        # Add VaR and strategy compliance if portfolio provided
        if portfolio_dict:
            var_assessment = await self._calculate_var(symbols, portfolio_dict)
            strategy_compliance = await self._assess_strategy_compliance(portfolio_dict)
            analysis.append(var_assessment)
            analysis.append(strategy_compliance)
        
        # Summary and recommendations
        analysis.append("## Risk Management Recommendations")
        analysis.append("- Monitor portfolio volatility against strategy targets")
        analysis.append("- Rebalance when drift exceeds 5% threshold")
        analysis.append("- Consider reducing position sizes during high volatility periods")
        analysis.append("- Maintain diversification to reduce correlation risk")
        analysis.append("- Regular stress testing and scenario analysis recommended")
        
        return "\n".join(analysis)
    
    def _get_volatility_trend(self, returns: pd.Series) -> str:
        """Determine if volatility is increasing or decreasing"""
        if len(returns) < 60:
            return "INSUFFICIENT_DATA"
        
        recent_vol = returns.tail(30).std()
        older_vol = returns.head(30).std()
        
        change = (recent_vol - older_vol) / older_vol
        
        if change > 0.2:
            return "INCREASING"
        elif change < -0.2:
            return "DECREASING"
        else:
            return "STABLE"
    
    def _calculate_recovery_time(self, prices: pd.Series, drawdowns: pd.Series) -> int:
        """Calculate average recovery time from drawdowns"""
        try:
            recovery_times = []
            in_drawdown = False
            drawdown_start = None
            
            for i, dd in enumerate(drawdowns):
                if dd < -0.05 and not in_drawdown:  # Start of significant drawdown
                    in_drawdown = True
                    drawdown_start = i
                elif dd >= 0 and in_drawdown:  # Recovery
                    if drawdown_start is not None:
                        recovery_times.append(i - drawdown_start)
                    in_drawdown = False
                    drawdown_start = None
            
            return int(np.mean(recovery_times)) if recovery_times else 0
        except:
            return 0

class StressTestTool(BaseTool):
    """Tool for stress testing and scenario analysis"""
    name = "stress_test"
    description = "Perform stress tests and scenario analysis on portfolio"
    market_data_service: MarketDataService = None
    
    def __init__(self, market_data_service: MarketDataService):
        super().__init__()
        self.market_data_service = market_data_service
    
    async def _arun(self, symbols: str, portfolio_values: str = None, scenario: str = "market_crash") -> str:
        """Perform stress testing"""
        try:
            symbol_list = [s.strip() for s in symbols.split(',')]
            
            # Parse portfolio values
            portfolio_dict = {}
            if portfolio_values:
                value_list = [float(v.strip()) for v in portfolio_values.split(',')]
                portfolio_dict = dict(zip(symbol_list, value_list))
            
            if scenario == "market_crash":
                return await self._stress_test_market_crash(symbol_list, portfolio_dict)
            elif scenario == "interest_rate_shock":
                return await self._stress_test_interest_rates(symbol_list, portfolio_dict)
            elif scenario == "inflation_spike":
                return await self._stress_test_inflation(symbol_list, portfolio_dict)
            else:
                return await self._comprehensive_stress_test(symbol_list, portfolio_dict)
                
        except Exception as e:
            return f"Error in stress testing: {str(e)}"
    
    def _run(self, symbols: str, portfolio_values: str = None, scenario: str = "market_crash") -> str:
        """Sync wrapper"""
        return asyncio.run(self._arun(symbols, portfolio_values, scenario))
    
    async def _stress_test_market_crash(self, symbols: List[str], portfolio_dict: Dict[str, float]) -> str:
        """Stress test for market crash scenario"""
        analysis = []
        analysis.append("## Market Crash Stress Test\n")
        analysis.append("**Scenario**: 20% market decline over 1 month\n")
        
        # Simulate market crash impact
        crash_impacts = {
            'VTI': -0.20,    # US stocks hit hard
            'BNDX': -0.05,   # International bonds less affected
            'GSG': -0.15     # Commodities decline but less than stocks
        }
        
        total_loss = 0
        original_value = sum(portfolio_dict.values()) if portfolio_dict else 0
        
        for symbol in symbols:
            impact = crash_impacts.get(symbol, -0.20)  # Default 20% decline
            original_pos = portfolio_dict.get(symbol, 0) if portfolio_dict else 0
            loss = original_pos * abs(impact)
            total_loss += loss
            
            analysis.append(f"### {symbol} Impact")
            analysis.append(f"- **Assumed Decline**: {impact:.1%}")
            analysis.append(f"- **Position Value**: ${original_pos:,.2f}")
            analysis.append(f"- **Estimated Loss**: ${loss:,.2f}")
            analysis.append("")
        
        if original_value > 0:
            total_decline = total_loss / original_value
            new_value = original_value - total_loss
            
            analysis.append(f"### Portfolio Impact Summary")
            analysis.append(f"- **Original Portfolio Value**: ${original_value:,.2f}")
            analysis.append(f"- **New Portfolio Value**: ${new_value:,.2f}")
            analysis.append(f"- **Total Decline**: {total_decline:.1%}")
            analysis.append(f"- **Recovery Needed**: {(total_loss/new_value):.1%}")
            analysis.append("")
        
        return "\n".join(analysis)
    
    async def _stress_test_interest_rates(self, symbols: List[str], portfolio_dict: Dict[str, float]) -> str:
        """Stress test for interest rate shock"""
        analysis = []
        analysis.append("## Interest Rate Shock Stress Test\n")
        analysis.append("**Scenario**: Fed raises rates by 200 basis points unexpectedly\n")
        
        # Interest rate sensitivity
        rate_impacts = {
            'VTI': -0.08,    # Stocks moderately affected
            'BNDX': -0.12,   # Bonds significantly affected
            'GSG': -0.05     # Commodities less affected
        }
        
        for symbol in symbols:
            impact = rate_impacts.get(symbol, -0.08)
            analysis.append(f"### {symbol} - Rate Sensitivity")
            analysis.append(f"- **Expected Impact**: {impact:.1%}")
            analysis.append(f"- **Risk Level**: {'HIGH' if abs(impact) > 0.10 else 'MODERATE' if abs(impact) > 0.05 else 'LOW'}")
            analysis.append("")
        
        return "\n".join(analysis)
    
    async def _stress_test_inflation(self, symbols: List[str], portfolio_dict: Dict[str, float]) -> str:
        """Stress test for inflation spike"""
        analysis = []
        analysis.append("## Inflation Spike Stress Test\n")
        analysis.append("**Scenario**: Inflation jumps to 8% unexpectedly\n")
        
        # Inflation sensitivity
        inflation_impacts = {
            'VTI': -0.10,    # Stocks hurt by inflation
            'BNDX': -0.15,   # Bonds hurt more
            'GSG': +0.05     # Commodities benefit
        }
        
        for symbol in symbols:
            impact = inflation_impacts.get(symbol, -0.10)
            analysis.append(f"### {symbol} - Inflation Sensitivity")
            analysis.append(f"- **Expected Impact**: {impact:+.1%}")
            analysis.append(f"- **Inflation Hedge**: {'YES' if impact > 0 else 'NO'}")
            analysis.append("")
        
        return "\n".join(analysis)
    
    async def _comprehensive_stress_test(self, symbols: List[str], portfolio_dict: Dict[str, float]) -> str:
        """Comprehensive stress testing"""
        crash_test = await self._stress_test_market_crash(symbols, portfolio_dict)
        rate_test = await self._stress_test_interest_rates(symbols, portfolio_dict)
        inflation_test = await self._stress_test_inflation(symbols, portfolio_dict)
        
        analysis = []
        analysis.append("# Comprehensive Stress Test Analysis")
        analysis.append(f"**Test Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        analysis.append(crash_test)
        analysis.append(rate_test)
        analysis.append(inflation_test)
        
        analysis.append("## Stress Test Summary")
        analysis.append("- Portfolio resilience tested across multiple scenarios")
        analysis.append("- Diversification provides protection in most scenarios")
        analysis.append("- Consider increasing commodity allocation for inflation protection")
        analysis.append("- Regular stress testing recommended quarterly")
        
        return "\n".join(analysis)

def create_risk_analyst_agent(market_data_service: MarketDataService, 
                             strategy_service: StraightArrowStrategyService) -> Agent:
    """Create the Risk Analyst agent"""
    
    # Create tools
    risk_tool = RiskAssessmentTool(market_data_service, strategy_service)
    stress_tool = StressTestTool(market_data_service)
    
    # Create agent
    risk_analyst = Agent(
        role="Senior Risk Analyst",
        goal="Monitor and assess portfolio risk, ensure compliance with strategy risk controls, and provide risk management recommendations",
        backstory="""You are a senior risk management professional with expertise in:
        - Portfolio risk measurement and monitoring
        - Value at Risk (VaR) and Expected Shortfall calculations
        - Stress testing and scenario analysis
        - Correlation analysis and diversification assessment
        - Risk control compliance and monitoring
        - Maximum drawdown and volatility risk assessment
        
        You are responsible for ensuring the portfolio stays within acceptable risk limits
        as defined by the Straight Arrow investment strategy. You provide early warnings
        of risk concentrations and recommend risk mitigation strategies. Your analysis
        helps maintain the portfolio's risk-adjusted return profile and protects against
        significant losses.""",
        
        tools=[risk_tool, stress_tool],
        verbose=True,
        allow_delegation=False,
        max_iter=3,
        memory=True
    )
    
    return risk_analyst

def create_risk_analysis_tasks(symbols: List[str], portfolio_values: List[float] = None) -> List[Task]:
    """Create risk analysis tasks"""
    
    symbols_str = ','.join(symbols)
    portfolio_values_str = ','.join(map(str, portfolio_values)) if portfolio_values else None
    
    tasks = []
    
    # Task 1: Risk Assessment
    risk_task = Task(
        description=f"""Perform comprehensive risk assessment for symbols: {symbols_str}
        
        Portfolio Values: {portfolio_values_str if portfolio_values_str else "Not provided"}
        
        Requirements:
        1. Assess volatility risk for each position and overall portfolio
        2. Calculate Value at Risk (VaR) and Expected Shortfall
        3. Analyze correlation risk and diversification benefits
        4. Evaluate maximum drawdown risk and recovery times
        5. Check compliance with Straight Arrow strategy risk controls
        
        Provide detailed risk assessment with specific recommendations.""",
        
        expected_output="Comprehensive risk assessment with volatility analysis, VaR calculations, correlation analysis, and strategy compliance check",
        tools=["risk_assessment"],
        async_execution=True
    )
    tasks.append(risk_task)
    
    # Task 2: Stress Testing
    stress_task = Task(
        description=f"""Perform stress testing and scenario analysis for symbols: {symbols_str}
        
        Portfolio Values: {portfolio_values_str if portfolio_values_str else "Not provided"}
        
        Requirements:
        1. Test portfolio resilience under market crash scenario
        2. Analyze impact of interest rate shocks
        3. Assess inflation spike sensitivity
        4. Evaluate recovery potential and time horizons
        5. Identify portfolio vulnerabilities and hedge recommendations
        
        Provide stress test results with scenario impact analysis.""",
        
        expected_output="Stress test analysis with scenario impacts, portfolio vulnerabilities, and risk mitigation recommendations",
        tools=["stress_test"],
        async_execution=True
    )
    tasks.append(stress_task)
    
    return tasks