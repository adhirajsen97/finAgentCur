# Task On Hand

## Status: [DISCOVERY]

## Project Overview
Building an async Python AI investing agent with 4 specialized models for deployment on Render platform.

## Required Models
1. **Data Analyst** - Processes live market data and financial metrics
2. **Trading Analyst** - Analyzes market trends and trading signals  
3. **Execution Analyst** - Handles trade execution and order management
4. **Risk Analyst** - Monitors risk metrics and portfolio exposure

## Investment Strategy
Following the "Straight Arrow" strategy from investment_strategy.py:
- **Core Strategy**: Diversified Three-Fund Style
- **Base Allocation**: VTI (60%), BNDX (30%), GSG (10%)
- **Risk Controls**: Sharpe > 0.5, Volatility Cap < 12%, Tax Loss Harvesting

## Technology Stack Decision Needed
- Framework: CrewAI vs Custom Multi-Agent vs Semantic Kernel
- Live Data Source: Alpha Vantage, Yahoo Finance, or Polygon.io
- Deployment: Render with async FastAPI

## Open Questions
1. Should we use CrewAI framework or build custom agents?
2. Which market data provider offers best real-time data for the budget?
3. How to implement the Straight Arrow strategy constraints in the agent logic?
4. What's the deployment architecture on Render for async agents?

## Progress Checkboxes
- [ ] Technology stack research and selection
- [ ] Environment setup and dependencies
- [ ] Agent architecture design
- [ ] Live market data integration
- [ ] Straight Arrow strategy implementation
- [ ] Async orchestration setup
- [ ] Render deployment configuration
- [ ] Testing and validation

## Next Steps
1. Research and compare agent frameworks
2. Set up development environment
3. Design multi-agent architecture
4. Implement market data feeds

## Last Updated
2025-01-06 - Initial project setup and discovery phase 