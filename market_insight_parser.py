"""
Market Insight Parser & Portfolio Rebalancer API
===============================================

This module provides two main APIs:
1. URL Content Parser - Extracts and validates financial content from URLs
2. AI Portfolio Rebalancer - Uses market insights to rebalance portfolios

Features:
- Web scraping with content validation
- Financial content classification
- AI-powered market insight analysis
- Dynamic portfolio rebalancing
- Risk profile adjustments
- Sector-based reallocation
"""

import asyncio
import re
import json
import aiohttp
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from urllib.parse import urlparse, urljoin
from fastapi import HTTPException
from pydantic import BaseModel, Field, validator
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class URLParseRequest(BaseModel):
    """Request to parse financial content from a URL"""
    url: str = Field(..., description="URL to parse for financial content")
    include_raw_content: Optional[bool] = Field(default=False, description="Include raw HTML content")
    max_content_length: Optional[int] = Field(default=50000, description="Maximum content length to process")
    
    @validator('url')
    def validate_url(cls, v):
        try:
            result = urlparse(v)
            if not result.scheme or not result.netloc:
                raise ValueError("Invalid URL format")
            return v
        except Exception:
            raise ValueError("Invalid URL provided")

class FinancialContent(BaseModel):
    """Extracted financial content from URL"""
    title: str = Field(..., description="Article/page title")
    content: str = Field(..., description="Main content text")
    summary: str = Field(..., description="AI-generated summary")
    financial_keywords: List[str] = Field(..., description="Detected financial keywords")
    market_sentiment: str = Field(..., description="Overall market sentiment (BULLISH/BEARISH/NEUTRAL)")
    sectors_mentioned: List[str] = Field(..., description="Financial sectors mentioned")
    tickers_mentioned: List[str] = Field(..., description="Stock tickers mentioned")
    key_insights: List[str] = Field(..., description="Key market insights extracted")
    confidence_score: float = Field(..., description="Confidence in financial relevance (0-100)")
    
class URLParseResponse(BaseModel):
    """Response from URL parsing"""
    url: str = Field(..., description="Original URL")
    is_financial: bool = Field(..., description="Whether content is financial/market related")
    financial_content: Optional[FinancialContent] = Field(None, description="Extracted financial content")
    error_message: Optional[str] = Field(None, description="Error message if parsing failed")
    metadata: Dict[str, Any] = Field(..., description="Additional metadata")
    timestamp: str = Field(..., description="Processing timestamp")

class PortfolioHolding(BaseModel):
    """Individual portfolio holding"""
    symbol: str = Field(..., description="Stock ticker symbol")
    current_shares: float = Field(..., gt=0, description="Current number of shares")
    current_value: float = Field(..., gt=0, description="Current market value")
    sector: Optional[str] = Field(None, description="Sector classification")
    purchase_price: Optional[float] = Field(None, description="Average purchase price")
    
class RiskProfile(BaseModel):
    """User risk profile"""
    risk_score: int = Field(..., ge=1, le=5, description="Risk score (1=Conservative, 5=Aggressive)")
    risk_tolerance: str = Field(..., description="Risk tolerance level")
    time_horizon: str = Field(..., description="Investment time horizon")
    liquidity_needs: str = Field(..., description="Liquidity requirements")
    sector_preferences: List[str] = Field(default=[], description="Preferred sectors")
    sector_restrictions: List[str] = Field(default=[], description="Restricted sectors")

class PortfolioRebalanceRequest(BaseModel):
    """Request for AI-powered portfolio rebalancing"""
    portfolio_holdings: List[PortfolioHolding] = Field(..., description="Current portfolio holdings")
    total_portfolio_value: float = Field(..., gt=0, description="Total portfolio value")
    available_cash: Optional[float] = Field(default=0.0, description="Available cash for rebalancing")
    risk_profile: RiskProfile = Field(..., description="Current risk profile")
    market_insights: FinancialContent = Field(..., description="Market insights from URL parsing")
    rebalance_type: Optional[str] = Field(default="market_insight", description="Type of rebalancing strategy (market_insight, time_based, risk_adjustment, manual)")
    rebalance_constraints: Optional[Dict[str, Any]] = Field(default={}, description="Rebalancing constraints")

class RebalancingAction(BaseModel):
    """Individual rebalancing action"""
    symbol: str = Field(..., description="Stock ticker symbol")
    action: str = Field(..., description="BUY, SELL, or HOLD")
    current_shares: float = Field(..., description="Current shares held")
    target_shares: float = Field(..., description="Target shares after rebalancing")
    shares_to_trade: float = Field(..., description="Shares to buy/sell")
    dollar_amount: float = Field(..., description="Dollar amount of trade")
    current_allocation: float = Field(..., description="Current allocation percentage")
    target_allocation: float = Field(..., description="Target allocation percentage")
    rationale: str = Field(..., description="Reason for this action")
    priority: str = Field(..., description="Priority level (HIGH/MEDIUM/LOW)")

class AdjustedRiskProfile(BaseModel):
    """Risk profile adjustments based on market insights"""
    original_risk_score: int = Field(..., description="Original risk score")
    adjusted_risk_score: int = Field(..., description="Adjusted risk score")
    adjustment_reason: str = Field(..., description="Reason for risk adjustment")
    sector_allocation_changes: Dict[str, float] = Field(..., description="Sector allocation adjustments")
    time_horizon_impact: str = Field(..., description="Impact on time horizon")

class PortfolioRebalanceResponse(BaseModel):
    """Response from portfolio rebalancing"""
    rebalancing_actions: List[RebalancingAction] = Field(..., description="Recommended rebalancing actions")
    adjusted_risk_profile: AdjustedRiskProfile = Field(..., description="Adjusted risk profile")
    rebalance_type: str = Field(..., description="Type of rebalancing strategy used")
    market_insight_summary: str = Field(..., description="Summary of how insights influenced rebalancing")
    projected_portfolio_value: float = Field(..., description="Projected portfolio value after rebalancing")
    risk_metrics: Dict[str, float] = Field(..., description="Projected risk metrics")
    implementation_timeline: str = Field(..., description="Recommended implementation timeline")
    monitoring_recommendations: List[str] = Field(..., description="Monitoring recommendations")
    confidence_score: float = Field(..., description="Confidence in rebalancing recommendations")
    timestamp: str = Field(..., description="Analysis timestamp")

# ============================================================================
# URL CONTENT PARSER SERVICE
# ============================================================================

class URLContentParser:
    """Service for parsing and validating financial content from URLs"""
    
    def __init__(self, ai_service):
        self.ai_service = ai_service
        self.financial_keywords = [
            'stock', 'market', 'trading', 'investment', 'portfolio', 'earnings', 'revenue',
            'profit', 'dividend', 'share', 'equity', 'bond', 'fund', 'etf', 'ipo',
            'acquisition', 'merger', 'volatility', 'inflation', 'fed', 'interest rate',
            'gdp', 'economy', 'recession', 'bull market', 'bear market', 'analyst',
            'recommendation', 'price target', 'upgrade', 'downgrade', 'sector',
            'financial', 'fiscal', 'monetary', 'currency', 'forex', 'commodity'
        ]
        
        self.sector_keywords = {
            'Technology': ['tech', 'software', 'ai', 'cloud', 'semiconductor', 'internet'],
            'Healthcare': ['health', 'pharma', 'biotech', 'medical', 'drug', 'hospital'],
            'Finance': ['bank', 'insurance', 'financial', 'credit', 'loan', 'fintech'],
            'Energy': ['oil', 'gas', 'renewable', 'solar', 'wind', 'energy'],
            'Consumer': ['retail', 'consumer', 'brand', 'restaurant', 'automotive'],
            'Industrial': ['manufacturing', 'industrial', 'aerospace', 'defense'],
            'Real Estate': ['real estate', 'reit', 'property', 'housing', 'construction'],
            'Utilities': ['utility', 'electric', 'water', 'telecommunications']
        }
    
    async def parse_url_content(self, request: URLParseRequest) -> URLParseResponse:
        """Parse financial content from URL"""
        try:
            # Fetch content from URL
            content_data = await self._fetch_url_content(request.url, request.max_content_length)
            
            if not content_data:
                return URLParseResponse(
                    url=request.url,
                    is_financial=False,
                    error_message="Failed to fetch content from URL. This may be due to unsupported content type (PDF, image, video) or network issues.",
                    metadata={"status": "failed", "reason": "content_fetch_error"},
                    timestamp=datetime.now().isoformat()
                )
            
            # Extract text content
            extracted_text = self._extract_text_content(content_data['html'])
            
            # Validate financial relevance
            financial_relevance = self._validate_financial_content(extracted_text)
            
            if not financial_relevance['is_financial']:
                return URLParseResponse(
                    url=request.url,
                    is_financial=False,
                    error_message="Content is not financial/market related",
                    metadata={
                        "status": "not_financial",
                        "confidence": financial_relevance['confidence'],
                        "detected_keywords": financial_relevance['keywords_found']
                    },
                    timestamp=datetime.now().isoformat()
                )
            
            # Extract financial insights using AI
            financial_content = await self._extract_financial_insights(
                extracted_text, content_data['title'], request.url
            )
            
            return URLParseResponse(
                url=request.url,
                is_financial=True,
                financial_content=financial_content,
                metadata={
                    "status": "success",
                    "content_length": len(extracted_text),
                    "processing_time": content_data.get('fetch_time', 0)
                },
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"URL parsing error for {request.url}: {e}")
            return URLParseResponse(
                url=request.url,
                is_financial=False,
                error_message=f"Error parsing URL: {str(e)}",
                metadata={"status": "error", "error_type": type(e).__name__},
                timestamp=datetime.now().isoformat()
            )
    
    async def _fetch_url_content(self, url: str, max_length: int) -> Optional[Dict[str, Any]]:
        """Fetch content from URL with error handling for different content types"""
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
                start_time = datetime.now()
                async with session.get(url) as response:
                    if response.status != 200:
                        logger.warning(f"HTTP {response.status} for URL: {url}")
                        return None
                    
                    # Check content type
                    content_type = response.headers.get('content-type', '').lower()
                    logger.info(f"Content type for {url}: {content_type}")
                    
                    # Handle different content types
                    if 'pdf' in content_type or url.lower().endswith('.pdf'):
                        logger.warning(f"PDF content detected at {url}. PDF parsing not supported yet.")
                        return None
                    elif 'image' in content_type or any(url.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']):
                        logger.warning(f"Image content detected at {url}. Image parsing not supported.")
                        return None
                    elif 'video' in content_type or any(url.lower().endswith(ext) for ext in ['.mp4', '.avi', '.mov']):
                        logger.warning(f"Video content detected at {url}. Video parsing not supported.")
                        return None
                    
                    # Try to read as text with encoding detection
                    try:
                        content = await response.text(encoding='utf-8')
                    except UnicodeDecodeError:
                        try:
                            # Try reading as bytes and decode with latin-1 (fallback)
                            content_bytes = await response.read()
                            content = content_bytes.decode('latin-1', errors='ignore')
                            logger.warning(f"Used latin-1 encoding fallback for {url}")
                        except Exception as decode_error:
                            logger.error(f"Failed to decode content from {url}: {decode_error}")
                            return None
                    
                    fetch_time = (datetime.now() - start_time).total_seconds()
                    
                    # Limit content length
                    if len(content) > max_length:
                        content = content[:max_length]
                    
                    # Extract title from HTML (with error handling)
                    title = "No Title"
                    try:
                        from bs4 import BeautifulSoup
                        soup = BeautifulSoup(content, 'html.parser')
                        title = soup.title.string if soup.title else "No Title"
                    except ImportError:
                        # Fallback without BeautifulSoup
                        import re
                        title_match = re.search(r'<title[^>]*>(.*?)</title>', content, re.IGNORECASE | re.DOTALL)
                        if title_match:
                            title = title_match.group(1).strip()
                        else:
                            # Try to extract from URL
                            title = url.split('/')[-1] if '/' in url else url
                    except Exception as e:
                        logger.warning(f"Failed to extract title from {url}: {e}")
                        title = url.split('/')[-1] if '/' in url else "Extracted Content"
                    
                    return {
                        'html': content,
                        'title': title.strip() if title else "No Title",
                        'fetch_time': fetch_time,
                        'status_code': response.status,
                        'content_type': content_type
                    }
                    
        except Exception as e:
            logger.error(f"Error fetching URL {url}: {e}")
            return None
    
    def _extract_text_content(self, html: str) -> str:
        """Extract clean text content from HTML with fallback options"""
        try:
            # Try BeautifulSoup first
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(html, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style", "nav", "header", "footer"]):
                    script.decompose()
                
                # Extract text from main content areas
                main_content = soup.find('main') or soup.find('article') or soup.find('body') or soup
                
                # Get text and clean it
                text = main_content.get_text()
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = ' '.join(chunk for chunk in chunks if chunk)
                
                return text
                
            except ImportError:
                # Fallback without BeautifulSoup - use regex
                logger.info("BeautifulSoup not available, using regex fallback for text extraction")
                import re
                
                # Remove script and style tags
                html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
                html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
                html = re.sub(r'<nav[^>]*>.*?</nav>', '', html, flags=re.DOTALL | re.IGNORECASE)
                html = re.sub(r'<header[^>]*>.*?</header>', '', html, flags=re.DOTALL | re.IGNORECASE)
                html = re.sub(r'<footer[^>]*>.*?</footer>', '', html, flags=re.DOTALL | re.IGNORECASE)
                
                # Remove all HTML tags
                text = re.sub(r'<[^>]+>', ' ', html)
                
                # Clean up whitespace
                text = re.sub(r'\s+', ' ', text)
                text = text.strip()
                
                return text
            
        except Exception as e:
            logger.error(f"Error extracting text content: {e}")
            return ""
    
    def _validate_financial_content(self, text: str) -> Dict[str, Any]:
        """Validate if content is financial/market related"""
        text_lower = text.lower()
        keywords_found = []
        
        # Count financial keywords
        for keyword in self.financial_keywords:
            if keyword in text_lower:
                keywords_found.append(keyword)
        
        # Calculate confidence based on keyword density
        total_words = len(text.split())
        keyword_density = len(keywords_found) / max(total_words, 1) * 100
        
        # Determine if content is financial
        is_financial = len(keywords_found) >= 3 and keyword_density > 0.1
        confidence = min(len(keywords_found) * 10, 100)
        
        return {
            'is_financial': is_financial,
            'confidence': confidence,
            'keywords_found': keywords_found,
            'keyword_density': keyword_density
        }
    
    async def _extract_financial_insights(self, text: str, title: str, url: str) -> FinancialContent:
        """Extract financial insights using AI"""
        try:
            # Detect sectors and tickers
            sectors_mentioned = self._detect_sectors(text)
            tickers_mentioned = self._extract_tickers(text)
            
            # AI analysis prompt
            prompt = f"""
            Analyze this financial content and provide structured insights:
            
            TITLE: {title}
            URL: {url}
            CONTENT: {text[:3000]}...
            
            Please provide:
            1. A concise summary (2-3 sentences)
            2. Overall market sentiment (BULLISH/BEARISH/NEUTRAL)
            3. Top 5 key market insights
            4. Confidence score (0-100) for financial relevance
            
            Focus on actionable investment insights and market trends.
            """
            
            ai_response = await self.ai_service._call_openai(prompt, "financial_content_analysis")
            
            # Parse AI response and extract insights
            insights = self._parse_ai_insights(ai_response)
            
            return FinancialContent(
                title=title,
                content=text[:1000] + "..." if len(text) > 1000 else text,
                summary=insights.get('summary', 'AI analysis not available'),
                financial_keywords=self._validate_financial_content(text)['keywords_found'][:10],
                market_sentiment=insights.get('sentiment', 'NEUTRAL'),
                sectors_mentioned=sectors_mentioned,
                tickers_mentioned=tickers_mentioned,
                key_insights=insights.get('key_insights', []),
                confidence_score=insights.get('confidence', 75.0)
            )
            
        except Exception as e:
            logger.error(f"Error extracting financial insights: {e}")
            # Return fallback content
            return FinancialContent(
                title=title,
                content=text[:1000] + "..." if len(text) > 1000 else text,
                summary="Financial content detected but AI analysis unavailable",
                financial_keywords=self._validate_financial_content(text)['keywords_found'][:10],
                market_sentiment="NEUTRAL",
                sectors_mentioned=self._detect_sectors(text),
                tickers_mentioned=self._extract_tickers(text),
                key_insights=["Content analysis pending"],
                confidence_score=60.0
            )
    
    def _detect_sectors(self, text: str) -> List[str]:
        """Detect financial sectors mentioned in text"""
        text_lower = text.lower()
        sectors_found = []
        
        for sector, keywords in self.sector_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                sectors_found.append(sector)
        
        return sectors_found
    
    def _extract_tickers(self, text: str) -> List[str]:
        """Extract stock tickers from text"""
        # Pattern for stock tickers (3-5 uppercase letters)
        ticker_pattern = r'\b[A-Z]{3,5}\b'
        potential_tickers = re.findall(ticker_pattern, text)
        
        # Filter out common false positives
        false_positives = ['THE', 'AND', 'FOR', 'YOU', 'ARE', 'BUT', 'NOT', 'ALL', 'CAN', 'HAD', 'HER', 'WAS', 'ONE', 'OUR', 'OUT', 'DAY', 'GET', 'HAS', 'HIM', 'HOW', 'ITS', 'NEW', 'NOW', 'OLD', 'SEE', 'TWO', 'WHO', 'BOY', 'DID', 'HAS', 'LET', 'PUT', 'SAY', 'SHE', 'TOO', 'USE']
        
        tickers = [ticker for ticker in potential_tickers if ticker not in false_positives]
        return list(set(tickers))[:10]  # Limit to 10 unique tickers
    
    def _parse_ai_insights(self, ai_response: str) -> Dict[str, Any]:
        """Parse AI response for structured insights"""
        try:
            # Extract key components from AI response
            insights = {
                'summary': self._extract_section(ai_response, 'summary'),
                'sentiment': self._extract_sentiment(ai_response),
                'key_insights': self._extract_key_insights(ai_response),
                'confidence': self._extract_confidence(ai_response)
            }
            return insights
            
        except Exception as e:
            logger.error(f"Error parsing AI insights: {e}")
            return {
                'summary': 'AI analysis parsing error',
                'sentiment': 'NEUTRAL',
                'key_insights': ['Analysis unavailable'],
                'confidence': 50.0
            }
    
    def _extract_section(self, text: str, section: str) -> str:
        """Extract specific section from AI response"""
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if section.lower() in line.lower():
                # Return next few lines as summary
                return ' '.join(lines[i+1:i+4]).strip()
        return "Section not found"
    
    def _extract_sentiment(self, text: str) -> str:
        """Extract sentiment from AI response"""
        text_upper = text.upper()
        if 'BULLISH' in text_upper:
            return 'BULLISH'
        elif 'BEARISH' in text_upper:
            return 'BEARISH'
        else:
            return 'NEUTRAL'
    
    def _extract_key_insights(self, text: str) -> List[str]:
        """Extract key insights from AI response"""
        # Look for numbered lists or bullet points
        insights = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if re.match(r'^\d+\.', line) or line.startswith('•') or line.startswith('-'):
                insight = re.sub(r'^\d+\.\s*|^[•-]\s*', '', line)
                if len(insight) > 10:  # Filter out very short items
                    insights.append(insight)
        
        return insights[:5]  # Limit to top 5 insights
    
    def _extract_confidence(self, text: str) -> float:
        """Extract confidence score from AI response"""
        # Look for confidence score patterns
        confidence_patterns = [
            r'confidence[:\s]+(\d+)',
            r'(\d+)%?\s*confidence',
            r'score[:\s]+(\d+)'
        ]
        
        for pattern in confidence_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return float(match.group(1))
        
        return 75.0  # Default confidence

# ============================================================================
# AI PORTFOLIO REBALANCER SERVICE
# ============================================================================

class AIPortfolioRebalancer:
    """Service for AI-powered portfolio rebalancing based on market insights"""
    
    def __init__(self, ai_service, market_service):
        self.ai_service = ai_service
        self.market_service = market_service
    
    async def rebalance_portfolio(self, request: PortfolioRebalanceRequest) -> PortfolioRebalanceResponse:
        """Rebalance portfolio based on market insights"""
        try:
            # Get current market data for portfolio holdings
            symbols = [holding.symbol for holding in request.portfolio_holdings]
            market_data = await self.market_service.get_quotes(symbols)
            
            # Analyze market insights impact on portfolio
            insight_analysis = await self._analyze_market_insights_impact(
                request.market_insights, request.portfolio_holdings, request.risk_profile
            )
            
            # Adjust risk profile based on insights
            adjusted_risk_profile = self._adjust_risk_profile(
                request.risk_profile, request.market_insights, insight_analysis
            )
            
            # Generate rebalancing recommendations
            rebalancing_actions = await self._generate_rebalancing_actions(
                request.portfolio_holdings, request.market_insights, 
                adjusted_risk_profile, market_data, request.available_cash, request.rebalance_type
            )
            
            # Calculate projected metrics
            projected_value = self._calculate_projected_portfolio_value(
                request.portfolio_holdings, rebalancing_actions, market_data
            )
            
            risk_metrics = self._calculate_risk_metrics(
                rebalancing_actions, adjusted_risk_profile, market_data
            )
            
            # Generate monitoring recommendations
            monitoring_recs = self._generate_monitoring_recommendations(
                request.market_insights, rebalancing_actions, adjusted_risk_profile
            )
            
            return PortfolioRebalanceResponse(
                rebalancing_actions=rebalancing_actions,
                adjusted_risk_profile=adjusted_risk_profile,
                rebalance_type=request.rebalance_type,
                market_insight_summary=self._create_insight_summary(request.market_insights, insight_analysis),
                projected_portfolio_value=projected_value,
                risk_metrics=risk_metrics,
                implementation_timeline=self._determine_implementation_timeline(rebalancing_actions, request.rebalance_type),
                monitoring_recommendations=monitoring_recs,
                confidence_score=min(request.market_insights.confidence_score * 0.8, 95.0),
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Portfolio rebalancing error: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to rebalance portfolio: {str(e)}")
    
    async def _analyze_market_insights_impact(self, insights: FinancialContent, 
                                            holdings: List[PortfolioHolding], 
                                            risk_profile: RiskProfile) -> Dict[str, Any]:
        """Analyze how market insights impact current portfolio"""
        
        # Create analysis prompt
        current_sectors = [h.sector for h in holdings if h.sector]
        current_symbols = [h.symbol for h in holdings]
        
        prompt = f"""
        PORTFOLIO IMPACT ANALYSIS
        
        Market Insights:
        - Sentiment: {insights.market_sentiment}
        - Key Insights: {', '.join(insights.key_insights)}
        - Sectors Mentioned: {', '.join(insights.sectors_mentioned)}
        - Tickers Mentioned: {', '.join(insights.tickers_mentioned)}
        
        Current Portfolio:
        - Holdings: {', '.join(current_symbols)}
        - Sectors: {', '.join(set(current_sectors))}
        - Risk Profile: {risk_profile.risk_tolerance} (Score: {risk_profile.risk_score})
        
        Analyze:
        1. How do these insights affect current holdings?
        2. Which sectors should be increased/decreased?
        3. What risk adjustments are needed?
        4. Specific rebalancing recommendations
        
        Provide structured analysis with actionable recommendations.
        """
        
        try:
            ai_response = await self.ai_service._call_openai(prompt, "portfolio_impact_analysis")
            
            return {
                'impact_analysis': ai_response,
                'affected_holdings': self._identify_affected_holdings(holdings, insights),
                'sector_recommendations': self._extract_sector_recommendations(ai_response),
                'risk_adjustments': self._extract_risk_adjustments(ai_response)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market insights impact: {e}")
            return {
                'impact_analysis': 'AI analysis unavailable',
                'affected_holdings': [],
                'sector_recommendations': {},
                'risk_adjustments': {}
            }
    
    def _adjust_risk_profile(self, current_profile: RiskProfile, 
                           insights: FinancialContent, 
                           impact_analysis: Dict[str, Any]) -> AdjustedRiskProfile:
        """Adjust risk profile based on market insights"""
        
        original_score = current_profile.risk_score
        adjusted_score = original_score
        adjustment_reason = "No adjustment needed"
        
        # Adjust based on market sentiment
        if insights.market_sentiment == "BULLISH" and insights.confidence_score > 80:
            if original_score < 5:
                adjusted_score = min(original_score + 1, 5)
                adjustment_reason = "Increased risk tolerance due to strong bullish sentiment"
        elif insights.market_sentiment == "BEARISH" and insights.confidence_score > 80:
            if original_score > 1:
                adjusted_score = max(original_score - 1, 1)
                adjustment_reason = "Decreased risk tolerance due to strong bearish sentiment"
        
        # Sector allocation adjustments
        sector_changes = {}
        for sector in insights.sectors_mentioned:
            if sector in current_profile.sector_preferences:
                if insights.market_sentiment == "BULLISH":
                    sector_changes[sector] = 5.0  # Increase by 5%
                else:
                    sector_changes[sector] = -3.0  # Decrease by 3%
        
        return AdjustedRiskProfile(
            original_risk_score=original_score,
            adjusted_risk_score=adjusted_score,
            adjustment_reason=adjustment_reason,
            sector_allocation_changes=sector_changes,
            time_horizon_impact=f"Maintained {current_profile.time_horizon} with {insights.market_sentiment.lower()} outlook"
        )
    
    async def _generate_rebalancing_actions(self, holdings: List[PortfolioHolding],
                                          insights: FinancialContent,
                                          adjusted_risk: AdjustedRiskProfile,
                                          market_data: Dict[str, Any],
                                          available_cash: float,
                                          rebalance_type: str = "market_insight") -> List[RebalancingAction]:
        """Generate specific rebalancing actions"""
        
        actions = []
        total_value = sum(h.current_value for h in holdings)
        
        for holding in holdings:
            current_allocation = (holding.current_value / total_value) * 100
            current_price = market_data.get(holding.symbol, {}).get('price', 0)
            
            if current_price <= 0:
                continue
            
            # Determine target allocation based on insights
            target_allocation = self._calculate_target_allocation(
                holding, insights, adjusted_risk, current_allocation
            )
            
            # Calculate trade requirements
            target_value = (target_allocation / 100) * total_value
            value_diff = target_value - holding.current_value
            
            if abs(value_diff) > 100:  # Only rebalance if difference > $100
                target_shares = target_value / current_price
                shares_to_trade = target_shares - holding.current_shares
                
                action = "BUY" if shares_to_trade > 0 else "SELL"
                priority = self._determine_action_priority(holding, insights, abs(value_diff))
                rationale = self._generate_action_rationale(holding, insights, action, value_diff, rebalance_type)
                
                actions.append(RebalancingAction(
                    symbol=holding.symbol,
                    action=action,
                    current_shares=holding.current_shares,
                    target_shares=target_shares,
                    shares_to_trade=abs(shares_to_trade),
                    dollar_amount=abs(value_diff),
                    current_allocation=current_allocation,
                    target_allocation=target_allocation,
                    rationale=rationale,
                    priority=priority
                ))
        
        return sorted(actions, key=lambda x: {"HIGH": 3, "MEDIUM": 2, "LOW": 1}[x.priority], reverse=True)
    
    def _calculate_target_allocation(self, holding: PortfolioHolding, 
                                   insights: FinancialContent, 
                                   adjusted_risk: AdjustedRiskProfile,
                                   current_allocation: float) -> float:
        """Calculate target allocation for holding based on insights"""
        
        target_allocation = current_allocation
        
        # Adjust based on sector mentions
        if holding.sector in insights.sectors_mentioned:
            if insights.market_sentiment == "BULLISH":
                target_allocation *= 1.2  # Increase by 20%
            elif insights.market_sentiment == "BEARISH":
                target_allocation *= 0.8  # Decrease by 20%
        
        # Adjust based on direct ticker mentions
        if holding.symbol in insights.tickers_mentioned:
            if insights.market_sentiment == "BULLISH":
                target_allocation *= 1.15  # Increase by 15%
            elif insights.market_sentiment == "BEARISH":
                target_allocation *= 0.85  # Decrease by 15%
        
        # Apply sector allocation changes from risk profile
        if holding.sector in adjusted_risk.sector_allocation_changes:
            adjustment = adjusted_risk.sector_allocation_changes[holding.sector]
            target_allocation = max(target_allocation + adjustment, 0)
        
        # Ensure target allocation is reasonable (between 0-30%)
        return min(max(target_allocation, 0), 30)
    
    def _determine_action_priority(self, holding: PortfolioHolding, 
                                 insights: FinancialContent, 
                                 value_diff: float) -> str:
        """Determine priority for rebalancing action"""
        
        # High priority if directly mentioned in insights
        if holding.symbol in insights.tickers_mentioned:
            return "HIGH"
        
        # High priority if sector is heavily mentioned and large value difference
        if holding.sector in insights.sectors_mentioned and value_diff > 5000:
            return "HIGH"
        
        # Medium priority for significant value differences
        if value_diff > 2000:
            return "MEDIUM"
        
        return "LOW"
    
    def _generate_action_rationale(self, holding: PortfolioHolding, 
                                 insights: FinancialContent, 
                                 action: str, 
                                 value_diff: float,
                                 rebalance_type: str = "market_insight") -> str:
        """Generate rationale for rebalancing action"""
        
        # Base rationale varies by rebalance type
        if rebalance_type == "market_insight":
            base_rationale = f"{action} {holding.symbol} based on market insights"
        elif rebalance_type == "time_based":
            base_rationale = f"{action} {holding.symbol} for periodic rebalancing"
        elif rebalance_type == "risk_adjustment":
            base_rationale = f"{action} {holding.symbol} for risk management"
        elif rebalance_type == "manual":
            base_rationale = f"{action} {holding.symbol} per manual strategy"
        else:
            base_rationale = f"{action} {holding.symbol} based on analysis"
        
        # Add specific reasoning based on rebalance type
        reasons = []
        
        if rebalance_type == "market_insight":
            if holding.symbol in insights.tickers_mentioned:
                reasons.append(f"{holding.symbol} directly mentioned in market analysis")
            
            if holding.sector in insights.sectors_mentioned:
                reasons.append(f"{holding.sector} sector highlighted in insights")
            
            if insights.market_sentiment == "BULLISH" and action == "BUY":
                reasons.append("Bullish sentiment supports increased allocation")
            elif insights.market_sentiment == "BEARISH" and action == "SELL":
                reasons.append("Bearish sentiment suggests reduced exposure")
                
            if abs(value_diff) > 5000:
                reasons.append(f"Significant market-driven adjustment needed (${abs(value_diff):,.0f})")
                
        elif rebalance_type == "time_based":
            reasons.append("Scheduled portfolio rebalancing")
            if abs(value_diff) > 1000:
                reasons.append(f"Allocation drift correction (${abs(value_diff):,.0f})")
                
        elif rebalance_type == "risk_adjustment":
            reasons.append("Risk profile optimization")
            if abs(value_diff) > 2000:
                reasons.append(f"Risk management adjustment (${abs(value_diff):,.0f})")
                
        elif rebalance_type == "manual":
            if abs(value_diff) > 1000:
                reasons.append(f"Manual rebalancing directive (${abs(value_diff):,.0f})")
        
        if reasons:
            return f"{base_rationale}: {'; '.join(reasons)}"
        else:
            return base_rationale
    
    def _identify_affected_holdings(self, holdings: List[PortfolioHolding], 
                                  insights: FinancialContent) -> List[str]:
        """Identify holdings affected by market insights"""
        affected = []
        
        for holding in holdings:
            if holding.symbol in insights.tickers_mentioned:
                affected.append(holding.symbol)
            elif holding.sector in insights.sectors_mentioned:
                affected.append(holding.symbol)
        
        return affected
    
    def _extract_sector_recommendations(self, ai_response: str) -> Dict[str, str]:
        """Extract sector recommendations from AI response"""
        # Simple pattern matching for sector recommendations
        recommendations = {}
        
        sectors = ['Technology', 'Healthcare', 'Finance', 'Energy', 'Consumer', 'Industrial']
        for sector in sectors:
            if f"increase {sector.lower()}" in ai_response.lower():
                recommendations[sector] = "INCREASE"
            elif f"decrease {sector.lower()}" in ai_response.lower():
                recommendations[sector] = "DECREASE"
        
        return recommendations
    
    def _extract_risk_adjustments(self, ai_response: str) -> Dict[str, Any]:
        """Extract risk adjustments from AI response"""
        adjustments = {}
        
        if "increase risk" in ai_response.lower():
            adjustments['risk_direction'] = "INCREASE"
        elif "decrease risk" in ai_response.lower():
            adjustments['risk_direction'] = "DECREASE"
        else:
            adjustments['risk_direction'] = "MAINTAIN"
        
        return adjustments
    
    def _calculate_projected_portfolio_value(self, holdings: List[PortfolioHolding],
                                           actions: List[RebalancingAction],
                                           market_data: Dict[str, Any]) -> float:
        """Calculate projected portfolio value after rebalancing"""
        
        projected_value = 0
        
        for holding in holdings:
            current_price = market_data.get(holding.symbol, {}).get('price', 0)
            if current_price <= 0:
                continue
            
            # Find corresponding action
            action = next((a for a in actions if a.symbol == holding.symbol), None)
            
            if action:
                projected_shares = action.target_shares
            else:
                projected_shares = holding.current_shares
            
            projected_value += projected_shares * current_price
        
        return projected_value
    
    def _calculate_risk_metrics(self, actions: List[RebalancingAction],
                              adjusted_risk: AdjustedRiskProfile,
                              market_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate risk metrics for rebalanced portfolio"""
        
        # Simplified risk metrics calculation
        high_priority_actions = len([a for a in actions if a.priority == "HIGH"])
        total_rebalancing_amount = sum(a.dollar_amount for a in actions)
        
        return {
            'portfolio_volatility': adjusted_risk.adjusted_risk_score * 5.0,  # Simplified
            'rebalancing_risk': min(high_priority_actions * 2.0, 10.0),
            'implementation_risk': min(total_rebalancing_amount / 10000, 10.0),
            'diversification_score': max(100 - len(actions) * 5, 60)
        }
    
    def _generate_monitoring_recommendations(self, insights: FinancialContent,
                                          actions: List[RebalancingAction],
                                          adjusted_risk: AdjustedRiskProfile) -> List[str]:
        """Generate monitoring recommendations"""
        
        recommendations = [
            "Monitor portfolio performance daily for first week after rebalancing",
            f"Track {insights.market_sentiment.lower()} sentiment indicators",
            "Review sector allocation weekly for first month"
        ]
        
        # Add specific monitoring for mentioned tickers
        if insights.tickers_mentioned:
            recommendations.append(f"Monitor {', '.join(insights.tickers_mentioned[:3])} closely")
        
        # Add risk-specific monitoring
        if adjusted_risk.adjusted_risk_score > adjusted_risk.original_risk_score:
            recommendations.append("Monitor volatility closely due to increased risk exposure")
        
        return recommendations
    
    def _create_insight_summary(self, insights: FinancialContent, 
                              impact_analysis: Dict[str, Any]) -> str:
        """Create summary of how insights influenced rebalancing"""
        
        summary_parts = [
            f"Market sentiment: {insights.market_sentiment}",
            f"Confidence: {insights.confidence_score:.0f}%"
        ]
        
        if insights.sectors_mentioned:
            summary_parts.append(f"Key sectors: {', '.join(insights.sectors_mentioned[:3])}")
        
        if insights.tickers_mentioned:
            summary_parts.append(f"Mentioned tickers: {', '.join(insights.tickers_mentioned[:3])}")
        
        return "; ".join(summary_parts)
    
    def _determine_implementation_timeline(self, actions: List[RebalancingAction], rebalance_type: str = "market_insight") -> str:
        """Determine recommended implementation timeline based on rebalance type"""
        
        high_priority = len([a for a in actions if a.priority == "HIGH"])
        total_actions = len(actions)
        
        # Adjust timeline based on rebalance type
        if rebalance_type == "market_insight":
            # Market insight-driven rebalancing should be more urgent
            if high_priority >= 3:
                return "Implement high priority actions within 4-6 hours (market insight driven), others within 24 hours"
            elif total_actions >= 5:
                return "Spread implementation over 2-3 trading days (market responsive)"
            else:
                return "Implement within 6-12 hours (capitalize on market insights)"
        elif rebalance_type == "time_based":
            # Time-based rebalancing can be more gradual
            return "Implement gradually over 1-2 weeks (time-based rebalancing)"
        elif rebalance_type == "risk_adjustment":
            # Risk adjustments should be prompt but not rushed
            return "Implement within 2-3 trading days (risk management priority)"
        elif rebalance_type == "manual":
            # Manual rebalancing follows standard timeline
            if high_priority >= 3:
                return "Implement high priority actions within 24 hours, others within 1 week"
            elif total_actions >= 5:
                return "Spread implementation over 3-5 trading days"
            else:
                return "Implement within 1-2 trading days"
        else:
            # Default timeline
            return "Implement within 1-2 trading days (standard rebalancing)"

# ============================================================================
# FASTAPI ENDPOINTS
# ============================================================================

async def parse_url_endpoint(request: URLParseRequest, parser_service: URLContentParser) -> URLParseResponse:
    """Parse financial content from URL endpoint"""
    return await parser_service.parse_url_content(request)

async def rebalance_portfolio_endpoint(request: PortfolioRebalanceRequest, 
                                     rebalancer_service: AIPortfolioRebalancer) -> PortfolioRebalanceResponse:
    """AI-powered portfolio rebalancing endpoint"""
    return await rebalancer_service.rebalance_portfolio(request) 