"""
Finnhub Recommendation Trends Integration
Enhances unified strategy with analyst recommendations
"""

import aiohttp
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

class FinnhubRecommendationService:
    """Service for fetching and analyzing Finnhub recommendation trends"""
    
    def __init__(self, finnhub_api_key: str):
        self.finnhub_key = finnhub_api_key
        self.base_url = "https://finnhub.io/api/v1"
    
    async def get_recommendation_trends(self, symbol: str) -> Dict[str, Any]:
        """Get latest analyst recommendation trends for a symbol"""
        url = f"{self.base_url}/stock/recommendation"
        params = {
            "symbol": symbol,
            "token": self.finnhub_key
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._process_recommendation_data(symbol, data)
                    else:
                        logger.error(f"Finnhub recommendation API error for {symbol}: HTTP {response.status}")
                        return self._create_fallback_recommendation(symbol, f"API error: HTTP {response.status}")
                        
        except Exception as e:
            logger.error(f"Failed to fetch recommendations for {symbol}: {e}")
            return self._create_fallback_recommendation(symbol, f"Network error: {str(e)}")
    
    async def get_bulk_recommendations(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get recommendation trends for multiple symbols concurrently"""
        tasks = [self.get_recommendation_trends(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        recommendations = {}
        for i, result in enumerate(results):
            symbol = symbols[i]
            if isinstance(result, Exception):
                recommendations[symbol] = self._create_fallback_recommendation(symbol, str(result))
            else:
                recommendations[symbol] = result
        
        return recommendations
    
    def _process_recommendation_data(self, symbol: str, data: List[Dict]) -> Dict[str, Any]:
        """Process raw Finnhub recommendation data"""
        if not data:
            return self._create_fallback_recommendation(symbol, "No recommendation data available")
        
        # Get the most recent recommendation
        latest = data[0] if data else {}
        
        # Calculate recommendation scores and trends
        buy_count = latest.get("buy", 0)
        hold_count = latest.get("hold", 0)
        sell_count = latest.get("sell", 0)
        strong_buy_count = latest.get("strongBuy", 0)
        strong_sell_count = latest.get("strongSell", 0)
        
        total_recommendations = buy_count + hold_count + sell_count + strong_buy_count + strong_sell_count
        
        if total_recommendations == 0:
            return self._create_fallback_recommendation(symbol, "No analyst recommendations")
        
        # Calculate weighted recommendation score (1-5 scale)
        # 5 = Strong Buy, 4 = Buy, 3 = Hold, 2 = Sell, 1 = Strong Sell
        weighted_score = (
            (strong_buy_count * 5) + 
            (buy_count * 4) + 
            (hold_count * 3) + 
            (sell_count * 2) + 
            (strong_sell_count * 1)
        ) / total_recommendations
        
        # Determine overall recommendation
        overall_recommendation = self._get_recommendation_text(weighted_score)
        
        # Calculate confidence based on number of analysts and consensus
        confidence = self._calculate_recommendation_confidence(
            total_recommendations, weighted_score, latest
        )
        
        # Analyze trend if we have historical data
        trend_analysis = self._analyze_recommendation_trend(data)
        
        return {
            "symbol": symbol,
            "available": True,
            "latest_data": {
                "period": latest.get("period", "Unknown"),
                "strong_buy": strong_buy_count,
                "buy": buy_count,
                "hold": hold_count,
                "sell": sell_count,
                "strong_sell": strong_sell_count,
                "total_analysts": total_recommendations
            },
            "analysis": {
                "weighted_score": round(weighted_score, 2),
                "overall_recommendation": overall_recommendation,
                "confidence": confidence,
                "consensus_strength": self._get_consensus_strength(weighted_score),
                "recommendation_rationale": self._generate_recommendation_rationale(
                    weighted_score, total_recommendations, latest
                )
            },
            "trend_analysis": trend_analysis,
            "impact_assessment": self._assess_investment_impact(weighted_score, confidence),
            "timestamp": datetime.now().isoformat(),
            "source": "finnhub_analyst_recommendations"
        }
    
    def _get_recommendation_text(self, weighted_score: float) -> str:
        """Convert weighted score to recommendation text"""
        if weighted_score >= 4.5:
            return "Strong Buy"
        elif weighted_score >= 3.5:
            return "Buy"
        elif weighted_score >= 2.5:
            return "Hold"
        elif weighted_score >= 1.5:
            return "Sell"
        else:
            return "Strong Sell"
    
    def _calculate_recommendation_confidence(self, total_analysts: int, weighted_score: float, latest: Dict) -> float:
        """Calculate confidence in the recommendation"""
        # Base confidence on number of analysts
        analyst_confidence = min(total_analysts / 10, 1.0)  # Max confidence at 10+ analysts
        
        # Adjust for consensus strength
        consensus_factor = 1.0
        if weighted_score > 4.0 or weighted_score < 2.0:
            consensus_factor = 1.2  # Higher confidence for strong consensus
        elif 2.5 <= weighted_score <= 3.5:
            consensus_factor = 0.8  # Lower confidence for neutral recommendations
        
        # Consider recency (if available)
        recency_factor = 1.0
        period = latest.get("period", "")
        if period:
            try:
                period_date = datetime.strptime(period, "%Y-%m-%d")
                days_old = (datetime.now() - period_date).days
                if days_old > 90:
                    recency_factor = 0.9  # Slightly lower confidence for old data
            except:
                pass
        
        confidence = analyst_confidence * consensus_factor * recency_factor
        return min(round(confidence * 100, 1), 100.0)
    
    def _get_consensus_strength(self, weighted_score: float) -> str:
        """Determine consensus strength"""
        if weighted_score >= 4.5 or weighted_score <= 1.5:
            return "Strong Consensus"
        elif weighted_score >= 4.0 or weighted_score <= 2.0:
            return "Moderate Consensus"
        elif 2.8 <= weighted_score <= 3.2:
            return "Neutral/Mixed"
        else:
            return "Weak Consensus"
    
    def _analyze_recommendation_trend(self, data: List[Dict]) -> Dict[str, Any]:
        """Analyze recommendation trends over time"""
        if len(data) < 2:
            return {
                "trend_available": False,
                "message": "Insufficient historical data for trend analysis"
            }
        
        # Compare latest vs previous recommendations
        latest = data[0]
        previous = data[1]
        
        latest_score = self._calculate_weighted_score(latest)
        previous_score = self._calculate_weighted_score(previous)
        
        score_change = latest_score - previous_score
        
        if abs(score_change) < 0.2:
            trend = "Stable"
        elif score_change > 0:
            trend = "Improving" if score_change > 0.5 else "Slightly Improving"
        else:
            trend = "Declining" if score_change < -0.5 else "Slightly Declining"
        
        return {
            "trend_available": True,
            "trend_direction": trend,
            "score_change": round(score_change, 2),
            "latest_period": latest.get("period", "Unknown"),
            "previous_period": previous.get("period", "Unknown"),
            "trend_strength": "Strong" if abs(score_change) > 0.5 else "Moderate" if abs(score_change) > 0.2 else "Weak"
        }
    
    def _calculate_weighted_score(self, recommendation_data: Dict) -> float:
        """Calculate weighted score for a single recommendation period"""
        buy_count = recommendation_data.get("buy", 0)
        hold_count = recommendation_data.get("hold", 0)
        sell_count = recommendation_data.get("sell", 0)
        strong_buy_count = recommendation_data.get("strongBuy", 0)
        strong_sell_count = recommendation_data.get("strongSell", 0)
        
        total = buy_count + hold_count + sell_count + strong_buy_count + strong_sell_count
        
        if total == 0:
            return 3.0  # Neutral if no data
        
        return (
            (strong_buy_count * 5) + 
            (buy_count * 4) + 
            (hold_count * 3) + 
            (sell_count * 2) + 
            (strong_sell_count * 1)
        ) / total
    
    def _generate_recommendation_rationale(self, weighted_score: float, total_analysts: int, latest: Dict) -> str:
        """Generate human-readable rationale for the recommendation"""
        recommendation = self._get_recommendation_text(weighted_score)
        
        if total_analysts < 3:
            coverage_note = "Limited analyst coverage"
        elif total_analysts < 7:
            coverage_note = "Moderate analyst coverage"
        else:
            coverage_note = "Strong analyst coverage"
        
        if weighted_score >= 4.0:
            sentiment = "positive analyst sentiment with most analysts recommending buying"
        elif weighted_score >= 3.0:
            sentiment = "neutral to positive sentiment with mixed buy/hold recommendations"
        elif weighted_score >= 2.0:
            sentiment = "cautious sentiment with mixed hold/sell recommendations"
        else:
            sentiment = "negative sentiment with most analysts recommending selling"
        
        return f"{recommendation} based on {coverage_note} ({total_analysts} analysts) showing {sentiment}."
    
    def _assess_investment_impact(self, weighted_score: float, confidence: float) -> Dict[str, Any]:
        """Assess how recommendations should impact investment decisions"""
        
        # Determine allocation impact
        if weighted_score >= 4.0 and confidence >= 70:
            allocation_suggestion = "Consider increasing allocation"
            risk_adjustment = "Lower perceived risk due to strong analyst support"
        elif weighted_score >= 3.5 and confidence >= 60:
            allocation_suggestion = "Maintain or slightly increase allocation"
            risk_adjustment = "Neutral risk adjustment"
        elif weighted_score <= 2.0 and confidence >= 70:
            allocation_suggestion = "Consider reducing allocation"
            risk_adjustment = "Higher perceived risk due to negative analyst sentiment"
        elif weighted_score <= 2.5 and confidence >= 60:
            allocation_suggestion = "Consider maintaining or reducing allocation"
            risk_adjustment = "Slight risk increase"
        else:
            allocation_suggestion = "No strong directional signal"
            risk_adjustment = "Neutral - low confidence in recommendations"
        
        # Determine investment urgency
        if confidence >= 80:
            urgency = "High confidence - consider acting on recommendation"
        elif confidence >= 60:
            urgency = "Moderate confidence - consider with other factors"
        else:
            urgency = "Low confidence - use as secondary factor only"
        
        return {
            "allocation_suggestion": allocation_suggestion,
            "risk_adjustment": risk_adjustment,
            "investment_urgency": urgency,
            "recommendation_weight": "Primary" if confidence >= 75 else "Secondary" if confidence >= 50 else "Minimal"
        }
    
    def _create_fallback_recommendation(self, symbol: str, error_message: str) -> Dict[str, Any]:
        """Create fallback recommendation when data is unavailable"""
        return {
            "symbol": symbol,
            "available": False,
            "error": error_message,
            "analysis": {
                "weighted_score": 3.0,  # Neutral
                "overall_recommendation": "Hold",
                "confidence": 0.0,
                "consensus_strength": "No Data",
                "recommendation_rationale": f"No analyst recommendation data available for {symbol}. {error_message}"
            },
            "trend_analysis": {
                "trend_available": False,
                "message": "No trend data available"
            },
            "impact_assessment": {
                "allocation_suggestion": "Use other analysis methods",
                "risk_adjustment": "Neutral - no analyst data",
                "investment_urgency": "No signal from analysts",
                "recommendation_weight": "Not applicable"
            },
            "timestamp": datetime.now().isoformat(),
            "source": "fallback_no_analyst_data"
        }

# ============================================================================
# INTEGRATION HELPER FUNCTIONS
# ============================================================================

def integrate_analyst_recommendations_into_strategy(
    stock_recommendations: Dict[str, List[str]], 
    analyst_recommendations: Dict[str, Dict[str, Any]],
    risk_score: int
) -> Dict[str, Any]:
    """Integrate analyst recommendations into investment strategy"""
    
    enhanced_recommendations = {}
    overall_analyst_sentiment = []
    high_confidence_signals = []
    
    for category, symbols in stock_recommendations.items():
        enhanced_recommendations[category] = []
        
        for symbol in symbols:
            analyst_data = analyst_recommendations.get(symbol, {})
            
            if analyst_data.get("available", False):
                analysis = analyst_data.get("analysis", {})
                weighted_score = analysis.get("weighted_score", 3.0)
                confidence = analysis.get("confidence", 0.0)
                
                # Adjust recommendation based on analyst sentiment
                adjusted_allocation = _calculate_analyst_adjusted_allocation(
                    weighted_score, confidence, risk_score
                )
                
                symbol_recommendation = {
                    "symbol": symbol,
                    "base_category": category,
                    "analyst_score": weighted_score,
                    "analyst_recommendation": analysis.get("overall_recommendation", "Hold"),
                    "analyst_confidence": confidence,
                    "adjusted_allocation_factor": adjusted_allocation,
                    "analyst_rationale": analysis.get("recommendation_rationale", ""),
                    "impact_assessment": analyst_data.get("impact_assessment", {})
                }
                
                enhanced_recommendations[category].append(symbol_recommendation)
                overall_analyst_sentiment.append(weighted_score)
                
                if confidence >= 75:
                    high_confidence_signals.append({
                        "symbol": symbol,
                        "recommendation": analysis.get("overall_recommendation"),
                        "confidence": confidence
                    })
            else:
                # No analyst data available
                symbol_recommendation = {
                    "symbol": symbol,
                    "base_category": category,
                    "analyst_score": 3.0,
                    "analyst_recommendation": "No Data",
                    "analyst_confidence": 0.0,
                    "adjusted_allocation_factor": 1.0,
                    "analyst_rationale": "No analyst recommendation data available",
                    "impact_assessment": {
                        "allocation_suggestion": "Use technical/fundamental analysis",
                        "risk_adjustment": "Neutral",
                        "investment_urgency": "No analyst signal",
                        "recommendation_weight": "Not applicable"
                    }
                }
                enhanced_recommendations[category].append(symbol_recommendation)
    
    # Calculate overall analyst sentiment
    avg_analyst_sentiment = sum(overall_analyst_sentiment) / len(overall_analyst_sentiment) if overall_analyst_sentiment else 3.0
    
    market_analyst_outlook = "Bullish" if avg_analyst_sentiment >= 3.7 else "Bearish" if avg_analyst_sentiment <= 2.3 else "Neutral"
    
    return {
        "enhanced_recommendations": enhanced_recommendations,
        "analyst_summary": {
            "average_analyst_score": round(avg_analyst_sentiment, 2),
            "market_analyst_outlook": market_analyst_outlook,
            "high_confidence_signals": high_confidence_signals,
            "symbols_with_analyst_data": len([s for s in overall_analyst_sentiment if s != 3.0]),
            "total_symbols_analyzed": sum(len(symbols) for symbols in stock_recommendations.values())
        },
        "strategy_adjustments": {
            "overall_risk_adjustment": _calculate_overall_risk_adjustment(avg_analyst_sentiment, risk_score),
            "allocation_confidence": _calculate_allocation_confidence(high_confidence_signals, overall_analyst_sentiment),
            "recommended_review_frequency": _determine_review_frequency(high_confidence_signals, risk_score)
        }
    }

def _calculate_analyst_adjusted_allocation(weighted_score: float, confidence: float, risk_score: int) -> float:
    """Calculate allocation adjustment factor based on analyst recommendations"""
    
    # Base adjustment on analyst score
    if weighted_score >= 4.5:
        base_factor = 1.3  # Increase allocation by 30%
    elif weighted_score >= 4.0:
        base_factor = 1.15  # Increase by 15%
    elif weighted_score >= 3.5:
        base_factor = 1.05  # Slight increase
    elif weighted_score <= 1.5:
        base_factor = 0.5   # Reduce allocation by 50%
    elif weighted_score <= 2.0:
        base_factor = 0.75  # Reduce by 25%
    elif weighted_score <= 2.5:
        base_factor = 0.9   # Slight reduction
    else:
        base_factor = 1.0   # No change
    
    # Adjust based on confidence
    confidence_adjustment = 1.0 + ((confidence - 50) / 100) * 0.3  # Max 30% adjustment
    confidence_adjustment = max(0.7, min(1.3, confidence_adjustment))
    
    # Adjust based on risk score (higher risk = more responsive to analyst recommendations)
    risk_adjustment = 1.0 + (risk_score - 3) * 0.1  # Range: 0.8 to 1.2
    
    final_factor = base_factor * confidence_adjustment * risk_adjustment
    return max(0.3, min(2.0, final_factor))  # Cap between 30% and 200%

def _calculate_overall_risk_adjustment(avg_analyst_sentiment: float, risk_score: int) -> str:
    """Calculate overall risk adjustment based on analyst sentiment"""
    
    if avg_analyst_sentiment >= 4.0:
        base_adjustment = "Reduce risk perception - strong analyst support"
    elif avg_analyst_sentiment >= 3.5:
        base_adjustment = "Slightly reduce risk perception"
    elif avg_analyst_sentiment <= 2.0:
        base_adjustment = "Increase risk perception - weak analyst support"
    elif avg_analyst_sentiment <= 2.5:
        base_adjustment = "Slightly increase risk perception"
    else:
        base_adjustment = "No risk adjustment - neutral analyst sentiment"
    
    # Consider user's risk tolerance
    if risk_score >= 4:
        return f"{base_adjustment} (High risk tolerance: act on analyst signals)"
    elif risk_score <= 2:
        return f"{base_adjustment} (Low risk tolerance: be cautious with changes)"
    else:
        return base_adjustment

def _calculate_allocation_confidence(high_confidence_signals: List[Dict], overall_sentiment: List[float]) -> str:
    """Calculate confidence in allocation recommendations"""
    
    high_conf_count = len(high_confidence_signals)
    total_signals = len(overall_sentiment)
    
    if total_signals == 0:
        return "No analyst data available"
    
    high_conf_ratio = high_conf_count / total_signals
    
    if high_conf_ratio >= 0.7:
        return "High confidence - strong analyst consensus"
    elif high_conf_ratio >= 0.4:
        return "Moderate confidence - mixed analyst signals"
    else:
        return "Low confidence - limited high-quality analyst data"

def _determine_review_frequency(high_confidence_signals: List[Dict], risk_score: int) -> str:
    """Determine how frequently to review based on analyst activity"""
    
    high_conf_count = len(high_confidence_signals)
    
    if high_conf_count >= 3:
        base_frequency = "Weekly"
    elif high_conf_count >= 1:
        base_frequency = "Bi-weekly"
    else:
        base_frequency = "Monthly"
    
    # Adjust for risk score
    if risk_score >= 4:
        return f"{base_frequency} (High risk: monitor closely)"
    elif risk_score <= 2:
        return f"{base_frequency} (Low risk: stable monitoring)"
    else:
        return base_frequency
