"""
AI Investment Advisor Agent - LangGraph-based multi-turn recommendation system.

This agent analyzes latest market news, economic indicators, and geopolitical factors
to provide personalized investment recommendations beyond the 5 tracked stocks.

Uses LangGraph for state management and tool orchestration with Groq LLM + NewsAPI.
"""

import logging
import os
import re
from typing import Annotated
from datetime import datetime, timedelta
import json
import hashlib
import time
import requests

from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Request Optimization & Caching ────────────────────────────────────────────

class RequestCache:
    """Simple in-memory cache for API responses with TTL and rate limiting."""
    
    def __init__(self, cache_ttl_minutes: int = 30, max_requests_per_hour: int = 20):
        self.cache = {}  # {query_hash: (result, timestamp)}
        self.cache_ttl = cache_ttl_minutes * 60  # Convert to seconds
        self.request_history = []  # List of request timestamps
        self.max_requests_per_hour = max_requests_per_hour
        self.last_request_time = 0
        self.min_request_interval = 3  # Minimum 3 seconds between API calls
    
    def _query_hash(self, query: str, profile: dict) -> str:
        """Generate hash of query + profile for caching."""
        query_str = f"{query}:{json.dumps(profile, sort_keys=True)}"
        return hashlib.md5(query_str.encode()).hexdigest()
    
    def get(self, query: str, profile: dict) -> dict | None:
        """Retrieve cached result if exists and not expired."""
        query_key = self._query_hash(query, profile)
        
        if query_key in self.cache:
            result, timestamp = self.cache[query_key]
            if time.time() - timestamp < self.cache_ttl:
                logger.info(f"⚡ Using cached recommendation (age: {time.time() - timestamp:.0f}s)")
                return result
            else:
                del self.cache[query_key]
                logger.info("Cache expired, fetching fresh recommendation")
        
        return None
    
    def set(self, query: str, profile: dict, result: dict) -> None:
        """Store result in cache."""
        query_key = self._query_hash(query, profile)
        self.cache[query_key] = (result, time.time())
    
    def can_make_request(self) -> tuple[bool, str]:
        """Check if request can be made (rate limiting)."""
        now = time.time()
        
        # Check minimum interval between requests
        if now - self.last_request_time < self.min_request_interval:
            wait_time = self.min_request_interval - (now - self.last_request_time)
            return False, f"Rate limited. Wait {wait_time:.1f}s before next request."
        
        # Clean old requests (older than 1 hour)
        hour_ago = now - 3600
        self.request_history = [t for t in self.request_history if t > hour_ago]
        
        # Check hourly limit
        if len(self.request_history) >= self.max_requests_per_hour:
            oldest_request = self.request_history[0]
            retry_after = oldest_request + 3600 - now
            msg = f"Hourly quota reached ({len(self.request_history)}/{self.max_requests_per_hour}). Retry after {retry_after:.0f}s."
            return False, msg
        
        return True, ""
    
    def record_request(self) -> None:
        """Record that a request was made."""
        self.last_request_time = time.time()
        self.request_history.append(self.last_request_time)
        remaining = self.max_requests_per_hour - len(self.request_history)
        logger.info(f"📊 API request made. Remaining quota: {remaining}/{self.max_requests_per_hour}")


# Global cache instance
_cache = RequestCache(cache_ttl_minutes=30, max_requests_per_hour=20)


# ── State Schema ───────────────────────────────────────────────────────────────

class AdvisorState(TypedDict):
    """State for investment advisor agent workflow."""
    
    messages: Annotated[list[BaseMessage], add_messages]
    user_query: str
    investment_profile: dict  # {risk_tolerance, portfolio_type, amount, etc}
    news_context: list[dict]  # Latest articles
    economic_context: dict  # Macro indicators
    sentiment_aggregate: dict  # Overall market sentiment
    recommendation: dict  # Final recommendation output
    confidence: float  # 0-1 confidence in recommendation
    status: str  # "initialized", "analyzing", "complete", "error"


# ── LLM Setup ──────────────────────────────────────────────────────────────────

def _get_groq_llm():
    """Initialize Groq LLM for reasoning."""
    # Environment variables already loaded at module level
    api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment")
    
    return ChatGroq(
        api_key=api_key,
        model="llama-3.1-8b-instant",  # Large, free model with good performance
        temperature=0.3,  # Lower = more deterministic reasoning
        max_tokens=1024,
    )


# ── Agent Nodes ────────────────────────────────────────────────────────────────

def initialize_node(state: AdvisorState) -> AdvisorState:
    """Initialize analysis with user query."""
    logger.info(f"Initializing analysis for query: {state['user_query']}")
    
    state["status"] = "initialized"
    state["messages"].append(
        HumanMessage(
            content=f"User Investment Query: {state['user_query']}\n\n"
                   f"Investment Profile: {state.get('investment_profile', {})}"
        )
    )
    return state


def _fetch_news_from_newsapi() -> list[dict]:
    """Fetch latest Indian market news from NewsAPI."""
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        logger.warning("NEWS_API_KEY not found, using placeholder data")
        return []
    
    try:
        # Fetch top Indian business/market news
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": "India stock market NSE BSE economy",
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 3,
            "apiKey": api_key
        }
        
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        if data.get("status") != "ok":
            logger.warning(f"NewsAPI error: {data.get('message')}")
            return []
        
        articles = []
        for article in data.get("articles", [])[:3]:
            articles.append({
                "title": article.get("title", "N/A")[:80],
                "description": article.get("description", "")[:100],
                "url": article.get("url", ""),
                "source": article.get("source", {}).get("name", "Unknown"),
                "publishedAt": article.get("publishedAt", "N/A"),
                "sentiment": "neutral",  # NewsAPI doesn't provide sentiment
                "impact": "medium"
            })
        
        logger.info(f"✅ Fetched {len(articles)} news articles from NewsAPI")
        return articles
        
    except requests.exceptions.RequestException as e:
        logger.warning(f"NewsAPI fetch failed: {e}, using fallback data")
        return []


def news_analysis_node(state: AdvisorState) -> AdvisorState:
    """Analyze latest news for context using NewsAPI."""
    logger.info("Analyzing latest news from NewsAPI...")
    
    # Try to fetch from NewsAPI first
    articles = _fetch_news_from_newsapi()
    
    # If NewsAPI fails or returns nothing, use fallback
    if not articles:
        logger.info("Using fallback news data")
        articles = [
            {
                "title": "RBI expected to cut rates next quarter",
                "description": "Reserve Bank may ease policy",
                "sentiment": "positive",
                "impact": "high",
                "source": "Fallback"
            },
            {
                "title": "IT exports surge amid global tech boom",
                "description": "Indian IT sector growing",
                "sentiment": "positive",
                "impact": "high",
                "source": "Fallback"
            },
        ]
    
    state["news_context"] = articles
    return state


def economic_analysis_node(state: AdvisorState) -> AdvisorState:
    """Fetch and analyze macro economic indicators."""
    logger.info("Analyzing economic indicators...")
    
    # TODO: Integrate with economic data sources (RBI, government stats)
    # For now, placeholder data
    state["economic_context"] = {
        "repo_rate": 6.5,
        "inflation_rate": 3.2,
        "gdp_growth": 6.2,
        "iip_growth": 2.1,
        "market_outlook": "cautiously_optimistic"
    }
    
    return state


def sentiment_aggregation_node(state: AdvisorState) -> AdvisorState:
    """Aggregate sentiment from news and economic data."""
    logger.info("Aggregating market sentiment...")
    
    positive_signals = sum(1 for n in state.get("news_context", []) if n.get("sentiment") == "positive")
    total_signals = len(state.get("news_context", []))
    
    sentiment_score = positive_signals / total_signals if total_signals > 0 else 0.5
    
    if sentiment_score > 0.65:
        label = "STRONGLY_POSITIVE"
    elif sentiment_score > 0.55:
        label = "POSITIVE"
    elif sentiment_score > 0.45:
        label = "NEUTRAL"
    elif sentiment_score > 0.35:
        label = "NEGATIVE"
    else:
        label = "STRONGLY_NEGATIVE"
    
    state["sentiment_aggregate"] = {
        "label": label,
        "score": sentiment_score,
        "positive_signals": positive_signals,
        "total_signals": total_signals,
    }
    
    return state


def recommendation_reasoning_node(state: AdvisorState) -> AdvisorState:
    """Generate intelligent investment recommendations using Groq."""
    logger.info("Generating AI-powered investment recommendations...")
    
    llm = _get_groq_llm()
    
    # Build comprehensive context for the LLM
    sentiment_str = state['sentiment_aggregate']
    news_articles = state.get("news_context", [])
    
    news_context_text = "\n".join([
        f"• {article.get('title', 'N/A')} - Source: {article.get('source', 'Unknown')}"
        for article in news_articles[:3]
    ]) if news_articles else "No recent news available."
    
    economic_data = state.get("economic_context", {})
    economic_text = "\n".join([
        f"• {key}: {value}"
        for key, value in economic_data.items()
    ]) if economic_data else "No economic data available."
    
    profile = state.get('investment_profile', {})
    
    # Detailed prompt for intelligent response
    prompt = f"""You are an expert financial advisor. Provide detailed, practical investment advice based on this information:

USER'S INVESTMENT QUESTION:
{state['user_query']}

INVESTOR PROFILE:
• Risk Tolerance: {profile.get('risk_tolerance', 'Not specified')}
• Investment Amount: {profile.get('amount', 'Not specified')}
• Time Horizon: {profile.get('time_horizon', 'Not specified')}
• Portfolio Type: {profile.get('portfolio_type', 'Not specified')}

CURRENT MARKET CONDITIONS:
• Market Sentiment: {sentiment_str.get('label', 'NEUTRAL')} (Confidence: {sentiment_str.get('score', 0.5):.0%})

RECENT NEWS & DEVELOPMENTS:
{news_context_text}

ECONOMIC INDICATORS:
{economic_text}

PROVIDE YOUR RECOMMENDATION as a detailed response with:
1. Overall Investment Strategy - Your main recommendation and why
2. Recommended Sectors/Assets - Specific areas to focus on based on the query
3. Key Considerations - Important factors to keep in mind given current market conditions
4. Action Items - Specific steps they should take
5. Risk Management - How to protect their investment

Be specific, contextual, and practical. Avoid generic or templated responses. Base your answer directly on the user's question and current market data provided above."""
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        response_text = response.content.strip()
        
        # Store the natural language response directly
        state["recommendation"] = {
            "advice": response_text,
            "status": "success"
        }
        state["confidence"] = 0.85  # High confidence in LLM response
        state["messages"].append(AIMessage(content=response_text))
        _cache.record_request()  # Track successful request for rate limiting
        
        logger.info("✅ Investment recommendation generated successfully")
        
    except Exception as e:
        error_str = str(e)
        logger.error(f"❌ Recommendation generation failed: {e}")
        
        # Graceful error handling
        if "429" in error_str or "quota" in error_str.lower():
            error_message = "API rate limit exceeded. Please try again in a few moments."
        else:
            error_message = f"Could not generate recommendation due to: {error_str[:100]}"
        
        state["recommendation"] = {
            "advice": f"I apologize, I was unable to generate a personalized recommendation at this time: {error_message}",
            "status": "error"
        }
        state["confidence"] = 0.0
    
    return state


def summary_node(state: AdvisorState) -> AdvisorState:
    """Finalize and prepare recommendation for display."""
    logger.info("Finalizing investment recommendation...")
    
    rec = state.get("recommendation", {})
    
    # Simply display the natural language advice from the LLM
    advice_text = rec.get('advice', 'Unable to generate recommendation.')
    
    state["status"] = "complete"
    state["messages"].append(AIMessage(content=advice_text))
    
    return state
    
    return state


# ── Build Graph ────────────────────────────────────────────────────────────

def build_investment_advisor_graph():
    """Build LangGraph workflow for investment advisor."""
    
    graph = StateGraph(AdvisorState)
    
    # Add nodes
    graph.add_node("initialize", initialize_node)
    graph.add_node("news_analysis", news_analysis_node)
    graph.add_node("economic_analysis", economic_analysis_node)
    graph.add_node("sentiment_aggregation", sentiment_aggregation_node)
    graph.add_node("generate_recommendation", recommendation_reasoning_node)
    graph.add_node("summary", summary_node)
    
    # Add edges (workflow)
    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", "news_analysis")
    graph.add_edge("news_analysis", "economic_analysis")
    graph.add_edge("economic_analysis", "sentiment_aggregation")
    graph.add_edge("sentiment_aggregation", "generate_recommendation")
    graph.add_edge("generate_recommendation", "summary")
    graph.add_edge("summary", END)
    
    return graph.compile()


# ── Main API ───────────────────────────────────────────────────────────────

def get_investment_advice(
    user_query: str,
    investment_profile: dict = None,
    include_sectors: list[str] = None,
) -> dict:
    """
    Get investment advice from the advisor agent.
    
    Args:
        user_query: User's investment question (e.g., "Where should I invest ₹1L in tech?")
        investment_profile: Dict with {risk_tolerance, portfolio_type, amount, etc}
        include_sectors: Sectors to focus on (Banking, IT, Auto, etc)
    
    Returns:
        {
            "recommendation": str,
            "target_sectors": list[str],
            "risk_level": str,
            "confidence": float,
            "key_drivers": list[str],
            "suggested_actions": list[str],
            "macro_context": str,
            "timestamp": str
        }
    """
    
    if investment_profile is None:
        investment_profile = {
            "risk_tolerance": "medium",
            "portfolio_type": "mixed",
            "time_horizon": "1-3 years",
        }
    
    # Initialize state
    initial_state = AdvisorState(
        messages=[],
        user_query=user_query,
        investment_profile=investment_profile,
        news_context=[],
        economic_context={},
        sentiment_aggregate={},
        recommendation={},
        confidence=0.0,
        status="pending",
    )
    
    # Run graph
    graph = build_investment_advisor_graph()
    final_state = graph.invoke(initial_state)
    
    # Format output
    return {
        **final_state["recommendation"],
        "timestamp": datetime.now().isoformat(),
        "status": final_state["status"],
        "confidence": final_state["confidence"],
    }


# ── CLI Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Test query
    advice = get_investment_advice(
        user_query="I have ₹2 lakhs and want to invest in IT sector for 2-3 years. What should I do?",
        investment_profile={
            "risk_tolerance": "medium",
            "portfolio_type": "growth",
            "amount": 200000,
            "time_horizon": "2-3 years"
        }
    )
    
    print("\n=== INVESTMENT ADVICE ===")
    print(json.dumps(advice, indent=2))
