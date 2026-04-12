"""
Live News Collector - Fetch latest news for real-time prediction context.
Used during prediction generation to provide current market context.
"""

import feedparser
import logging
import time
from datetime import datetime, timedelta
import pandas as pd
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LiveNewsCollector:
    """Fetch latest news articles for stocks (real-time, not stored)."""

    def __init__(self, max_age_hours: int = 24):
        """
        Args:
            max_age_hours: Only fetch articles from last N hours (default: 24h)
        """
        self.max_age_hours = max_age_hours
        self.sources = {
            "moneycontrol": "https://www.moneycontrol.com/rss/MCtopnews.xml",
            "economic_times": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
            "business_standard": "https://www.business-standard.com/rss/markets-106.rss",
            "reuters_business": "https://feeds.reuters.com/reuters/businessNews",
            "livemint_markets": "https://www.livemint.com/rss/markets",
            "rbi_press": "https://www.rbi.org.in/pressreleases/rss.aspx",
        }

    # ── Single stock news ──────────────────────────────────────────────────────

    def fetch_stock_specific_news(self, symbol: str, max_articles: int = 10) -> pd.DataFrame:
        """
        Fetch latest news specific to one stock symbol.
        
        Args:
            symbol: Stock symbol (e.g., "RELIANCE.NS")
            max_articles: Max articles to return
            
        Returns:
            DataFrame with columns: [title, link, published, summary, source, symbol, timestamp]
        """
        company = symbol.replace(".NS", "").replace(".BO", "").upper()
        
        try:
            # Yahoo Finance RSS for stock-specific news
            url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=IN&lang=en-IN"
            logger.info(f"Fetching latest news for {symbol} from Yahoo Finance...")
            
            feed = feedparser.parse(url)
            
            articles = []
            for entry in feed.entries[:max_articles]:
                articles.append({
                    "title": entry.get("title", ""),
                    "link": entry.get("link", ""),
                    "published": entry.get("published", ""),
                    "summary": entry.get("summary", ""),
                    "source": "yahoo_finance",
                    "symbol": symbol,
                    "timestamp": datetime.now(),
                })
            
            logger.info(f"✓ Fetched {len(articles)} articles for {symbol}")
            
            if articles:
                return pd.DataFrame(articles)
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.warning(f"✗ Error fetching Yahoo news for {symbol}: {e}")
            return pd.DataFrame()

    # ── General market news ───────────────────────────────────────────────────

    def fetch_market_news(self, keywords: Optional[list[str]] = None, max_articles: int = 20) -> pd.DataFrame:
        """
        Fetch general market/economic news that may affect stock predictions.
        
        Args:
            keywords: Filter articles by keywords (e.g., ["rate", "inflation", "earnings"])
            max_articles: Max articles to return
            
        Returns:
            DataFrame with general market news
        """
        all_articles = []
        
        for source_name, url in self.sources.items():
            try:
                logger.info(f"Fetching from {source_name}...")
                feed = feedparser.parse(url)
                
                for entry in feed.entries[:max_articles]:
                    title = entry.get("title", "").lower()
                    summary = entry.get("summary", "").lower()
                    
                    # Filter by keywords if provided
                    if keywords:
                        if not any(kw.lower() in title or kw.lower() in summary for kw in keywords):
                            continue
                    
                    all_articles.append({
                        "title": entry.get("title", ""),
                        "link": entry.get("link", ""),
                        "published": entry.get("published", ""),
                        "summary": entry.get("summary", ""),
                        "source": source_name,
                        "symbol": None,  # General news
                        "timestamp": datetime.now(),
                    })
                
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                logger.warning(f"Error fetching from {source_name}: {e}")
                continue
        
        logger.info(f"✓ Fetched {len(all_articles)} market news articles")
        
        if all_articles:
            return pd.DataFrame(all_articles)
        else:
            return pd.DataFrame()

    # ── Sector-specific news ───────────────────────────────────────────────────

    def fetch_sector_news(self, sector: str, max_articles: int = 15) -> pd.DataFrame:
        """
        Fetch news for a specific sector (Banking, IT, Auto, Pharma, etc).
        
        Args:
            sector: Sector name (e.g., "Banking", "IT")
            max_articles: Max articles
            
        Returns:
            DataFrame with sector news
        """
        sector_keywords = {
            "Banking": ["bank", "rbi", "npa", "mortgage", "credit", "rate"],
            "IT": ["tcs", "infosys", "cognizant", "wipro", "tech", "software"],
            "Auto": ["auto", "car", "vehicle", "ev", "tata", "maruti"],
            "Pharma": ["pharma", "drug", "medicine", "fda", "clinical"],
            "Finance": ["sector", "market", "index", "sensex", "nifty", "fund"],
        }
        
        keywords = sector_keywords.get(sector.title(), [sector.lower()])
        
        logger.info(f"Fetching {sector} sector news with keywords: {keywords}")
        return self.fetch_market_news(keywords=keywords, max_articles=max_articles)

    # ── Utility: Get context for predictions ─────────────────────────────────

    def get_prediction_context(self, symbol: str, include_market_news: bool = True) -> dict:
        """
        Get all relevant news context for a stock prediction.
        Returns: {
            "stock_news": DataFrame,
            "market_news": DataFrame,
            "digest": str (one-liner summary)
        }
        """
        stock_news = self.fetch_stock_specific_news(symbol, max_articles=10)
        
        market_news = pd.DataFrame()
        if include_market_news:
            market_news = self.fetch_market_news(keywords=["rate", "inflation", "market"], max_articles=10)
        
        # Generate digest
        digest = ""
        if not stock_news.empty:
            digest += f"{len(stock_news)} articles about {symbol.replace('.NS', '')}. "
        if not market_news.empty:
            digest += f"{len(market_news)} market-context articles."
        
        return {
            "stock_news": stock_news,
            "market_news": market_news,
            "digest": digest,
        }


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    collector = LiveNewsCollector()
    
    # Test: Fetch news for one stock
    print("\n=== Testing Stock-Specific News ===")
    df = collector.fetch_stock_specific_news("RELIANCE.NS", max_articles=5)
    if not df.empty:
        print(df[["title", "source", "timestamp"]].head())
    
    # Test: Fetch market news
    print("\n=== Testing Market News ===")
    df = collector.fetch_market_news(keywords=["rate", "inflation"], max_articles=5)
    if not df.empty:
        print(df[["title", "source"]].head())
    
    # Test: Get full context
    print("\n=== Testing Prediction Context ===")
    context = collector.get_prediction_context("INFY.NS")
    print(context["digest"])
    print(f"Stock news rows: {len(context['stock_news'])}")
    print(f"Market context rows: {len(context['market_news'])}")
