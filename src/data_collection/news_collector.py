import requests
import feedparser
from datetime import datetime, timedelta
import pandas as pd
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NewsCollector:
    """Collect news from multiple sources"""
    
    def __init__(self):
        self.sources = {
            'moneycontrol': 'https://www.moneycontrol.com/rss/latestnews.xml',
            'economic_times': 'https://economictimes.indiatimes.com/rssfeedstopstories.cms',
            'business_standard': 'https://www.business-standard.com/rss/home_page_top_stories.rss',
        }
        
    def fetch_rss_news(self, source_name, url, max_articles=20):
        """Fetch news from RSS feed"""
        try:
            logger.info(f"Fetching news from {source_name}")
            feed = feedparser.parse(url)
            
            articles = []
            for entry in feed.entries[:max_articles]:
                article = {
                    'source': source_name,
                    'title': entry.get('title', ''),
                    'link': entry.get('link', ''),
                    'published': entry.get('published', ''),
                    'summary': entry.get('summary', ''),
                    'timestamp': datetime.now()
                }
                articles.append(article)
            
            logger.info(f"Fetched {len(articles)} articles from {source_name}")
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching from {source_name}: {str(e)}")
            return []
    
    def fetch_yahoo_finance_news(self, symbol, max_articles=10):
        """Fetch news specific to a stock from Yahoo Finance"""
        try:
            company_code = symbol.replace('.NS', '').replace('.BO', '')
            url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=IN&lang=en-IN"
            
            logger.info(f"Fetching Yahoo Finance news for {symbol}")
            feed = feedparser.parse(url)
            
            articles = []
            for entry in feed.entries[:max_articles]:
                article = {
                    'source': 'yahoo_finance',
                    'symbol': symbol,
                    'title': entry.get('title', ''),
                    'link': entry.get('link', ''),
                    'published': entry.get('published', ''),
                    'summary': entry.get('summary', ''),
                    'timestamp': datetime.now()
                }
                articles.append(article)
            
            logger.info(f"Fetched {len(articles)} articles for {symbol}")
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching Yahoo news for {symbol}: {str(e)}")
            return []
    
    def collect_all_news(self, symbols=None):
        """Collect news from all sources"""
        all_articles = []
        
        # General market news
        for source_name, url in self.sources.items():
            articles = self.fetch_rss_news(source_name, url)
            all_articles.extend(articles)
            time.sleep(1)  # Rate limiting
        
        # Stock-specific news
        if symbols:
            for symbol in symbols:
                articles = self.fetch_yahoo_finance_news(symbol)
                all_articles.extend(articles)
                time.sleep(1)
        
        if all_articles:
            return pd.DataFrame(all_articles)
        return None
    
    def save_news(self, news_df, filename="news_data.csv"):
        """Save news data"""
        import os
        try:
            filepath = os.path.join("data", "raw", filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            news_df.to_csv(filepath, index=False)
            logger.info(f"News saved to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving news: {str(e)}")
            return None

if __name__ == "__main__":
    symbols = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
    collector = NewsCollector()
    
    news = collector.collect_all_news(symbols)
    if news is not None:
        print(f"Collected {len(news)} news articles")
        print(news.head())
        collector.save_news(news)