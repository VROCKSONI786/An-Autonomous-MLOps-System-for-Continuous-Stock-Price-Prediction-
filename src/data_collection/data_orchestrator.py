import yaml
import pandas as pd
from datetime import datetime
import logging
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_collection.yfinance_collector import YFinanceCollector
from data_collection.screener_collector import ScreenerCollector
from data_collection.news_collector import NewsCollector
from data_collection.sentiment_analyzer import SentimentAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataOrchestrator:
    """Orchestrate all data collection activities"""
    
    def __init__(self, config_path=None):
        # Find config file relative to project root
        if config_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            config_path = os.path.join(project_root, "config", "config.yaml")
        
        logger.info(f"Loading config from: {config_path}")
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.symbols = self.config['data_collection']['stock_symbols']
        self.period = self.config['data_collection']['period']
        self.interval = self.config['data_collection']['interval']
        
        # Set data directories
        self.data_dir = os.path.join(project_root, "data", "raw")
        os.makedirs(self.data_dir, exist_ok=True)
    
    def collect_all_data(self):
        """Collect data from all sources"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info("=== Starting Data Collection ===")
        
        # 1. Stock Price Data
        logger.info("Step 1: Collecting stock price data...")
        yf_collector = YFinanceCollector(self.symbols, self.period, self.interval)
        price_data = yf_collector.fetch_all_stocks()
        
        if price_data is not None:
            price_file = f"stock_prices_{timestamp}.csv"
            yf_collector.save_data(price_data, price_file)
            logger.info(f"✓ Stock prices collected: {len(price_data)} records")
        
        # 2. Fundamental Data
        logger.info("Step 2: Collecting fundamental data...")
        screener_collector = ScreenerCollector()
        fundamental_data = screener_collector.fetch_all_financials(self.symbols)
        
        if fundamental_data is not None:
            fund_file = os.path.join(self.data_dir, f"fundamentals_{timestamp}.csv")
            fundamental_data.to_csv(fund_file, index=False)
            logger.info(f"✓ Fundamentals collected: {len(fundamental_data)} records")
        
        # 3. News Data
        logger.info("Step 3: Collecting news articles...")
        news_collector = NewsCollector()
        news_data = news_collector.collect_all_news(self.symbols)
        
        if news_data is not None:
            news_file = f"news_{timestamp}.csv"
            news_collector.save_news(news_data, news_file)
            logger.info(f"✓ News collected: {len(news_data)} articles")
            
            # 4. Sentiment Analysis (limit to first 5 articles for speed)
            logger.info("Step 4: Analyzing sentiment...")
            sentiment_analyzer = SentimentAnalyzer()
            sentiment_data = sentiment_analyzer.analyze_bulk_sentiment(news_data.head(5))
            
            sent_file = os.path.join(self.data_dir, f"sentiment_{timestamp}.csv")
            sentiment_data.to_csv(sent_file, index=False)
            logger.info(f"✓ Sentiment analyzed: {len(sentiment_data)} articles")
            
            # Market Summary
            summary = sentiment_analyzer.get_market_sentiment_summary(sentiment_data)
            logger.info(f"\n=== Market Sentiment Summary ===\n{summary}\n")
        else:
            sentiment_data = None
        
        logger.info("=== Data Collection Complete ===")
        
        return {
            'price_data': price_data,
            'fundamental_data': fundamental_data,
            'news_data': news_data,
            'sentiment_data': sentiment_data
        }

if __name__ == "__main__":
    orchestrator = DataOrchestrator()
    data = orchestrator.collect_all_data()