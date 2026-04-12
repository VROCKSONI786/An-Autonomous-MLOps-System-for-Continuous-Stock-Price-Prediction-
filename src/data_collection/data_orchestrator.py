import yaml
import pandas as pd
from datetime import datetime
import logging
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_collection.yfinance_collector import YFinanceCollector
from data_collection.screener_collector import ScreenerCollector
from data_collection.news_collector import NewsCollector
from data_collection.sentiment_analyzer import SentimentAnalyzer
from config_manager import ConfigManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _project_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(here))


class DataOrchestrator:
    """Orchestrate all data collection activities with support for custom stocks."""

    def __init__(self, config_path: str | None = None, use_custom_stocks: bool = True):
        root = _project_root()
        
        # Use ConfigManager for unified config handling
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.get_config(include_custom_stocks=use_custom_stocks)
        
        self.symbols  = self.config["data_collection"]["stock_symbols"]
        self.period   = self.config["data_collection"]["period"]
        self.interval = self.config["data_collection"]["interval"]

        # Always save raw data under project_root/data/raw/
        self.data_dir = os.path.join(root, "data", "raw")
        os.makedirs(self.data_dir, exist_ok=True)
        
        logger.info(f"DataOrchestrator initialized with {len(self.symbols)} stocks: {', '.join([s.replace('.NS', '') for s in self.symbols[:5]])}{'...' if len(self.symbols) > 5 else ''}")

    def collect_all_data(self) -> dict:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info("=== Starting Data Collection ===")
        
        sentiment_summary = None

        # ── 1. Stock prices ───────────────────────────────────────────────────
        logger.info("Step 1: Collecting stock price data…")
        yf = YFinanceCollector(self.symbols, self.period, self.interval)
        price_data = yf.fetch_all_stocks()

        if price_data is not None:
            price_file = os.path.join(self.data_dir, f"stock_prices_{timestamp}.csv")
            price_data.to_csv(price_file)
            logger.info(f"✓ Stock prices: {len(price_data)} records → {price_file}")
        else:
            logger.warning("✗ Stock price collection failed.")

        # ── 2. Fundamentals ───────────────────────────────────────────────────
        logger.info("Step 2: Collecting fundamental data…")
        screener = ScreenerCollector()
        fundamental_data = screener.fetch_all_financials(self.symbols)

        if fundamental_data is not None:
            fund_file = os.path.join(self.data_dir, f"fundamentals_{timestamp}.csv")
            fundamental_data.to_csv(fund_file, index=False)
            logger.info(f"✓ Fundamentals: {len(fundamental_data)} records → {fund_file}")
        else:
            logger.warning("✗ Fundamentals collection failed (screener.in may be blocked).")

        # ── 3. News ───────────────────────────────────────────────────────────
        logger.info("Step 3: Collecting news articles…")
        news_collector = NewsCollector()
        news_data = news_collector.collect_all_news(self.symbols)

        sentiment_data = None

        if news_data is not None and not news_data.empty:
            news_file = os.path.join(self.data_dir, f"news_{timestamp}.csv")
            news_data.to_csv(news_file, index=False)
            logger.info(f"✓ News: {len(news_data)} articles → {news_file}")

            # ── 4. Sentiment ──────────────────────────────────────────────────
            logger.info("Step 4: Analyzing sentiment…")
            logger.info(
                "  NOTE: Free-tier Gemini allows 15 req/min. "
                "Each article takes ~5s. Analyzing top 10 articles."
            )

            try:
                # min_delay=5s → 12 req/min, safely under 15 RPM limit
                analyzer = SentimentAnalyzer(min_delay_seconds=5.0, max_retries=2)
                sentiment_data = analyzer.analyze_bulk_sentiment(
                    news_data,
                    max_articles=10,      # conservative for free tier
                )

                sent_file = os.path.join(self.data_dir, f"sentiment_{timestamp}.csv")
                sentiment_data.to_csv(sent_file, index=False)
                logger.info(f"✓ Sentiment: {len(sentiment_data)} articles → {sent_file}")

                # Market summary
                sentiment_summary = analyzer.get_market_sentiment_summary(sentiment_data)
                logger.info(f"\n=== Market Sentiment Summary ===\n{sentiment_summary}\n")

            except Exception as e:
                logger.error(f"Sentiment analysis failed: {e}")
                sentiment_data = None
        else:
            logger.warning("✗ No news collected — skipping sentiment.")

        logger.info("=== Data Collection Complete ===")

        return {
            "price_data":           price_data,
            "fundamental_data":     fundamental_data,
            "news_data":            news_data,
            "sentiment_data":       sentiment_data,
            "sentiment_summary":    sentiment_summary,
        }


if __name__ == "__main__":
    orchestrator = DataOrchestrator()
    data = orchestrator.collect_all_data()