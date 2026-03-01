"""
economic_data_collector.py
──────────────────────────
Collects two types of data that improve stock price prediction beyond pure TA:

1. MACRO INDICATORS (via FRED API — free, register at fred.stlouisfed.org)
   ┌─────────────────────────────────────────────────────────────────────┐
   │ Series        │ What it tells us                                    │
   ├─────────────────────────────────────────────────────────────────────┤
   │ DEXINUS       │ USD/INR exchange rate → import costs, FII flows     │
   │ VIXCLS        │ CBOE VIX → global fear/risk-off sentiment           │
   │ DGS10         │ US 10-yr yield → capital flows from EMs to US bonds │
   │ FEDFUNDS      │ Fed Funds rate → global liquidity                   │
   │ CPIAUCSL      │ US CPI → Fed policy outlook                         │
   │ DCOILWTICO    │ Crude oil (WTI) → India's import bill, inflation    │
   │ GOLDAMGBD228NLBM│ Gold price → safe-haven demand, INR correlation   │
   └─────────────────────────────────────────────────────────────────────┘

2. ECONOMIC NEWS (via free RSS feeds — no API key needed)
   • Reuters Business         • Economic Times Markets
   • Moneycontrol              • RBI Press Releases
   • Business Standard         • Investing.com RSS

Usage:
    collector = EconomicDataCollector()
    indicators = collector.fetch_macro_indicators()        # DataFrame
    news       = collector.fetch_economic_news()           # DataFrame
    collector.save_all()                                   # saves both to data/raw/
"""

import os
import sys
import logging
import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import numpy as np
import requests
import feedparser
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _project_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(here))


# ── FRED series we want ───────────────────────────────────────────────────────
FRED_SERIES = {
    "DEXINUS":           "usd_inr_rate",          # USD/INR exchange rate
    "VIXCLS":            "vix",                   # CBOE VIX
    "DGS10":             "us_10yr_yield",          # US 10-year treasury yield
    "FEDFUNDS":          "fed_funds_rate",         # Fed Funds Rate
    "CPIAUCSL":          "us_cpi",                 # US Consumer Price Index
    "DCOILWTICO":        "crude_oil_wti",          # WTI crude oil price
    "GOLDAMGBD228NLBM":  "gold_price_usd",         # Gold price (London fix)
    "T10YIE":            "us_10yr_inflation_exp",  # 10-yr inflation expectations
}

# ── RSS feeds for economic news ────────────────────────────────────────────────
# All free, no API key required
ECONOMIC_NEWS_FEEDS = {
    "reuters_business": "https://feeds.reuters.com/reuters/businessNews",
    "reuters_markets":  "https://feeds.reuters.com/reuters/companyNews",
    "economic_times":   "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "moneycontrol":     "https://www.moneycontrol.com/rss/MCtopnews.xml",
    "business_standard":"https://www.business-standard.com/rss/markets-106.rss",
    "livemint_markets": "https://www.livemint.com/rss/markets",
    "rbi_press":        "https://www.rbi.org.in/pressreleases/rss.aspx",
    "investing_india":  "https://in.investing.com/rss/news_283.rss",
}

# Economic keywords to filter relevant articles
ECONOMIC_KEYWORDS = [
    "rbi", "repo rate", "inflation", "cpi", "gdp", "fed", "federal reserve",
    "interest rate", "monetary policy", "forex", "rupee", "dollar", "crude",
    "oil price", "gold", "treasury", "yield", "fii", "dii", "sensex", "nifty",
    "market rally", "market fall", "recession", "rate hike", "rate cut",
    "trade deficit", "current account", "fiscal deficit", "budget", "imf",
    "world bank", "global economy", "china economy", "us economy",
]


class EconomicDataCollector:

    def __init__(self):
        self.fred_api_key = os.getenv("FRED_API_KEY", "")
        self.fred_base    = "https://api.stlouisfed.org/fred/series/observations"
        self.root         = _project_root()
        self.data_dir     = os.path.join(self.root, "data", "raw")
        os.makedirs(self.data_dir, exist_ok=True)

        if not self.fred_api_key:
            logger.warning(
                "FRED_API_KEY not set. Macro indicators will use yfinance fallback.\n"
                "Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html"
            )

    # ── FRED indicators ────────────────────────────────────────────────────────

    def _fetch_fred_series(
        self,
        series_id: str,
        observation_start: str,
        observation_end: str,
    ) -> Optional[pd.Series]:
        """Fetch a single FRED series. Returns daily pd.Series or None."""
        if not self.fred_api_key:
            return None

        url = self.fred_base
        params = {
            "series_id":          series_id,
            "api_key":            self.fred_api_key,
            "file_type":          "json",
            "observation_start":  observation_start,
            "observation_end":    observation_end,
            "frequency":          "d",          # daily
            "aggregation_method": "last",
        }

        try:
            resp = requests.get(url, params=params, timeout=15)
            # some FRED series (eg. CPI, Fed Funds) are only available monthly;
            # requesting daily frequency will raise a 400 bad request.  we
            # detect that and retry with monthly before giving up.
            if resp.status_code == 400 and params.get("frequency") == "d":
                logger.warning(
                    f"400 from FRED for {series_id} with daily frequency, retrying monthly"
                )
                params["frequency"] = "m"
                resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()

            data  = resp.json()
            obs   = data.get("observations", [])
            if not obs:
                return None

            series = pd.Series(
                {
                    o["date"]: float(o["value"]) if o["value"] != "." else np.nan
                    for o in obs
                },
                name=series_id,
            )
            series.index = pd.to_datetime(series.index)
            return series

        except Exception as e:
            logger.error(f"FRED fetch failed for {series_id}: {e}")
            return None

    def _fetch_yfinance_fallback(self, period: str = "2y") -> pd.DataFrame:
        """
        Fallback when no FRED key: use yfinance for proxy indicators.
        Fetches: USD/INR, VIX, Gold, Crude oil, US 10-yr yield proxy.
        """
        import yfinance as yf

        tickers = {
            "USDINR=X":   "usd_inr_rate",
            "^VIX":       "vix",
            "GC=F":       "gold_price_usd",
            "CL=F":       "crude_oil_wti",
            "^TNX":       "us_10yr_yield",
        }

        frames = []
        for ticker, col_name in tickers.items():
            try:
                df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
                if not df.empty:
                    s = df["Close"].copy()
                    s.name = col_name
                    frames.append(s)
                    logger.info(f"  yfinance fallback: {ticker} → {col_name} ({len(s)} rows)")
            except Exception as e:
                logger.warning(f"  yfinance failed for {ticker}: {e}")

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, axis=1)
        combined.index = pd.to_datetime(combined.index)
        combined.index = combined.index.tz_localize(None)
        return combined

    def fetch_macro_indicators(self, years: int = 2) -> pd.DataFrame:
        """
        Fetch all macro indicators. Returns a daily DataFrame indexed by date.
        Columns: usd_inr_rate, vix, us_10yr_yield, fed_funds_rate,
                 us_cpi, crude_oil_wti, gold_price_usd, us_10yr_inflation_exp
        """
        end_date   = datetime.today()
        start_date = end_date - timedelta(days=365 * years)
        end_str    = end_date.strftime("%Y-%m-%d")
        start_str  = start_date.strftime("%Y-%m-%d")

        logger.info(f"Fetching macro indicators ({start_str} → {end_str})")

        if self.fred_api_key:
            all_series = []
            fetched_ids = []
            for series_id, col_name in FRED_SERIES.items():
                logger.info(f"  FRED: {series_id} → {col_name}")
                s = self._fetch_fred_series(series_id, start_str, end_str)
                if s is not None:
                    s.name = col_name
                    all_series.append(s)
                    fetched_ids.append(series_id)
                time.sleep(0.3)   # FRED rate limit: be polite

            if all_series:
                df = pd.concat(all_series, axis=1)
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
                # Forward-fill missing days (weekends, holidays)
                df = df.ffill().bfill()
                logger.info(f"✓ FRED: {df.shape[0]} rows, {df.shape[1]} indicators")

                # if some series failed, attempt yfinance proxies for them
                missing = [col for (sid, col) in FRED_SERIES.items() if sid not in fetched_ids]
                if missing:
                    logger.warning(
                        f"FRED could not retrieve {len(missing)} series ({missing}); "
                        "backfilling from yfinance where possible."
                    )
                    fallback = self._fetch_yfinance_fallback(period=f"{years}y")
                    if not fallback.empty:
                        # keep only missing columns from fallback
                        use = [c for c in fallback.columns if c in missing]
                        if use:
                            df = df.join(fallback[use], how="left")
                            df = df.ffill().bfill()
                            logger.info(f"✓ Added {len(use)} proxies from yfinance: {use}")
                return df

        logger.info("Using yfinance fallback for macro indicators…")
        df = self._fetch_yfinance_fallback(period=f"{years}y")

        if df.empty:
            logger.warning("Both FRED and yfinance failed. Returning empty DataFrame.")
            return pd.DataFrame()

        df = df.sort_index().ffill().bfill()
        logger.info(f"✓ yfinance macro: {df.shape[0]} rows, {df.shape[1]} indicators")
        return df

    # ── Derived macro features ─────────────────────────────────────────────────

    def compute_macro_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer additional macro features from raw indicators:
        - Rate-of-change (momentum) for each indicator
        - Rolling z-score (how extreme is current value vs 60-day history)
        - Yield curve proxy (if both available)
        """
        if df.empty:
            return df

        out = df.copy()

        for col in df.columns:
            # 5-day momentum
            out[f"{col}_mom5"]   = df[col].pct_change(5)
            # 20-day momentum
            out[f"{col}_mom20"]  = df[col].pct_change(20)
            # 60-day rolling z-score
            roll = df[col].rolling(60)
            out[f"{col}_zscore"] = (df[col] - roll.mean()) / (roll.std() + 1e-10)

        # INR weakness signal (high = rupee weakening = bad for import-heavy stocks)
        if "usd_inr_rate" in df.columns:
            out["inr_weakness"] = (
                df["usd_inr_rate"].rolling(20).mean()
                / df["usd_inr_rate"].rolling(60).mean()
                - 1
            )

        # Equity risk premium proxy
        if "vix" in df.columns:
            out["vix_regime"] = (df["vix"] > 25).astype(int)   # 1 = high fear

        out = out.ffill().bfill()
        logger.info(f"Macro features: {out.shape[1]} total columns")
        return out

    # ── Economic news RSS ──────────────────────────────────────────────────────

    def fetch_economic_news(
        self,
        max_per_feed: int = 20,
        days_back: int = 7,
    ) -> pd.DataFrame:
        """
        Fetch economic/market news from RSS feeds.
        Returns DataFrame with: title, summary, source, published, url, is_economic
        """
        cutoff   = datetime.now() - timedelta(days=days_back)
        articles = []

        for source, url in ECONOMIC_NEWS_FEEDS.items():
            logger.info(f"  Fetching RSS: {source}")
            try:
                feed = feedparser.parse(url)
                count = 0

                for entry in feed.entries[:max_per_feed]:
                    # Parse publish date
                    pub_date = None
                    if hasattr(entry, "published_parsed") and entry.published_parsed:
                        try:
                            from calendar import timegm
                            pub_date = datetime.fromtimestamp(timegm(entry.published_parsed))
                        except Exception:
                            pub_date = datetime.now()
                    else:
                        pub_date = datetime.now()

                    # Filter by recency
                    if pub_date < cutoff:
                        continue

                    title   = getattr(entry, "title",   "")
                    summary = getattr(entry, "summary", "")
                    link    = getattr(entry, "link",    "")

                    # Check economic relevance
                    text_lower = (title + " " + summary).lower()
                    is_economic = any(kw in text_lower for kw in ECONOMIC_KEYWORDS)

                    articles.append({
                        "title":        title,
                        "summary":      summary,
                        "source":       source,
                        "published":    pub_date.strftime("%Y-%m-%d %H:%M:%S"),
                        "url":          link,
                        "is_economic":  is_economic,
                        "symbol":       "GLOBAL",      # economic news = all symbols
                    })
                    count += 1

                logger.info(f"    {source}: {count} articles (last {days_back} days)")
                time.sleep(0.5)   # polite delay

            except Exception as e:
                logger.error(f"    {source} RSS failed: {e}")

        if not articles:
            logger.warning("No economic news articles fetched.")
            return pd.DataFrame()

        df = pd.DataFrame(articles).drop_duplicates(subset=["title"])
        logger.info(f"✓ Economic news: {len(df)} unique articles "
                    f"({df['is_economic'].sum()} economic-relevant)")
        return df

    # ── Save all ───────────────────────────────────────────────────────────────

    def save_all(self, years: int = 2) -> dict:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Macro indicators
        macro_raw = self.fetch_macro_indicators(years=years)
        macro     = self.compute_macro_features(macro_raw)

        macro_path = None
        if not macro.empty:
            macro_path = os.path.join(self.data_dir, f"macro_indicators_{ts}.csv")
            macro.to_csv(macro_path)
            logger.info(f"Macro indicators saved: {macro_path}")

        # Economic news
        news      = self.fetch_economic_news()
        news_path = None
        if not news.empty:
            news_path = os.path.join(self.data_dir, f"economic_news_{ts}.csv")
            news.to_csv(news_path, index=False)
            logger.info(f"Economic news saved: {news_path}")

        return {
            "macro_indicators": macro,
            "macro_path":       macro_path,
            "economic_news":    news,
            "news_path":        news_path,
        }

    # ── Merge with stock data ──────────────────────────────────────────────────

    @staticmethod
    def merge_with_stock_data(
        stock_df: pd.DataFrame,
        macro_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Left-join macro indicators onto stock price DataFrame.
        Stock dates without macro data are forward-filled.
        """
        if macro_df.empty:
            return stock_df

        stock_df.index = pd.to_datetime(stock_df.index)
        macro_df.index = pd.to_datetime(macro_df.index)

        merged = stock_df.join(macro_df, how="left")
        merged = merged.ffill().bfill()

        added  = [c for c in macro_df.columns if c in merged.columns]
        logger.info(f"Merged {len(added)} macro features into stock data")
        return merged


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    collector = EconomicDataCollector()
    results   = collector.save_all(years=2)

    macro = results["macro_indicators"]
    news  = results["economic_news"]

    print("\n=== Macro Indicators ===")
    if not macro.empty:
        print(macro.tail(5).to_string())
    else:
        print("No macro data.")

    print("\n=== Economic News Sample ===")
    if not news.empty:
        print(news[["title", "source", "published", "is_economic"]].head(10).to_string())
    else:
        print("No economic news.")