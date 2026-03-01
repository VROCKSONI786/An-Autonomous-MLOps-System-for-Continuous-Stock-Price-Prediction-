import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YFinanceCollector:
    """Collect stock data from Yahoo Finance"""
    
    def __init__(self, symbols, period="2y", interval="1d", data_dir=None):
        self.symbols = symbols
        self.period = period
        self.interval = interval
        self.data_dir = data_dir
        
    def fetch_stock_data(self, symbol):
        """Fetch historical data for a single stock"""
        try:
            logger.info(f"Fetching data for {symbol}")
            ticker = yf.Ticker(symbol)
            
            # Get historical data
            hist_data = ticker.history(period=self.period, interval=self.interval)
            
            # Get additional info (safe handling for None)
            info = ticker.info or {}
            
            # Add fundamental data
            hist_data['Symbol'] = symbol
            hist_data['Market_Cap'] = info.get('marketCap', None)
            hist_data['PE_Ratio'] = info.get('trailingPE', None)
            hist_data['Dividend_Yield'] = info.get('dividendYield', None)
            hist_data['52Week_High'] = info.get('fiftyTwoWeekHigh', None)
            hist_data['52Week_Low'] = info.get('fiftyTwoWeekLow', None)
            
            logger.info(f"Successfully fetched {len(hist_data)} records for {symbol}")
            return hist_data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    def fetch_all_stocks(self):
        """Fetch data for all configured stocks"""
        all_data = []
        
        for symbol in self.symbols:
            data = self.fetch_stock_data(symbol)
            if data is not None:
                all_data.append(data)
        
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=False)
            return combined_data
        return None
    
    def save_data(self, data, filename="raw_stock_data.csv"):
        """Save collected data to CSV"""
        try:
            if self.data_dir:
                filepath = os.path.join(self.data_dir, filename)
            else:
                filepath = os.path.join("data", "raw", filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            data.to_csv(filepath)
            logger.info(f"Data saved to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            return None

if __name__ == "__main__":
    # Test the collector
    symbols = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
    collector = YFinanceCollector(symbols, period="1y")
    
    data = collector.fetch_all_stocks()
    if data is not None:
        print(f"Collected data shape: {data.shape}")
        print(data.head())
        collector.save_data(data)