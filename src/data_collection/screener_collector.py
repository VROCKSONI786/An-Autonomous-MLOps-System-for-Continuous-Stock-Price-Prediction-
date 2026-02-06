import requests
from bs4 import BeautifulSoup
import pandas as pd
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScreenerCollector:
    """Collect fundamental data from Screener.in"""
    
    def __init__(self):
        self.base_url = "https://www.screener.in/company"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def get_company_code(self, symbol):
        """Convert NSE symbol to Screener company code"""
        # Remove .NS suffix
        return symbol.replace('.NS', '').replace('.BO', '')
    
    def fetch_financials(self, symbol):
        """Fetch financial data for a company"""
        try:
            company_code = self.get_company_code(symbol)
            url = f"{self.base_url}/{company_code}/"
            
            logger.info(f"Fetching financials for {company_code}")
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code != 200:
                logger.warning(f"Failed to fetch {company_code}: Status {response.status_code}")
                return None
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract key metrics
            financials = {
                'Symbol': symbol,
                'Company': company_code,
                'Timestamp': pd.Timestamp.now()
            }
            
            # Find key ratios
            ratios_section = soup.find_all('li', class_='flex flex-space-between')
            
            for ratio in ratios_section:
                try:
                    name = ratio.find('span', class_='name')
                    value = ratio.find('span', class_='number')
                    
                    if name and value:
                        key = name.text.strip().replace(' ', '_')
                        val = value.text.strip()
                        financials[key] = val
                except:
                    continue
            
            logger.info(f"Successfully fetched financials for {company_code}")
            time.sleep(2)  # Rate limiting
            
            return financials
            
        except Exception as e:
            logger.error(f"Error fetching financials for {symbol}: {str(e)}")
            return None
    
    def fetch_all_financials(self, symbols):
        """Fetch financials for multiple stocks"""
        all_financials = []
        
        for symbol in symbols:
            financials = self.fetch_financials(symbol)
            if financials:
                all_financials.append(financials)
        
        if all_financials:
            return pd.DataFrame(all_financials)
        return None

if __name__ == "__main__":
    # Test the collector
    symbols = ["RELIANCE.NS", "TCS.NS"]
    collector = ScreenerCollector()
    
    financials = collector.fetch_all_financials(symbols)
    if financials is not None:
        print(financials)