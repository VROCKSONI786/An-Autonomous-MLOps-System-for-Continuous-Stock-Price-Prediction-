"""
Unified configuration management with user preferences support
Merges default config.yaml with user-selected stocks
"""
import os
import yaml
import json
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConfigManager:
    """Manage application config with user preferences"""
    
    def __init__(self, config_path=None):
        """Initialize config manager
        
        Args:
            config_path: Path to config.yaml (auto-detected if None)
        """
        # Find project root
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.dirname(current_dir)
        
        # Load default config
        if config_path is None:
            config_path = os.path.join(self.project_root, "config", "config.yaml")
        
        with open(config_path, 'r') as f:
            self.default_config = yaml.safe_load(f)
        
        # User preferences file
        self.prefs_dir = os.path.join(self.project_root, "config")
        os.makedirs(self.prefs_dir, exist_ok=True)
        self.prefs_file = os.path.join(self.prefs_dir, "user_stocks.json")
        
        # Load or create user preferences
        self.user_prefs = self._load_user_preferences()
        
        logger.info(f"ConfigManager initialized with project root: {self.project_root}")
    
    def _load_user_preferences(self):
        """Load user stock preferences from JSON file"""
        if not os.path.exists(self.prefs_file):
            return {
                'custom_stocks': [],
                'created_at': datetime.now().isoformat()
            }
        
        try:
            with open(self.prefs_file, 'r') as f:
                prefs = json.load(f)
            logger.info(f"Loaded user preferences: {len(prefs.get('custom_stocks', []))} custom stocks")
            return prefs
        except Exception as e:
            logger.error(f"Error loading user preferences: {e}")
            return {'custom_stocks': []}
    
    def _save_user_preferences(self):
        """Save user stock preferences to JSON file"""
        try:
            with open(self.prefs_file, 'w') as f:
                json.dump(self.user_prefs, f, indent=2)
            logger.info(f"Saved user preferences: {len(self.user_prefs['custom_stocks'])} custom stocks")
        except Exception as e:
            logger.error(f"Error saving user preferences: {e}")
    
    def get_default_stocks(self):
        """Get default stocks from config"""
        return self.default_config['data_collection']['stock_symbols']
    
    def get_custom_stocks(self):
        """Get user-added custom stocks"""
        return self.user_prefs.get('custom_stocks', [])
    
    def get_all_stocks(self, include_custom=True):
        """Get all stocks (default + custom)
        
        Args:
            include_custom: Include user-added stocks (default: True)
        
        Returns:
            List of stock symbols
        """
        stocks = self.get_default_stocks()
        
        if include_custom:
            custom = self.get_custom_stocks()
            # Avoid duplicates
            stocks = list(set(stocks + custom))
        
        return sorted(stocks)
    
    def add_stock(self, symbol):
        """Add a custom stock to track
        
        Args:
            symbol: Stock symbol (e.g., "WIPRO.NS", "SBIN.NS")
        
        Returns:
            bool: True if added, False if already exists
        """
        # Normalize format
        if not symbol.endswith(('.NS', '.BO', '.BSE', '.NSE')):
            symbol = symbol + '.NS'  # Default to NSE
        
        symbol = symbol.upper()
        
        custom_stocks = self.get_custom_stocks()
        
        # Check if already exists (in both default and custom)
        all_stocks = self.get_all_stocks(include_custom=True)
        if symbol in all_stocks:
            logger.warning(f"Stock {symbol} already tracked")
            return False
        
        custom_stocks.append(symbol)
        self.user_prefs['custom_stocks'] = custom_stocks
        self.user_prefs['last_updated'] = datetime.now().isoformat()
        self._save_user_preferences()
        
        logger.info(f"Added custom stock: {symbol}")
        return True
    
    def remove_stock(self, symbol):
        """Remove a custom stock from tracking
        
        Args:
            symbol: Stock symbol to remove
        
        Returns:
            bool: True if removed, False if not found or is default
        """
        # Normalize format
        if not symbol.endswith(('.NS', '.BO', '.BSE', '.NSE')):
            symbol = symbol + '.NS'
        
        symbol = symbol.upper()
        
        # Can't remove default stocks
        default_stocks = self.get_default_stocks()
        if symbol in default_stocks:
            logger.warning(f"Cannot remove default stock: {symbol}")
            return False
        
        custom_stocks = self.get_custom_stocks()
        
        if symbol not in custom_stocks:
            logger.warning(f"Stock {symbol} not in custom list")
            return False
        
        custom_stocks.remove(symbol)
        self.user_prefs['custom_stocks'] = custom_stocks
        self.user_prefs['last_updated'] = datetime.now().isoformat()
        self._save_user_preferences()
        
        logger.info(f"Removed custom stock: {symbol}")
        return True
    
    def reset_custom_stocks(self):
        """Reset to only default stocks"""
        self.user_prefs['custom_stocks'] = []
        self.user_prefs['last_updated'] = datetime.now().isoformat()
        self._save_user_preferences()
        logger.info("Reset to default stocks only")
    
    def get_config(self, include_custom_stocks=True):
        """Get full config with merged stock symbols
        
        Args:
            include_custom_stocks: Include custom stocks in returned config
        
        Returns:
            dict: Configuration with merged stock_symbols
        """
        config = self.default_config.copy()
        config['data_collection']['stock_symbols'] = self.get_all_stocks(include_custom_stocks)
        return config
    
    def validate_stock(self, symbol):
        """Validate stock symbol format
        
        Args:
            symbol: Stock symbol to validate
        
        Returns:
            tuple: (bool, str) - (is_valid, normalized_symbol)
        """
        if not symbol or not isinstance(symbol, str):
            return False, ""
        
        symbol = symbol.strip().upper()
        
        # Check if it looks like a valid Indian stock symbol
        if not any(symbol.endswith(suffix) for suffix in ['.NS', '.BO', '.BSE', '.NSE']):
            symbol = symbol + '.NS'
        
        # Basic format check
        parts = symbol.split('.')
        if len(parts) != 2 or len(parts[0]) < 1 or len(parts[0]) > 20:
            return False, ""
        
        return True, symbol
    
    def get_stats(self):
        """Get configuration statistics"""
        default_stocks = self.get_default_stocks()
        custom_stocks = self.get_custom_stocks()
        
        return {
            'default_stocks_count': len(default_stocks),
            'custom_stocks_count': len(custom_stocks),
            'total_tracked': len(self.get_all_stocks()),
            'default_stocks': default_stocks,
            'custom_stocks': custom_stocks
        }


if __name__ == "__main__":
    # Test the config manager
    cm = ConfigManager()
    
    print("=== Default Stocks ===")
    print(cm.get_default_stocks())
    
    print("\n=== Custom Stocks ===")
    print(cm.get_custom_stocks())
    
    print("\n=== All Stocks ===")
    print(cm.get_all_stocks())
    
    print("\n=== Stats ===")
    import pprint
    pprint.pprint(cm.get_stats())
