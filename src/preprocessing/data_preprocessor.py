import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import logging
import os
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Preprocess stock data for LSTM model"""
    
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'MA_7', 'MA_21', 'MA_50', 'EMA_12', 'EMA_26',
            'MACD', 'Signal_Line', 'RSI', 'BB_Middle', 'BB_Upper', 'BB_Lower',
            'ROC', 'ATR', 'Volume_MA', 'Volume_Ratio', 'Momentum'
        ]
        
    def load_latest_data(self, data_dir=None):
        """Load the most recent data files"""
        try:
            # Get project root
            if data_dir is None:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(os.path.dirname(current_dir))
                data_dir = os.path.join(project_root, "data", "raw")
            
            logger.info(f"Looking for data in: {data_dir}")
            
            # Find latest stock prices file
            files = [f for f in os.listdir(data_dir) if f.startswith('stock_prices_')]
            if not files:
                raise FileNotFoundError(f"No stock price data found in {data_dir}")
            
            latest_file = sorted(files)[-1]
            filepath = os.path.join(data_dir, latest_file)
            
            logger.info(f"Loading data from {filepath}")
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return None
    
    def create_technical_indicators(self, df):
        """Create technical indicators as features"""
        logger.info("Creating technical indicators...")
        
        # Make a copy
        df = df.copy()
        
        # Moving Averages
        df['MA_7'] = df.groupby('Symbol')['Close'].transform(lambda x: x.rolling(window=7).mean())
        df['MA_21'] = df.groupby('Symbol')['Close'].transform(lambda x: x.rolling(window=21).mean())
        df['MA_50'] = df.groupby('Symbol')['Close'].transform(lambda x: x.rolling(window=50).mean())
        
        # Exponential Moving Averages
        df['EMA_12'] = df.groupby('Symbol')['Close'].transform(lambda x: x.ewm(span=12).mean())
        df['EMA_26'] = df.groupby('Symbol')['Close'].transform(lambda x: x.ewm(span=26).mean())
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['Signal_Line'] = df.groupby('Symbol')['MACD'].transform(lambda x: x.ewm(span=9).mean())
        
        # RSI (Relative Strength Index)
        def calculate_rsi(series, period=14):
            delta = series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        df['RSI'] = df.groupby('Symbol')['Close'].transform(lambda x: calculate_rsi(x))
        
        # Bollinger Bands
        df['BB_Middle'] = df.groupby('Symbol')['Close'].transform(lambda x: x.rolling(window=20).mean())
        df['BB_Std'] = df.groupby('Symbol')['Close'].transform(lambda x: x.rolling(window=20).std())
        df['BB_Upper'] = df['BB_Middle'] + (df['BB_Std'] * 2)
        df['BB_Lower'] = df['BB_Middle'] - (df['BB_Std'] * 2)
        
        # Price Rate of Change
        df['ROC'] = df.groupby('Symbol')['Close'].transform(lambda x: x.pct_change(periods=12) * 100)
        
        # Average True Range (ATR)
        df['High_Low'] = df['High'] - df['Low']
        df['High_Close'] = abs(df['High'] - df['Close'].shift(1))
        df['Low_Close'] = abs(df['Low'] - df['Close'].shift(1))
        df['TR'] = df[['High_Low', 'High_Close', 'Low_Close']].max(axis=1)
        df['ATR'] = df.groupby('Symbol')['TR'].transform(lambda x: x.rolling(window=14).mean())
        
        # Volume indicators
        df['Volume_MA'] = df.groupby('Symbol')['Volume'].transform(lambda x: x.rolling(window=20).mean())
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        # Price momentum
        df['Momentum'] = df.groupby('Symbol')['Close'].transform(lambda x: x - x.shift(10))
        
        # Drop temporary columns
        df.drop(['High_Low', 'High_Close', 'Low_Close', 'TR', 'BB_Std'], axis=1, inplace=True)
        
        logger.info(f"Created technical indicators. Shape: {df.shape}")
        return df
    
    def prepare_features(self, df):
        """Prepare feature set for model"""
        logger.info("Preparing features...")
        
        # Drop rows with NaN (from rolling calculations)
        df_clean = df.dropna()
        
        logger.info(f"Features prepared. Clean data shape: {df_clean.shape}")
        return df_clean
    
    def scale_data(self, df, fit=True):
        """Scale features using MinMaxScaler"""
        logger.info("Scaling data...")
        
        if fit:
            scaled_data = self.scaler.fit_transform(df[self.feature_columns])
        else:
            scaled_data = self.scaler.transform(df[self.feature_columns])
        
        scaled_df = pd.DataFrame(
            scaled_data,
            columns=self.feature_columns,
            index=df.index
        )
        
        # Keep Symbol column
        scaled_df['Symbol'] = df['Symbol'].values
        
        return scaled_df
    
    def create_sequences(self, data, symbol, sequence_length=60, target_col='Close'):
        """Create sequences for LSTM"""
        logger.info(f"Creating sequences for {symbol} with length {sequence_length}")
        
        # Filter data for specific symbol
        symbol_data = data[data['Symbol'] == symbol].copy()
        symbol_data = symbol_data.sort_index()
        
        # Get feature values
        feature_values = symbol_data[self.feature_columns].values
        
        X, y = [], []
        
        # Find close price index
        close_idx = self.feature_columns.index(target_col)
        
        for i in range(sequence_length, len(feature_values)):
            X.append(feature_values[i-sequence_length:i])
            # Target is the Close price
            y.append(feature_values[i, close_idx])
        
        logger.info(f"Created {len(X)} sequences for {symbol}")
        
        return np.array(X), np.array(y)
    
    def save_scaler(self, filepath=None):
        """Save the fitted scaler"""
        try:
            if filepath is None:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(os.path.dirname(current_dir))
                filepath = os.path.join(project_root, "models", "scaler.pkl")
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            joblib.dump(self.scaler, filepath)
            joblib.dump(self.feature_columns, filepath.replace('.pkl', '_features.pkl'))
            logger.info(f"Scaler saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving scaler: {str(e)}")
    
    def load_scaler(self, filepath=None):
        """Load a saved scaler"""
        try:
            if filepath is None:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(os.path.dirname(current_dir))
                filepath = os.path.join(project_root, "models", "scaler.pkl")
            
            self.scaler = joblib.load(filepath)
            # Load feature columns if available
            feature_file = filepath.replace('.pkl', '_features.pkl')
            if os.path.exists(feature_file):
                self.feature_columns = joblib.load(feature_file)
            logger.info(f"Scaler loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading scaler: {str(e)}")
    
    def preprocess_pipeline(self, data_dir=None, sequence_length=60):
        """Complete preprocessing pipeline"""
        logger.info("=== Starting Preprocessing Pipeline ===")
        
        # Load data
        df = self.load_latest_data(data_dir)
        if df is None:
            return None
        
        # Create technical indicators
        df = self.create_technical_indicators(df)
        
        # Prepare features
        df = self.prepare_features(df)
        
        # Scale data
        scaled_df = self.scale_data(df, fit=True)
        
        # Save scaler
        self.save_scaler()
        
        # Save preprocessed data
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        output_dir = os.path.join(project_root, "data", "processed")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "preprocessed_data.csv")
        scaled_df.to_csv(output_file)
        logger.info(f"Preprocessed data saved to {output_file}")
        
        logger.info("=== Preprocessing Complete ===")
        return scaled_df

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    preprocessed_data = preprocessor.preprocess_pipeline()
    
    if preprocessed_data is not None:
        print(f"\nPreprocessed data shape: {preprocessed_data.shape}")
        print(f"\nFeature columns: {preprocessor.feature_columns}")
        print(f"\nSample data:\n{preprocessed_data.head()}")