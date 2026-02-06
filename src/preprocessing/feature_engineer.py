import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Advanced feature engineering for stock prediction"""
    
    def __init__(self):
        pass
    
    def add_sentiment_features(self, price_df, sentiment_df):
        """Merge sentiment data with price data"""
        logger.info("Adding sentiment features...")
        
        try:
            # Convert timestamp to date
            sentiment_df['Date'] = pd.to_datetime(sentiment_df['timestamp']).dt.date
            price_df.index = pd.to_datetime(price_df.index)
            
            # Aggregate sentiment by date and symbol
            daily_sentiment = sentiment_df.groupby(['Date', 'symbol']).agg({
                'confidence': 'mean',
                'sentiment': lambda x: x.mode()[0] if len(x) > 0 else 'neutral'
            }).reset_index()
            
            # Encode sentiment
            sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}
            daily_sentiment['sentiment_score'] = daily_sentiment['sentiment'].map(sentiment_map)
            
            # Merge with price data
            price_df['Date'] = price_df.index.date
            merged_df = price_df.merge(
                daily_sentiment,
                left_on=['Date', 'Symbol'],
                right_on=['Date', 'symbol'],
                how='left'
            )
            
            # Fill missing sentiment with neutral
            merged_df['sentiment_score'].fillna(0, inplace=True)
            merged_df['confidence'].fillna(0.5, inplace=True)
            
            merged_df.drop(['Date', 'symbol', 'sentiment'], axis=1, inplace=True, errors='ignore')
            
            logger.info("Sentiment features added successfully")
            return merged_df
            
        except Exception as e:
            logger.error(f"Error adding sentiment features: {str(e)}")
            return price_df
    
    def add_fundamental_features(self, price_df, fundamental_df):
        """Add fundamental data features"""
        logger.info("Adding fundamental features...")
        
        try:
            # Simple merge on Symbol
            merged_df = price_df.merge(
                fundamental_df[['Symbol', 'Market_Cap', 'PE_Ratio', 'Dividend_Yield']],
                on='Symbol',
                how='left'
            )
            
            # Forward fill fundamental data (as they don't change daily)
            merged_df[['Market_Cap', 'PE_Ratio', 'Dividend_Yield']] = \
                merged_df.groupby('Symbol')[['Market_Cap', 'PE_Ratio', 'Dividend_Yield']].ffill()
            
            logger.info("Fundamental features added successfully")
            return merged_df
            
        except Exception as e:
            logger.error(f"Error adding fundamental features: {str(e)}")
            return price_df
    
    def add_time_features(self, df):
        """Add time-based features"""
        logger.info("Adding time features...")
        
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        
        df['Day_of_Week'] = df.index.dayofweek
        df['Month'] = df.index.month
        df['Quarter'] = df.index.quarter
        df['Is_Month_Start'] = df.index.is_month_start.astype(int)
        df['Is_Month_End'] = df.index.is_month_end.astype(int)
        
        logger.info("Time features added successfully")
        return df
    
    def add_lag_features(self, df, lags=[1, 2, 3, 5, 10]):
        """Add lagged price features"""
        logger.info(f"Adding lag features for periods: {lags}")
        
        df = df.copy()
        
        for lag in lags:
            df[f'Close_Lag_{lag}'] = df.groupby('Symbol')['Close'].shift(lag)
            df[f'Volume_Lag_{lag}'] = df.groupby('Symbol')['Volume'].shift(lag)
        
        logger.info("Lag features added successfully")
        return df

if __name__ == "__main__":
    engineer = FeatureEngineer()
    print("Feature Engineer module ready")