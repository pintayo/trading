import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np
import sqlite3
from .technical_indicators import TechnicalIndicators
from config.model_config import SIMPLE_CONFIG, ADVANCED_CONFIG
from config.trading_config import TIMEFRAME

class DataProcessor:
    """Simplified data processor for profitable 52%+ model"""
    
    def __init__(self, symbol=None, model_type="simple", interval=TIMEFRAME):
        self.model_type = model_type
        self.interval = interval
        if model_type == "simple":
            if symbol:
                self.db_path = f"data/{symbol.replace('=X', '').lower()}_data.db"
            else:
                self.db_path = "data/usdjpy_data.db"
        elif model_type == "advanced":
            self.db_path = "data/massive_forex_data.db"
            self.symbol = symbol # Store symbol for filtering

        self.indicators = TechnicalIndicators()
    
    def load_data(self):
        """Load and clean price data - simplified approach"""
        conn = sqlite3.connect(self.db_path)
        if self.model_type == "simple":
            df = pd.read_sql_query("SELECT * FROM price_data", conn)
        elif self.model_type == "advanced":
            # Filter by symbol and interval for advanced model
            df = pd.read_sql_query(f"SELECT * FROM forex_data WHERE symbol='{self.symbol}' AND interval='{self.interval}'", conn)
            # Drop symbol and interval columns as they are not features
            df = df.drop(columns=['symbol', 'interval'])
        conn.close()
        
        # Handle datetime column (most common cases)
        if 'Datetime' in df.columns:
            df.set_index(pd.to_datetime(df['Datetime']), inplace=True)
        elif 'timestamp' in df.columns:
            df.set_index(pd.to_datetime(df['timestamp']), inplace=True)
        else:
            # Use first column as datetime
            df.set_index(pd.to_datetime(df.iloc[:, 0]), inplace=True)
        
        # Clean column names (handle yfinance multi-index)
        df.columns = df.columns.str.lower() if not df.columns.empty else df.columns
        
        print(f"✅ Loaded {len(df)} price records")
        return df
    
    def prepare_training_data(self):
        """Prepare simple, proven feature set"""
        df = self.load_data()
        
        # Add core technical indicators
        df = self.indicators.add_all_indicators(df)
        
        # --- NEW TARGET VARIABLE DEFINITION ---
        # Use config based on model type
        if self.model_type == "simple":
            target_threshold = SIMPLE_CONFIG['target_percent_threshold']
            target_future_candles = SIMPLE_CONFIG['target_future_candles']
            feature_cols = SIMPLE_CONFIG['feature_columns']
        elif self.model_type == "advanced":
            target_threshold = ADVANCED_CONFIG['target_percent_threshold']
            target_future_candles = ADVANCED_CONFIG['target_future_candles']
            feature_cols = ADVANCED_CONFIG['feature_columns']

        # Calculate future close price
        df['future_close'] = df['close'].shift(-target_future_candles)
        
        # Define thresholds
        upper_bound = df['close'] * (1 + target_threshold)
        lower_bound = df['close'] * (1 - target_threshold)
        
        # Assign target: 1 for up, 0 for down
        df['target'] = np.nan # Initialize with NaN
        df.loc[df['future_close'] >= upper_bound, 'target'] = 1
        df.loc[df['future_close'] <= lower_bound, 'target'] = 0
        
        # Drop rows where target is NaN (i.e., price stayed within the threshold)
        df = df.dropna(subset=['target'])
        df['target'] = df['target'].astype(int)
        # --- END NEW TARGET VARIABLE DEFINITION ---
        
        # Add simple derived features
        df['price_change'] = df['close'].pct_change()
        df['volatility'] = df['price_change'].rolling(12).std()
        df['rsi_momentum'] = df['rsi'].diff()
        
        # --- NEW FEATURES FOR IMPROVED ACCURACY ---
        # Lagged Price Features
        df['close_lag_1'] = df['close'].shift(1)
        df['close_lag_2'] = df['close'].shift(2)
        df['close_lag_3'] = df['close'].shift(3)
        
        # Volume-Based Indicator (On-Balance Volume)
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        
        # Additional Volatility Measures
        df['rolling_std_7'] = df['close'].rolling(window=7).std()
        df['rolling_std_21'] = df['close'].rolling(window=21).std()
        # --- END NEW FEATURES ---

        # Clean data (after target definition and new features)
        df = df.dropna()
        
        # SIMPLE FEATURE SET (now with more features)
        
        # Verify all features exist
        missing_features = [col for col in feature_cols if col not in df.columns]
        if missing_features:
            print(f"⚠️  Missing features: {missing_features}")
            # Remove missing features from list
            feature_cols = [col for col in feature_cols if col in df.columns]
        
        features = df[feature_cols]
        targets = df['target']
        
        print(f"✅ Prepared {len(features)} samples with {len(feature_cols)} features")
        print(f"📊 Target distribution: {targets.value_counts().to_dict()}")
        
        return features, targets
    
    def get_feature_info(self):
        """Get information about features for debugging"""
        df = self.load_data()
        df = self.indicators.add_all_indicators(df)
        
        print("Available columns after indicators:")
        for i, col in enumerate(df.columns, 1):
            print(f"{i:2d}. {col}")
        
        return list(df.columns)

if __name__ == "__main__":
    processor = DataProcessor()
    
    # Debug: show available features
    available_features = processor.get_feature_info()
    
    # Process training data
    features, targets = processor.prepare_training_data()
    print("\n🚀 Simple data processing complete!")

