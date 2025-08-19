import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import sqlite3
from technical_indicators import TechnicalIndicators

class DataProcessor:
    """Simplified data processor for profitable 52%+ model"""
    
    def __init__(self, db_path="data/usdjpy_data.db"):
        self.db_path = db_path
        self.indicators = TechnicalIndicators()
    
    def load_data(self):
        """Load and clean price data - simplified approach"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM price_data", conn)
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
        
        # Create target (1 = price up in 4 hours, 0 = price down)
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        
        # Add simple derived features
        df['price_change'] = df['close'].pct_change()
        df['volatility'] = df['price_change'].rolling(12).std()
        df['rsi_momentum'] = df['rsi'].diff()
        
        # Clean data
        df = df.dropna()
        
        # SIMPLE FEATURE SET (16 features - proven effective)
        feature_cols = [
            # Price data (5 features)
            'open', 'high', 'low', 'close', 'volume',
            
            # Core technical indicators (8 features)
            'rsi', 'macd', 'macd_signal', 
            'bb_upper', 'bb_middle', 'bb_lower',
            'atr', 'bb_position',
            
            # Simple derived features (3 features)
            'price_change', 'volatility', 'rsi_momentum'
        ]
        
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
    processor = SimpleDataProcessor()
    
    # Debug: show available features
    available_features = processor.get_feature_info()
    
    # Process training data
    features, targets = processor.prepare_training_data()
    print("\n🚀 Simple data processing complete!")
