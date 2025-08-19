import pandas as pd
import sqlite3
from technical_indicators import TechnicalIndicators

class DataProcessor:
    def __init__(self, db_path="data/usdjpy_data.db"):
        self.db_path = db_path
        self.indicators = TechnicalIndicators()
    
    def load_data(self):
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM price_data", conn, parse_dates=True, index_col=0)
        conn.close()
        
        # Handle potential column naming issues
        df.columns = df.columns.str.lower()
        
        return df
    
    def prepare_training_data(self):
        df = self.load_data()
        df = self.indicators.add_all_indicators(df)
        
        # Create target variable (1 if price goes up in next 4 hours, 0 if down)
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        
        # Create additional features
        df['price_change'] = df['close'].pct_change()
        df['volatility'] = df['price_change'].rolling(window=12).std()
        
        # Remove NaN values
        df = df.dropna()
        
        # Feature columns for AI model
        feature_cols = [
            'open', 'high', 'low', 'close', 'volume',
            'rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_middle', 'bb_lower',
            'atr', 'bb_position', 'price_change', 'volatility'
        ]
        
        print(f"Prepared {len(df)} samples with {len(feature_cols)} features")
        return df[feature_cols], df['target']

if __name__ == "__main__":
    processor = DataProcessor()
    features, targets = processor.prepare_training_data()
    print("Data processing complete!")