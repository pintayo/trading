import pandas as pd
import sqlite3
from technical_indicators import TechnicalIndicators

class DataProcessor:
    def __init__(self, db_path="data/usdjpy_data.db"):
        self.db_path = db_path
        self.indicators = TechnicalIndicators()
    
    def load_data(self):
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM price_data", conn)
        conn.close()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        return df
    
    def prepare_training_data(self):
        df = self.load_data()
        df = self.indicators.add_all_indicators(df)
        
        # Create target variable (1 if price goes up in next 4 hours, 0 if down)
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        
        # Remove NaN values
        df = df.dropna()
        
        # Feature columns for AI model
        feature_cols = ['open', 'high', 'low', 'close', 'volume', 
                       'rsi', 'macd', 'macd_signal', 'bb_upper', 
                       'bb_middle', 'bb_lower', 'atr']
        
        return df[feature_cols], df['target']