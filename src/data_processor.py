import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
        
        print("Columns in database:", df.columns.tolist())
        print("First few rows:")
        print(df.head())
        
        # Handle different possible column structures
        if 'Datetime' in df.columns:
            df['Datetime'] = pd.to_datetime(df['Datetime'])
            df.set_index('Datetime', inplace=True)
        elif 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        elif 'index' in df.columns:
            df['index'] = pd.to_datetime(df['index'])
            df.set_index('index', inplace=True)
        else:
            # If no clear datetime column, use the first column as datetime index
            datetime_col = df.columns[0]
            df[datetime_col] = pd.to_datetime(df[datetime_col])
            df.set_index(datetime_col, inplace=True)
        
        # Handle multi-level column names like ('Close', 'USDJPY=X')
        if any(isinstance(col, tuple) or '(' in str(col) for col in df.columns):
            # Clean up column names - extract just the price type
            new_columns = []
            for col in df.columns:
                if isinstance(col, tuple):
                    new_columns.append(col[0].lower())
                elif '(' in str(col) and ')' in str(col):
                    # Extract the first part from strings like "('Close', 'USDJPY=X')"
                    clean_col = str(col).split(',').replace('(', '').replace("'", '').strip().lower()
                    new_columns.append(clean_col)
                else:
                    new_columns.append(str(col).lower())
            
            df.columns = new_columns
        else:
            # Simple column name cleanup
            df.columns = df.columns.str.lower()
        
        print("Final columns after processing:", df.columns.tolist())
        print("Index name:", df.index.name)
        print("Data shape:", df.shape)
        
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