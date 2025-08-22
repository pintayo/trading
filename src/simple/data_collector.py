import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import sqlite3
from config.trading_config import LOOKBACK_YEARS # Keep LOOKBACK_YEARS

class ForexDataCollector:
    def __init__(self, db_path="data/forex_data.db"):
        self.db_path = db_path
        self.setup_database()
    
    def setup_database(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS price_data (
                timestamp TEXT PRIMARY KEY,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER
            )
        ''')
        conn.close()
    
    def download_historical_data(self, symbol, interval, years=LOOKBACK_YEARS):
        print(f"Downloading {years} years of {symbol} {interval} data...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * years)
        
        data = yf.download(symbol, start=start_date, end=end_date, interval=interval, auto_adjust=True)
        
        if data.empty:
            raise ValueError(f"No data downloaded for {symbol}")
        
        # Flatten column names if multi-level
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        # Ensure proper column names
        data.columns = data.columns.str.lower()
        
        self.save_to_database(data)
        print(f"Downloaded {len(data)} 4-hour candles")
        return data
        
    def save_to_database(self, data):
        conn = sqlite3.connect(self.db_path)
        data.to_sql('price_data', conn, if_exists='replace', index=True)
        conn.close()
        print("Data saved to database")

if __name__ == "__main__":
    from config.trading_config import SYMBOL, TIMEFRAME # Import for standalone run
    collector = ForexDataCollector()
    collector.download_historical_data(symbol=SYMBOL, interval=TIMEFRAME)