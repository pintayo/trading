# src/data_collector.py
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import sqlite3

class USDJPYDataCollector:
    def __init__(self, db_path="data/usdjpy_data.db"):
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
    
    def download_historical_data(self, years=2):
        # Download 2 years of USD/JPY data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * years)
        
        data = yf.download("USDJPY=X", start=start_date, end=end_date, interval="4h")
        self.save_to_database(data)
        print(f"Downloaded {len(data)} 4-hour candles")
        
    def save_to_database(self, data):
        conn = sqlite3.connect(self.db_path)
        data.to_sql('price_data', conn, if_exists='replace', index=True)
        conn.close()

# Usage
if __name__ == "__main__":
    collector = USDJPYDataCollector()
    collector.download_historical_data()
