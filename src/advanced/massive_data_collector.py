import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import sqlite3
import requests
import time

class MassiveDataCollector:
    """Collect massive forex datasets for serious AI training"""
    
    def __init__(self, db_path="data/massive_forex_data.db"):
        self.db_path = db_path
        self.setup_database()
    
    def setup_database(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS forex_data (
                timestamp TEXT,
                symbol TEXT,
                interval TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                PRIMARY KEY (timestamp, symbol, interval)
            )
        ''')
        conn.close()
        
    def download_multiple_pairs_multiple_timeframes(self):
        """Download massive dataset: multiple pairs, multiple timeframes, 5+ years"""
        
        # Major forex pairs for cross-learning
        pairs = [
            "USDJPY=X",  # Primary
            "EURUSD=X",  # Most liquid
            "GBPUSD=X",  # Volatile
            "AUDUSD=X",  # Commodity currency
            "USDCAD=X",  # Oil correlation
            "USDCHF=X",  # Safe haven
            "NZDUSD=X",  # Risk-on/off
            "EURJPY=X",  # Cross pair
        ]
        
        # Multiple timeframes for better patterns
        intervals = {
            "1h": "1h",    # 43,800 samples per year
            "4h": "4h",    # 10,950 samples per year  
            "1d": "1d"     # 365 samples per year
        }
        
        # 5 years of data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * 5)
        
        total_samples = 0
        
        for pair in pairs:
            for interval_name, interval_code in intervals.items():
                print(f"Downloading {pair} {interval_name} data...")
                
                try:
                    data = yf.download(pair, start=start_date, end=end_date, interval=interval_code)
                    
                    if not data.empty:
                        # Clean column names
                        if isinstance(data.columns, pd.MultiIndex):
                            data.columns = data.columns.get_level_values(0)
                        data.columns = data.columns.str.lower()
                        
                        # Add metadata
                        data['symbol'] = pair
                        data['interval'] = interval_name
                        
                        # Save to database
                        conn = sqlite3.connect(self.db_path)
                        data.to_sql('forex_data', conn, if_exists='append', index=True)
                        conn.close()
                        
                        samples = len(data)
                        total_samples += samples
                        print(f"✅ {pair} {interval_name}: {samples} candles")
                        
                except Exception as e:
                    print(f"❌ Error downloading {pair} {interval_name}: {e}")
                
                time.sleep(1)  # Rate limiting
        
        print(f"\n🚀 MASSIVE DATASET COMPLETE: {total_samples:,} total samples!")
        return total_samples
    
    def get_alternative_data_sources(self):
        """Add economic indicators and sentiment data"""
        # You can add APIs for:
        # - FRED economic data
        # - VIX volatility index
        # - Interest rate differentials
        # - News sentiment scores
        pass

if __name__ == "__main__":
    collector = MassiveDataCollector()
    total_samples = collector.download_multiple_pairs_multiple_timeframes()
    print(f"Ready for massive AI training with {total_samples:,} samples!")