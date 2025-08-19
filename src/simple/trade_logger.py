import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import sqlite3
import pandas as pd
from datetime import datetime
import json

class TradeLogger:
    def __init__(self, db_path="logs/trading_logs.db"):
        self.db_path = db_path
        self.setup_database()
    
    def setup_database(self):
        conn = sqlite3.connect(self.db_path)
        
        # Trades table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                direction TEXT,
                entry_price REAL,
                exit_price REAL,
                position_size REAL,
                stop_loss REAL,
                take_profit REAL,
                pnl REAL,
                pnl_percent REAL,
                duration_hours REAL,
                exit_reason TEXT,
                ai_confidence REAL,
                market_conditions TEXT,
                trade_thesis TEXT
            )
        ''')
        
        # Performance tracking table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS performance (
                date TEXT PRIMARY KEY,
                total_trades INTEGER,
                winning_trades INTEGER,
                losing_trades INTEGER,
                total_pnl REAL,
                win_rate REAL,
                avg_win REAL,
                avg_loss REAL,
                profit_factor REAL,
                account_balance REAL
            )
        ''')
        
        conn.close()
    
    def log_trade(self, trade_data):
        """Log a completed trade"""
        conn = sqlite3.connect(self.db_path)
        
        columns = ', '.join(trade_data.keys())
        placeholders = ', '.join(['?' for _ in trade_data])
        values = list(trade_data.values())
        
        conn.execute(f'''
            INSERT INTO trades ({columns})
            VALUES ({placeholders})
        ''', values)
        
        conn.commit()
        conn.close()
        
        print(f"Trade logged: {trade_data['direction']} {trade_data['symbol']} PnL: {trade_data['pnl']:.2f}")
    
    def update_daily_performance(self, date, account_balance):
        """Update daily performance metrics"""
        conn = sqlite3.connect(self.db_path)
        
        # Get trades for the day
        trades_df = pd.read_sql_query('''
            SELECT * FROM trades 
            WHERE DATE(timestamp) = ?
        ''', conn, params=[date])
        
        if len(trades_df) > 0:
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['pnl'] > 0])
            losing_trades = len(trades_df[trades_df['pnl'] < 0])
            total_pnl = trades_df['pnl'].sum()
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
            avg_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].mean()) if losing_trades > 0 else 0
            profit_factor = (winning_trades * avg_win) / (losing_trades * avg_loss) if losing_trades > 0 else float('inf')
            
            conn.execute('''
                INSERT OR REPLACE INTO performance 
                (date, total_trades, winning_trades, losing_trades, total_pnl, 
                 win_rate, avg_win, avg_loss, profit_factor, account_balance)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (date, total_trades, winning_trades, losing_trades, total_pnl,
                  win_rate, avg_win, avg_loss, profit_factor, account_balance))
            
            conn.commit()
        
        conn.close()
    
    def get_performance_summary(self, days=30):
        """Get performance summary for last N days"""
        conn = sqlite3.connect(self.db_path)
        
        summary = pd.read_sql_query('''
            SELECT 
                COUNT(*) as total_trades,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
                SUM(pnl) as total_pnl,
                AVG(CASE WHEN pnl > 0 THEN pnl END) as avg_win,
                AVG(CASE WHEN pnl < 0 THEN ABS(pnl) END) as avg_loss
            FROM trades 
            WHERE timestamp >= date('now', '-{} days')
        '''.format(days), conn)
        
        conn.close()
        
        if summary['total_trades'].iloc[0] > 0:
            summary['win_rate'] = summary['winning_trades'] / summary['total_trades']
            summary['profit_factor'] = (summary['winning_trades'] * summary['avg_win']) / (summary['losing_trades'] * summary['avg_loss'])
        
        return summary