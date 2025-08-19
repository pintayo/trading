import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from trading_system import USDJPYTradingSystem
import time
from datetime import datetime

def run_paper_trading_with_learning():
    """Run paper trading with continuous improvement"""
    
    print("🚀 Starting USD/JPY Paper Trading System")
    print(f"💡 Model Accuracy: 51.30% (Profitable with 2:1 R/R)")
    print(f"⏰ Started: {datetime.now()}")
    
    # Initialize system with paper trading
    system = USDJPYTradingSystem(account_balance=1000, paper_trading=True)
    
    trade_count = 0
    session_count = 0
    
    try:
        while True:
            session_count += 1
            print(f"\n=== Session #{session_count} ===")
            
            # Run trading session
            signal_generated = system.run_trading_session()
            
            if signal_generated:
                trade_count += 1
                print(f"📈 Total signals generated: {trade_count}")
            
            # Show some stats every 10 sessions
            if session_count % 10 == 0:
                print(f"📊 Sessions completed: {session_count}")
                print(f"🎯 Signals generated: {trade_count}")
                print(f"📈 Signal rate: {(trade_count/session_count)*100:.1f}%")
            
            # Wait 4 hours (or 5 minutes for testing)
            print("⏳ Waiting for next 4-hour candle...")
            time.sleep(300)  # 5 minutes for testing (change to 14400 for real 4h)
            
    except KeyboardInterrupt:
        print(f"\n📊 Final Stats:")
        print(f"Sessions: {session_count}")
        print(f"Signals: {trade_count}")
        print("👋 Paper trading stopped")

if __name__ == "__main__":
    run_paper_trading_with_learning()