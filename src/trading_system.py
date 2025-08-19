import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
from lstm_model import USDJPYLSTMModel, ModelTrainer
from data_processor import DataProcessor
from ibkr_integration import IBKRTrading
from trade_logger import TradeLogger
from technical_indicators import TechnicalIndicators
from config.model_config import MODEL_PATH, SCALER_PATH
from config.trading_config import *

class USDJPYTradingSystem:
    def __init__(self, account_balance=1000, paper_trading=True):
        self.model = USDJPYLSTMModel()
        self.model.load_state_dict(torch.load(MODEL_PATH))
        self.model.eval()
        
        self.trainer = ModelTrainer()
        self.trainer.load_scaler()
        
        self.ibkr = IBKRTrading(account_balance, paper_trading)
        self.trade_logger = TradeLogger()
        self.indicators = TechnicalIndicators()
        
        self.last_signal_time = None
        self.position_open = False
        
    def get_latest_data(self):
        """Get latest market data and calculate features"""
        if not self.ibkr.connected:
            if not self.ibkr.connect():
                raise ConnectionError("Cannot connect to IBKR")
        
        contract = self.ibkr.create_forex_contract()
        
        # Get recent 4-hour data
        df = self.ibkr.get_historical_data(contract, duration="10 D", barSize="4 hours")
        
        if df.empty:
            raise ValueError("No historical data received")
        
        # Add technical indicators
        df = self.indicators.add_all_indicators(df)
        
        # Add additional features
        df['price_change'] = df['close'].pct_change()
        df['volatility'] = df['price_change'].rolling(window=12).std()
        
        return df.dropna()
    
    def prepare_model_input(self, df):
        """Prepare data for model prediction"""
        feature_cols = [
            'open', 'high', 'low', 'close', 'volume',
            'rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_middle', 'bb_lower',
            'atr', 'bb_position', 'price_change', 'volatility'
        ]
        
        # Get last SEQUENCE_LENGTH periods
        features = df[feature_cols].tail(SEQUENCE_LENGTH).values
        
        # Normalize using saved scaler
        features_scaled = self.trainer.scaler.transform(features.reshape(1, -1))
        features_scaled = features_scaled.reshape(1, SEQUENCE_LENGTH, len(feature_cols))
        
        return torch.FloatTensor(features_scaled)
    
    def generate_signal(self):
        """Generate trading signal from AI model"""
        try:
            # Get latest data
            df = self.get_latest_data()
            
            if len(df) < SEQUENCE_LENGTH:
                print("Insufficient data for prediction")
                return None, None, None
            
            # Prepare input for model
            model_input = self.prepare_model_input(df)
            
            # Generate prediction
            with torch.no_grad():
                prediction = self.model(model_input)
                signal = prediction.item()
                confidence = abs(signal - 0.5) * 2  # Convert to confidence score
            
            # Get current price and ATR
            current_price = df['close'].iloc[-1]
            atr_value = df['atr'].iloc[-1]
            
            print(f"AI Signal: {signal:.3f}, Confidence: {confidence:.3f}")
            print(f"Current Price: {current_price:.3f}, ATR: {atr_value:.3f}")
            
            return signal, confidence, current_price, atr_value, df
            
        except Exception as e:
            print(f"Error generating signal: {e}")
            return None, None, None, None, None
    
    def should_trade(self, signal, confidence):
        """Determine if signal is strong enough to trade"""
        # Minimum confidence threshold
        if confidence < 0.3:
            return False, "Low confidence"
        
        # Strong buy signal
        if signal > 0.6:
            return True, "Strong buy signal"
        
        # Strong sell signal  
        if signal < 0.4:
            return True, "Strong sell signal"
        
        return False, "Signal not strong enough"
    
    def run_trading_session(self):
        """Run one trading session"""
        print(f"\n--- Trading Session: {datetime.now()} ---")
        
        # Generate signal
        signal, confidence, current_price, atr_value, df = self.generate_signal()
        
        if signal is None:
            return
        
        # Check if we should trade
        should_trade, reason = self.should_trade(signal, confidence)
        
        if not should_trade:
            print(f"No trade: {reason}")
            return
        
        # Check trading session times (optional filter)
        current_hour = datetime.now().hour
        if current_hour < 6 or current_hour > 22:  # Outside major trading hours
            print("Outside trading hours")
            return
        
        # Execute trade
        if not self.position_open:
            trade_data = self.ibkr.execute_ai_signal(signal, confidence, current_price, atr_value)
            
            if trade_data:
                self.position_open = True
                self.last_signal_time = datetime.now()
                print("Trade executed successfully")
            else:
                print("Trade execution failed")
    
    def monitor_positions(self):
        """Monitor open positions and update trade logs"""
        # This would monitor IBKR positions and update trade_logger
        # when positions are closed (simplified for brevity)
        pass
    
    def run_continuous(self, check_interval_minutes=240):  # Check every 4 hours
        """Run trading system continuously"""
        print("Starting continuous USD/JPY trading system...")
        print(f"Check interval: {check_interval_minutes} minutes")
        
        try:
            while True:
                self.run_trading_session()
                self.monitor_positions()
                
                # Wait for next check
                print(f"Waiting {check_interval_minutes} minutes until next check...")
                time.sleep(check_interval_minutes * 60)
                
        except KeyboardInterrupt:
            print("\nTrading system stopped by user")
        except Exception as e:
            print(f"Trading system error: {e}")
        finally:
            if self.ibkr.connected:
                self.ibkr.disconnect()

# Example usage
if __name__ == "__main__":
    # Initialize trading system for paper trading
    trading_system = USDJPYTradingSystem(account_balance=1000, paper_trading=True)
    
    # Run single session
    trading_system.run_trading_session()
    
    # Or run continuously (uncomment below)
    # trading_system.run_continuous()