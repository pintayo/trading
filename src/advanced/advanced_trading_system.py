import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
from ensemble_model import EnsembleForexModel
from advanced_features import AdvancedFeatureEngineering
from ibkr_integration import IBKRTrading
from trade_logger import TradeLogger
from config.trading_config import *
from config.model_config import SEQUENCE_LENGTH

class USDJPYTradingSystem:
    def __init__(self, account_balance=1000, paper_trading=True):
        self.model = EnsembleForexModel()
        self.model.load_model() # Ensemble model handles its own loading
        
        self.ibkr = IBKRTrading(account_balance, paper_trading)
        self.trade_logger = TradeLogger()
        self.feature_engineer = AdvancedFeatureEngineering()
        
        self.last_signal_time = None
        self.position_open = False
        
    def get_latest_data(self):
        """Get latest market data and calculate advanced features"""
        if not self.ibkr.connected:
            if not self.ibkr.connect():
                raise ConnectionError("Cannot connect to IBKR")
        
        contract = self.ibkr.create_forex_contract()
        
        # Get recent 4-hour data (more data for advanced features)
        df = self.ibkr.get_historical_data(contract, duration="30 D", barSize="4 hours")
        
        if df.empty:
            raise ValueError("No historical data received")
        
        # Create advanced features
        df = self.feature_engineer.create_advanced_features(df.copy())
        
        # Select only the features the ensemble model was trained on
        # This assumes self.model.selected_features is populated after load_model()
        if self.model.selected_features is None:
            raise ValueError("Ensemble model's selected_features not loaded. Ensure model is trained and saved correctly.")
        
        # Ensure the DataFrame has enough rows after feature engineering and dropping NaNs
        # The feature engineering process might introduce NaNs at the beginning.
        # We need enough data points to form the required sequence_length for LSTM inputs.
        df = df.dropna()
        
        return df[self.model.selected_features]
    
    def prepare_ensemble_inputs(self, df):
        """Prepare data for ensemble model prediction (LSTM and XGBoost parts)"""
        features = df.values
        
        # Prepare LSTM data
        lstm_X = []
        if len(features) >= SEQUENCE_LENGTH:
            lstm_X.append(features[-SEQUENCE_LENGTH:])
        else:
            raise ValueError(f"Insufficient data for LSTM sequence. Need at least {SEQUENCE_LENGTH} data points, got {len(features)}.")
        lstm_X = np.array(lstm_X)
        
        # Prepare XGBoost data (last 12 periods flattened)
        xgb_X = []
        if len(features) >= 12: # XGBoost uses last 12 periods
            xgb_X.append(features[-12:].flatten())
        else:
            raise ValueError(f"Insufficient data for XGBoost input. Need at least 12 data points, got {len(features)}.")
        xgb_X = np.array(xgb_X)
        
        return torch.FloatTensor(lstm_X), xgb_X
    
    def generate_signal(self):
        """Generate trading signal from AI ensemble model"""
        try:
            # Get latest data with advanced features
            df = self.get_latest_data()
            
            # Ensure enough data for both LSTM and XGBoost inputs
            if len(df) < max(SEQUENCE_LENGTH, 12): # Assuming XGBoost needs at least 12
                print("Insufficient data for ensemble prediction.")
                return None, None, None, None, None
            
            # Prepare inputs for ensemble model
            lstm_input, xgb_input = self.prepare_ensemble_inputs(df)
            
            # Generate prediction
            with torch.no_grad():
                prediction = self.model.predict(lstm_input, xgb_input)
                signal = prediction.item() # Ensemble predict returns a single value
                confidence = abs(signal - 0.5) * 2  # Convert to confidence score
            
            # Get current price and ATR from the original (un-engineered) data
            # Need to fetch raw data again or pass it through
            # For simplicity, let's assume the last row of the feature-engineered df still corresponds to current price/ATR
            # A more robust solution would pass original OHLCV data alongside engineered features.
            # For now, we'll get it from the raw historical data before feature engineering.
            
            # Re-fetch raw data to get current price and ATR
            contract = self.ibkr.create_forex_contract()
            raw_df = self.ibkr.get_historical_data(contract, duration="1 D", barSize="4 hours")
            raw_df = raw_df.dropna()
            
            if raw_df.empty:
                print("Could not get raw data for current price/ATR.")
                return None, None, None, None, None
            
            current_price = raw_df['close'].iloc[-1]
            
            # Recalculate ATR on raw data for consistency with risk manager
            # This is a simplified approach; ideally, ATR would be part of the selected features
            # and passed through. For now, we'll use the simple TechnicalIndicators class.
            from technical_indicators import TechnicalIndicators # Import locally to avoid circular dependency
            ti = TechnicalIndicators()
            raw_df = ti.add_all_indicators(raw_df)
            atr_value = raw_df['atr'].iloc[-1]
            
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
        
        return False, "Strong sell signal" # Corrected typo
        
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
                
    def generate_signal_with_learning(self):
        """Generate signal and log for continuous learning"""
        signal, confidence, current_price, atr_value, df = self.generate_signal()
        
        if signal is None:
            return None, None, None, None, None
        
        # Wait 4 hours, then check if prediction was correct
        # (This would be called periodically)
        
        return signal, confidence, current_price, atr_value, df

# Example usage
if __name__ == "__main__":
    # Initialize trading system for paper trading
    trading_system = USDJPYTradingSystem(account_balance=1000, paper_trading=True)
    
    # Run single session
    trading_system.run_trading_session()
    
    # Or run continuously (uncomment below)
    # trading_system.run_continuous()
