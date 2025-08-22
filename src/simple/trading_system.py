import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
import traceback
from .lstm_model import ForexLSTMModel, ModelTrainer
from .data_processor import DataProcessor
from .ibkr_integration import IBKRTrading
from .trade_logger import TradeLogger
from .technical_indicators import TechnicalIndicators
from config.model_config import SIMPLE_CONFIG, ADVANCED_CONFIG, get_model_paths
from src.advanced.ensemble_model import EnsembleForexModel
from config.trading_config import *

class TradingSystem:
    def __init__(self, model_type, pair_config, account_balance=1000, paper_trading=True):
        self.model_type = model_type
        self.pair_config = pair_config
        self.paths = get_model_paths(model_type=model_type, pair_name=pair_config['name'])

        if model_type == "simple":
            self.model = ForexLSTMModel(
                input_size=SIMPLE_CONFIG['input_size'],
                hidden_size=SIMPLE_CONFIG['hidden_size'],
                num_layers=SIMPLE_CONFIG['num_layers'],
                output_size=SIMPLE_CONFIG['output_size'],
                dropout=SIMPLE_CONFIG['dropout']
            )
            self.model.load_state_dict(torch.load(self.paths['model_path'], weights_only=False))
            self.model.eval()
            
            self.trainer = ModelTrainer(self.model)
            self.trainer.load_scaler(self.paths['scaler_path'])
        elif model_type == "advanced":
            self.model = EnsembleForexModel()
            self.model.load_model(self.paths)

        self.ibkr = IBKRTrading(account_balance, paper_trading, pair_config)
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
        
        # Get recent 1-hour data for 1 day
        df = self.ibkr.get_historical_data(contract, duration="3 D", barSize="1 hour")
        
        if df.empty:
            raise ValueError("No historical data received")
        
        # Add technical indicators
        df = self.indicators.add_all_indicators(df)
        
        # Add additional features (matching data_processor.py)
        df['price_change'] = df['close'].pct_change()
        df['volatility'] = df['price_change'].rolling(12).std()
        df['rsi_momentum'] = df['rsi'].diff()
        df['close_lag_1'] = df['close'].shift(1)
        df['close_lag_2'] = df['close'].shift(2)
        df['close_lag_3'] = df['close'].shift(3)
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        df['rolling_std_7'] = df['close'].rolling(window=7).std()
        df['rolling_std_21'] = df['close'].rolling(window=21).std()
        
        return df.dropna()
    
    def prepare_model_input(self, df):
        """Prepare data for model prediction"""
        if self.model_type == 'simple':
            config = SIMPLE_CONFIG
        else:
            config = ADVANCED_CONFIG

        feature_cols = config['feature_columns']
        sequence_length = config['sequence_length']
        
        # Get last sequence_length periods
        features = df[feature_cols].tail(sequence_length).values
        
        # Normalize using saved scaler
        # The scaler expects (n_samples, n_features), where n_samples is sequence_length
        features_scaled = self.trainer.scaler.transform(features)
        
        # Reshape back for LSTM: (1, sequence_length, num_features)
        features_scaled_reshaped = features_scaled.reshape(1, sequence_length, len(feature_cols))
        
        return torch.FloatTensor(features_scaled_reshaped)
    
    def generate_signal(self):
        """Generate trading signal from AI model"""
        try:
            if self.model_type == 'simple':
                config = SIMPLE_CONFIG
            else:
                config = ADVANCED_CONFIG
            sequence_length = config['sequence_length']

            # Get latest data
            df = self.get_latest_data()
            
            if len(df) < sequence_length:
                print("Insufficient data for prediction")
                return None, None, None, None, None
            
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
            print(f"Error generating signal: {type(e).__name__}: {e}")
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
            traceback.print_exc() # Print full traceback
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

    def log_prediction_outcome_after_4h(self, original_signal, original_confidence, original_price, current_price, market_context):
        """Check if our 4-hour prediction was correct"""
        actual_outcome = 1 if current_price > original_price else 0
        
        # Log the outcome for learning
        self.continuous_learner.log_prediction_outcome(
            original_confidence, 
            actual_outcome,
            market_context,
            None  # No trade details yet
        )

# Example usage
def run_trading_system_cli(model_type, pair_config):
    print(f"\n Starting {model_type} Trading System for {pair_config['name']}")
    print(f" Model Accuracy: 51.30% (Profitable with 2:1 R/R)") # Placeholder
    print(f"⏰ Started: {datetime.now()}")
    
    # Initialize trading system for paper trading
    system = TradingSystem(model_type=model_type, pair_config=pair_config, account_balance=1000, paper_trading=True)
    
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
    except Exception as e:
        print(f"Trading system error: {e}")
        traceback.print_exc() # Print full traceback
    finally:
        if system.ibkr.connected:
            system.ibkr.disconnect()

if __name__ == "__main__":
    # For standalone testing, assume simple model and USD/JPY
    from config.model_config import SIMPLE_CONFIG
    from config.trading_config import SYMBOL, IBKR_SYMBOL, IBKR_CURRENCY, IBKR_EXCHANGE
    test_pair_config = {
        "name": "USD/JPY",
        "symbol": SYMBOL,
        "ibkr_symbol": IBKR_SYMBOL,
        "ibkr_currency": IBKR_CURRENCY,
        "ibkr_exchange": IBKR_EXCHANGE
    }
    run_trading_system_cli(model_type="simple", pair_config=test_pair_config)