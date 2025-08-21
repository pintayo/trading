import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import pandas as pd
import numpy as np
from datetime import datetime

from data_processor import DataProcessor
from lstm_model import USDJPYLSTMModel, ModelTrainer
from risk_manager import RiskManager
from technical_indicators import TechnicalIndicators # Needed for ATR calculation in backtest
from config.model_config import MODEL_PATH, SCALER_PATH, SEQUENCE_LENGTH, TARGET_PERCENT_THRESHOLD
from config.trading_config import MAX_POSITION_SIZE, STOP_LOSS_ATR_MULTIPLIER, TAKE_PROFIT_ATR_MULTIPLIER

class Backtester:
    def __init__(self, initial_balance=1000, paper_trading=True):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.trades = []
        self.open_position = None # To track a single open position
        
        # Load model and scaler
        self.model = USDJPYLSTMModel()
        self.model.load_state_dict(torch.load(MODEL_PATH, weights_only=False))
        self.model.eval()
        
        self.trainer = ModelTrainer()
        self.trainer.load_scaler()
        
        self.risk_manager = RiskManager(self.current_balance) # Risk manager needs current balance
        self.indicators = TechnicalIndicators() # For ATR calculation
        
        print("Backtester initialized. Model and scaler loaded.")

    def _prepare_model_input(self, df_segment):
        # This needs to match the feature columns used in data_processor.py
        feature_cols = [
            'open', 'high', 'low', 'close', 'volume',
            'rsi', 'macd', 'macd_signal', 
            'bb_upper', 'bb_middle', 'bb_lower',
            'atr', 'bb_position',
            'price_change', 'volatility', 'rsi_momentum',
            'close_lag_1', 'close_lag_2', 'close_lag_3',
            'obv',
            'rolling_std_7', 'rolling_std_21'
        ]
        
        features = df_segment[feature_cols].values # This is (SEQUENCE_LENGTH, 22)
        
        # Apply the scaler directly to the 2D features array
        features_scaled = self.trainer.scaler.transform(features) # This will be (SEQUENCE_LENGTH, 22)
        
        # Reshape for LSTM: (1, sequence_length, num_features)
        features_scaled_reshaped = features_scaled.reshape(1, SEQUENCE_LENGTH, len(feature_cols))
        
        return torch.FloatTensor(features_scaled_reshaped)

    def _generate_signal(self, df_segment):
        if len(df_segment) < SEQUENCE_LENGTH:
            return None, None # Not enough data for a signal
        
        model_input = self._prepare_model_input(df_segment)
        
        with torch.no_grad():
            prediction = self.model(model_input)
            signal = prediction.item()
            confidence = abs(signal - 0.5) * 2
        
        return signal, confidence

    def _simulate_trade(self, entry_index, df_full, signal_type, entry_price, atr_value):
        # This function simulates a single trade from entry to exit
        # Exit can be due to SL, TP, or a reverse signal, or end of data
        
        trade_details = {
            'entry_time': df_full.index[entry_index],
            'entry_price': entry_price,
            'signal_type': 'BUY' if signal_type > 0.5 else 'SELL',
            'exit_time': None,
            'exit_price': None,
            'pnl': 0,
            'pnl_percent': 0,
            'duration_candles': 0,
            'exit_reason': 'N/A'
        }
        
        direction = 'long' if trade_details['signal_type'] == 'BUY' else 'short'
        
        # Calculate SL/TP based on entry price and ATR
        stop_loss_price = self.risk_manager.calculate_stop_loss(entry_price, atr_value, direction)
        take_profit_price = self.risk_manager.calculate_take_profit(entry_price, atr_value, direction)
        
        # Simulate price movement candle by candle
        for i in range(entry_index + 1, len(df_full)):
            current_candle = df_full.iloc[i]
            trade_details['duration_candles'] += 1
            
            # Check for Stop Loss or Take Profit hit
            if direction == 'long':
                if current_candle['low'] <= stop_loss_price:
                    trade_details['exit_price'] = stop_loss_price
                    trade_details['exit_reason'] = 'STOP_LOSS'
                    break
                if current_candle['high'] >= take_profit_price:
                    trade_details['exit_price'] = take_profit_price
                    trade_details['exit_reason'] = 'TAKE_PROFIT'
                    break
            elif direction == 'short':
                if current_candle['high'] >= stop_loss_price:
                    trade_details['exit_price'] = stop_loss_price
                    trade_details['exit_reason'] = 'STOP_LOSS'
                    break
                if current_candle['low'] <= take_profit_price:
                    trade_details['exit_price'] = take_profit_price
                    trade_details['exit_reason'] = 'TAKE_PROFIT'
                    break
            
            # Optional: Exit on reverse signal (more complex, for later)
            # Optional: Time-based exit (e.g., close after X candles)
            
            # If end of data reached without SL/TP hit, close at current price
            if i == len(df_full) - 1:
                trade_details['exit_price'] = current_candle['close']
                trade_details['exit_reason'] = 'END_OF_DATA'
        
        # Calculate PnL
        if trade_details['exit_price'] is not None:
            if direction == 'long':
                trade_details['pnl'] = trade_details['exit_price'] - trade_details['entry_price']
            else: # short
                trade_details['pnl'] = trade_details['entry_price'] - trade_details['exit_price']
            
            trade_details['pnl_percent'] = (trade_details['pnl'] / trade_details['entry_price']) * 100
            
            # Update balance (simplified: assuming 1 unit of currency for now)
            self.current_balance += trade_details['pnl']
            self.risk_manager.update_account_balance(self.current_balance) # Update risk manager's balance
            
        trade_details['exit_time'] = df_full.index[i] # Set exit time to the candle where exit occurred
        self.trades.append(trade_details)
        
        return trade_details['exit_reason'] != 'N/A' # Return True if trade was closed

    def run_backtest(self, df_full):
        print(f"Starting backtest with {len(df_full)} candles...")
        
        # Ensure ATR is calculated for the full DataFrame
        df_full = self.indicators.add_all_indicators(df_full)
        df_full = df_full.dropna() # Drop NaNs introduced by indicators
        
        # Re-index after dropping NaNs to ensure iloc works correctly
        df_full = df_full.reset_index(drop=True)
        
        for i in range(len(df_full) - SEQUENCE_LENGTH):
            # Get segment for model input
            df_segment = df_full.iloc[i : i + SEQUENCE_LENGTH]
            
            # Get current price and ATR for risk management
            current_price = df_segment['close'].iloc[-1]
            atr_value = df_segment['atr'].iloc[-1]
            
            # Generate signal
            signal, confidence = self._generate_signal(df_segment)
            
            if signal is None:
                continue # Not enough data or signal not generated
            
            # Check if we should trade (using confidence threshold from trading_system)
            # For backtesting, let's use a fixed confidence threshold for now
            if confidence < 0.3: # Same as in trading_system.py
                continue
            
            # Check if a position is already open
            if self.open_position:
                # For simplicity, if a position is open, we don't open another
                # In a real system, you'd manage multiple positions or close existing ones
                continue
            
            # Simulate trade entry and exit
            trade_closed = self._simulate_trade(i + SEQUENCE_LENGTH -1, df_full, signal, current_price, atr_value)
            
            if trade_closed:
                self.open_position = None # Reset for next trade
            else:
                # If trade is not closed, it means it's still open (e.g., end of data)
                # For simplicity, we'll assume it's closed at the end of the backtest
                pass # Handled by end_of_data logic in _simulate_trade
            
            # Update risk manager's balance (already done in _simulate_trade)
            self.risk_manager.update_account_balance(self.current_balance)
            
        print("Backtest complete.")
        self._calculate_metrics()

    def _calculate_metrics(self):
        total_pnl = sum(trade['pnl'] for trade in self.trades)
        winning_trades = [trade for trade in self.trades if trade['pnl'] > 0]
        losing_trades = [trade for trade in self.trades if trade['pnl'] < 0]
        
        num_winning_trades = len(winning_trades)
        num_losing_trades = len(losing_trades)
        total_trades = len(self.trades)
        
        win_rate = (num_winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        avg_win = sum(trade['pnl'] for trade in winning_trades) / num_winning_trades if num_winning_trades > 0 else 0
        avg_loss = sum(trade['pnl'] for trade in losing_trades) / num_losing_trades if num_losing_trades > 0 else 0
        
        # Max Drawdown
        if not self.trades:
            max_drawdown = 0
        else:
            equity_curve = [self.initial_balance]
            for trade in self.trades:
                equity_curve.append(equity_curve[-1] + trade['pnl'])
            
            peak = equity_curve[0]
            max_drawdown = 0
            for equity in equity_curve:
                if equity > peak:
                    peak = equity
                drawdown = (peak - equity) / peak
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
            max_drawdown *= 100 # Convert to percentage
        
        print("""
--- Backtest Results ---""")
        print(f"Initial Balance: ${self.initial_balance:,.2f}")
        print(f"Final Balance:   ${self.current_balance:,.2f}")
        print(f"Total PnL:       ${total_pnl:,.2f}")
        print(f"Total Trades:    {total_trades}")
        print(f"Winning Trades:  {num_winning_trades}")
        print(f"Losing Trades:   {num_losing_trades}")
        print(f"Win Rate:        {win_rate:.2f}%")
        print(f"Avg Win:         ${avg_win:,.2f}")
        print(f"Avg Loss:        ${avg_loss:,.2f}")
        print(f"Max Drawdown:    {max_drawdown:.2f}%")
        print("------------------------")

if __name__ == "__main__":
    processor = DataProcessor()
    # Load raw data first, then add indicators for backtesting
    # The prepare_training_data method also adds indicators and target
    # For backtesting, we need the full historical data with indicators
    # but without the target variable filtering, as we simulate forward.
    
    # Let's load raw data and then manually add indicators
    df_raw = processor.load_data()
    df_full = processor.indicators.add_all_indicators(df_raw.copy())
    
    # Add the new features manually as prepare_training_data filters samples
    # This needs to match the logic in data_processor.py's prepare_training_data
    df_full['price_change'] = df_full['close'].pct_change()
    df_full['volatility'] = df_full['price_change'].rolling(12).std()
    df_full['rsi_momentum'] = df_full['rsi'].diff()
    df_full['close_lag_1'] = df_full['close'].shift(1)
    df_full['close_lag_2'] = df_full['close'].shift(2)
    df_full['close_lag_3'] = df_full['close'].shift(3)
    df_full['obv'] = (np.sign(df_full['close'].diff()) * df_full['volume']).fillna(0).cumsum()
    df_full['rolling_std_7'] = df_full['close'].rolling(window=7).std()
    df_full['rolling_std_21'] = df_full['close'].rolling(window=21).std()
    
    # Drop NaNs after all feature calculations
    df_full = df_full.dropna()
    
    # Ensure the DataFrame is reset_index for iloc to work correctly in backtest loop
    df_full = df_full.reset_index(drop=True)
    
    print(f"Loaded {len(df_full)} candles for backtesting after feature engineering.")
    
    backtester = Backtester(initial_balance=1000)
    backtester.run_backtest(df_full)
