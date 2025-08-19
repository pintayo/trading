import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from config.trading_config import *

class RiskManager:
    def __init__(self, account_balance):
        self.account_balance = account_balance
        self.daily_loss = 0
        self.open_positions = 0
        
    def can_open_position(self):
        """Check if new position can be opened"""
        if self.open_positions >= MAX_OPEN_POSITIONS:
            return False, "Maximum open positions reached"
        
        if self.daily_loss >= self.account_balance * MAX_DAILY_LOSS:
            return False, "Daily loss limit reached"
            
        return True, "OK"
    
    def calculate_position_size(self, atr_value, entry_price):
        """Calculate position size based on risk management rules"""
        # Risk amount (2% of account)
        risk_amount = self.account_balance * MAX_POSITION_SIZE
        
        # Stop loss distance
        stop_loss_distance = atr_value * STOP_LOSS_ATR_MULTIPLIER
        
        # Position size calculation
        position_size = risk_amount / stop_loss_distance
        
        # Ensure position doesn't exceed account limits
        max_position_value = self.account_balance * 0.1  # Max 10% of account per position
        max_position_size = max_position_value / entry_price
        
        return min(position_size, max_position_size)
    
    def calculate_stop_loss(self, entry_price, atr_value, direction):
        """Calculate stop loss level"""
        stop_distance = atr_value * STOP_LOSS_ATR_MULTIPLIER
        
        if direction == "long":
            return entry_price - stop_distance
        else:
            return entry_price + stop_distance
    
    def calculate_take_profit(self, entry_price, atr_value, direction):
        """Calculate take profit level"""
        profit_distance = atr_value * TAKE_PROFIT_ATR_MULTIPLIER
        
        if direction == "long":
            return entry_price + profit_distance
        else:
            return entry_price - profit_distance
    
    def update_daily_loss(self, trade_pnl):
        """Update daily loss tracking"""
        if trade_pnl < 0:
            self.daily_loss += abs(trade_pnl)
    
    def reset_daily_tracking(self):
        """Reset daily tracking (call at start of each day)"""
        self.daily_loss = 0
    
    def update_account_balance(self, new_balance):
        """Update account balance"""
        self.account_balance = new_balance