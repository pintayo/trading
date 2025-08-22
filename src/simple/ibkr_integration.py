import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ib_insync import *
import pandas as pd
from datetime import datetime
import time
from config.trading_config import *
from .trade_logger import TradeLogger
from .risk_manager import RiskManager

class IBKRTrading:
    def __init__(self, account_balance=1000, paper_trading=True, pair_config=None):
        self.ib = IB()
        self.paper_trading = paper_trading
        self.trade_logger = TradeLogger()
        self.risk_manager = RiskManager(account_balance)
        self.connected = False
        self.pair_config = pair_config
        
    def connect(self, host='127.0.0.1', port=7497, clientId=1):
        """Connect to IBKR TWS/Gateway"""
        try:
            # Port 7497 for paper trading, 7496 for live trading
            port = 7497 if self.paper_trading else 7496
            self.ib.connect(host, port, clientId)
            self.connected = True
            print(f"Connected to IBKR ({'Paper' if self.paper_trading else 'Live'} Trading)")
            return True
        except Exception as e:
            print(f"Failed to connect to IBKR: {e}")
            return False
    
    def create_forex_contract(self):
        """Create forex contract using pair_config values"""
        if not self.pair_config:
            raise ValueError("pair_config is not set")
        contract = Forex(symbol=self.pair_config['ibkr_symbol'], currency=self.pair_config['ibkr_currency'], exchange=self.pair_config['ibkr_exchange'])
        print(f"DEBUG: Created contract: {contract}")
        return contract
    
    def get_current_price(self, contract):
        """Get current bid/ask prices"""
        ticker = self.ib.reqMktData(contract)
        time.sleep(2)  # Wait for data
        self.ib.cancelMktData(contract)
        
        if ticker.bid and ticker.ask:
            return float(ticker.bid), float(ticker.ask)
        else:
            raise ValueError("Unable to get current price")
    
    def get_historical_data(self, contract, duration="1 D", barSize="4 hours"):
        """Get historical data for analysis"""
        try:
            # Use a specific endDateTime to avoid potential issues with empty string
            # Request a shorter duration and different bar size for testing
            end_date_time = self.ib.reqCurrentTime().strftime('%Y%m%d %H:%M:%S')
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime=end_date_time,
                durationStr="10 D", # Request 10 days of data
                barSizeSetting="4 hours", # Request 4 hour bars
                whatToShow='MIDPOINT',
                useRTH=True
            )
        except Exception as e:
            print(f"Error requesting historical data: {type(e).__name__}: {e}")
            return pd.DataFrame() # Return empty DataFrame on error
        
        if not bars:
            return pd.DataFrame() # Return empty DataFrame if no bars received
        
        # Check for errors within the bars object
        for bar in bars:
            if hasattr(bar, 'error') and bar.error:
                print(f"IBKR Historical Data Error: {bar.error}")
                return pd.DataFrame() # Return empty DataFrame if error found

        print(f"DEBUG: Bars object received: {bars}") # Added debug print
        print(f"DEBUG: Type of bars: {type(bars)}")
        print(f"DEBUG: Length of bars: {len(bars)}")
        if len(bars) > 0:
            print(f"DEBUG: First bar: {repr(bars[0])}")
            if len(bars) > 1:
                print(f"DEBUG: Second bar: {repr(bars[1])}")

        df = util.df(bars)
        return df
    
    def place_order(self, contract, action, quantity, order_type="MKT", limit_price=None):
        """Place trading order"""
        if order_type == "MKT":
            order = MarketOrder(action, quantity)
        elif order_type == "LMT":
            order = LimitOrder(action, quantity, limit_price)
        else:
            raise ValueError("Unsupported order type")
        
        # Add risk management
        can_trade, reason = self.risk_manager.can_open_position()
        if not can_trade:
            print(f"Order rejected: {reason}")
            return None
        
        trade = self.ib.placeOrder(contract, order)
        print(f"Order placed: {action} {quantity} {contract.symbol}")
        return trade
    
    def create_bracket_order(self, contract, action, quantity, stop_loss, take_profit):
        """Create bracket order with stop loss and take profit"""
        # Parent order
        parent = MarketOrder(action, quantity)
        parent.orderId = self.ib.client.getReqId()
        parent.transmit = False
        
        # Stop loss order
        stop_loss_order = StopOrder(
            'SELL' if action == 'BUY' else 'BUY',
            quantity,
            stop_loss
        )
        stop_loss_order.orderId = parent.orderId + 1
        stop_loss_order.parentId = parent.orderId
        stop_loss_order.transmit = False
        
        # Take profit order
        take_profit_order = LimitOrder(
            'SELL' if action == 'BUY' else 'BUY',
            quantity,
            take_profit
        )
        take_profit_order.orderId = parent.orderId + 2
        take_profit_order.parentId = parent.orderId
        take_profit_order.transmit = True
        
        # Place bracket orders
        parent_trade = self.ib.placeOrder(contract, parent)
        stop_trade = self.ib.placeOrder(contract, stop_loss_order)
        profit_trade = self.ib.placeOrder(contract, take_profit_order)
        
        return parent_trade, stop_trade, profit_trade
    
    def execute_ai_signal(self, signal, confidence, current_price, atr_value):
        """Execute trading signal from AI model"""
        contract = self.create_forex_contract()
        
        # Determine action
        action = "BUY" if signal > 0.5 else "SELL"
        direction = "long" if action == "BUY" else "short"
        
        # Calculate position size
        position_size = self.risk_manager.calculate_position_size(atr_value, current_price)
        
        # Calculate stop loss and take profit
        stop_loss = self.risk_manager.calculate_stop_loss(current_price, atr_value, direction)
        take_profit = self.risk_manager.calculate_take_profit(current_price, atr_value, direction)
        
        # Create trade data for logging
        trade_data = {
            'timestamp': datetime.now().isoformat(),
            'symbol': self.pair_config['name'],
            'direction': direction,
            'entry_price': current_price,
            'position_size': position_size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'ai_confidence': confidence,
            'trade_thesis': f"AI signal: {signal:.3f}, confidence: {confidence:.3f}"
        }
        
        # Place bracket order
        try:
            parent_trade, stop_trade, profit_trade = self.create_bracket_order(
                contract, action, position_size, stop_loss, take_profit
            )
            
            print(f"Bracket order placed: {action} {position_size} USDJPY")
            print(f"Entry: {current_price}, SL: {stop_loss:.3f}, TP: {take_profit:.3f}")
            
            return trade_data
            
        except Exception as e:
            print(f"Error placing order: {e}")
            return None
    
    def disconnect(self):
        """Disconnect from IBKR"""
        if self.connected:
            self.ib.disconnect()
            self.connected = False
            print("Disconnected from IBKR")

# Example usage for paper trading
if __name__ == "__main__":
    trader = IBKRTrading(account_balance=1000, paper_trading=True)
    
    if trader.connect():
        contract = trader.create_forex_contract()
        bid, ask = trader.get_current_price(contract)
        print(f"Current USD/JPY: Bid={bid}, Ask={ask}")
        
        trader.disconnect()