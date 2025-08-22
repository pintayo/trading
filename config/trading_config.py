# Trading Configuration - USD/JPY 4-Hour Swing Trading

# Currency Pair
SYMBOL = "USDJPY=X"
IBKR_SYMBOL = "USD"
IBKR_CURRENCY = "JPY"
IBKR_EXCHANGE = "IDEALPRO"

# Trading Parameters
TIMEFRAME = "4h"
SEQUENCE_LENGTH = 24  # 4 days of 4-hour candles
LOOKBACK_YEARS = 2

# Advanced Data Collector Lookback Years
LOOKBACK_YEARS_ADVANCED = {
    "1h": 2,  # Max 2 years for 1-hour data from Yahoo Finance
    "4h": 2,  # Max 2 years for 4-hour data from Yahoo Finance
    "1d": 5   # 5 years for daily data
}

# Alpha Vantage API Key (Get yours from https://www.alphavantage.co/support/#api-key)
ALPHA_VANTAGE_API_KEY = "9ZVTND5WHX5DRDK2"

# Risk Management
MAX_POSITION_SIZE = 0.02  # 2% of account per trade
STOP_LOSS_ATR_MULTIPLIER = 1.5
TAKE_PROFIT_ATR_MULTIPLIER = 3.0
MAX_DAILY_LOSS = 0.05  # 5% of account
MAX_OPEN_POSITIONS = 2

# Performance Targets
TARGET_WIN_RATE = {
    "phase1": 0.55,  # Learning phase (months 1-3)
    "phase2": 0.65,  # Profitability phase (months 4-6)
    "phase3": 0.65   # Scaling phase (months 7+)
}

TARGET_MONTHLY_RETURN = {
    "phase1": 0.0,   # Learning focus
    "phase2": 0.10,  # 8-12% monthly
    "phase3": 0.15   # 15%+ monthly
}

# Trading Schedule
TRADING_SESSIONS = {
    "tokyo": {"start": "00:00", "end": "09:00"},
    "london": {"start": "08:00", "end": "17:00"},
    "ny": {"start": "13:00", "end": "22:00"}
}