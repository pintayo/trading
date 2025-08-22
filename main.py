import warnings
import os
import pandas as pd
import sys

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module='urllib3')
warnings.filterwarnings("ignore", category=FutureWarning, module='yfinance')
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# Add project root to sys.path for module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.simple.data_collector import ForexDataCollector
from src.advanced.massive_data_collector import MassiveDataCollector
from src.simple.data_processor import DataProcessor
from src.simple.lstm_model import ForexLSTMModel, ModelTrainer
from src.simple.train import train_simple_model # Import refactored simple training function
from src.simple.backtester import Backtester, run_backtest_cli # Import refactored backtesting function
from src.simple.trading_system import TradingSystem, run_trading_system_cli # Import refactored trading system function

from src.advanced.train_ensemble import train_advanced_model # Import refactored advanced training function

from config.model_config import (
    SIMPLE_CONFIG, ADVANCED_CONFIG
)
from config.trading_config import (
    SYMBOL, IBKR_SYMBOL, IBKR_CURRENCY, IBKR_EXCHANGE,
    MAX_POSITION_SIZE, STOP_LOSS_ATR_MULTIPLIER, TAKE_PROFIT_ATR_MULTIPLIER, TIMEFRAME, LOOKBACK_YEARS # Import TIMEFRAME and LOOKBACK_YEARS
)

# --- Configuration Mapping for Forex Pairs ---
FOREX_PAIRS = {
    "1": {
        "name": "USD/JPY",
        "symbol": "USDJPY=X",
        "ibkr_symbol": "USD",
        "ibkr_currency": "JPY",
        "ibkr_exchange": "IDEALPRO"
    },
    "2": {
        "name": "AUD/NZD",
        "symbol": "AUDNZD=X", # Assuming Yahoo Finance symbol
        "ibkr_symbol": "AUD",
        "ibkr_currency": "NZD",
        "ibkr_exchange": "IDEALPRO" # Most forex is IDEALPRO
    },
    "3": {
        "name": "EUR/CHF",
        "symbol": "EURCHF=X", # Assuming Yahoo Finance symbol
        "ibkr_symbol": "EUR",
        "ibkr_currency": "CHF",
        "ibkr_exchange": "IDEALPRO"
    }
}

# --- Refactored Functions ---

def get_historical_data_for_pair(pair_config, model_type):
    print(f"\n--- Collecting Historical Data for {pair_config['name']} ---")
    if model_type == "simple":
        collector = ForexDataCollector()
        df_raw = collector.download_historical_data(symbol=pair_config['symbol'], interval=TIMEFRAME, years=LOOKBACK_YEARS) # Use TIMEFRAME and LOOKBACK_YEARS from config
    elif model_type == "advanced":
        collector = MassiveDataCollector()
        # The massive data collector downloads multiple pairs and timeframes
        # For now, we'll just call it to ensure data is collected
        # In a real scenario, you might want to filter the data for the specific pair
        collector.download_multiple_pairs_multiple_timeframes()
        # For advanced model, we don't return df_raw here as it's handled internally by the ensemble training
        df_raw = None # Or load the specific pair data from the massive DB if needed later

    print(f"Finished collecting data for {pair_config['name']}.")
    return df_raw


def train_model(model_type, pair_config):
    print(f"\n--- Training {model_type} Model for {pair_config['name']} ---")
    
    # For now, we assume data is already collected for the selected pair
    # In future, we might want to pass df_raw here or collect it dynamically
    
    if model_type == "simple":
        # Call the refactored simple model training function
        train_simple_model(SIMPLE_CONFIG, pair_config)

    elif model_type == "advanced":
        # Call the refactored advanced model training function
        train_advanced_model(ADVANCED_CONFIG, pair_config)


def run_backtest(model_type, pair_config):
    print(f"\n--- Running Backtest for {model_type} Model on {pair_config['name']} ---")
    run_backtest_cli(model_type, pair_config)

def connect_to_ibkr(model_type, pair_config):
    print(f"\n--- Connecting to IBKR for {model_type} Model on {pair_config['name']} ---")
    run_trading_system_cli(model_type, pair_config)


# --- Main CLI Logic ---
def main_cli():
    print("\n--- Welcome to the Trading System CLI ---")

    # Step 1: Select Model Type
    model_type_choice = input("Select Model Type (1: Simple, 2: Advanced): ")
    if model_type_choice == "1":
        selected_model_type = "simple"
    elif model_type_choice == "2":
        selected_model_type = "advanced"
    else:
        print("Invalid model type. Exiting.")
        return

    # Step 2: Select Forex Pair
    print("\nSelect Forex Pair:")
    for key, value in FOREX_PAIRS.items():
        print(f"{key}: {value['name']}")
    pair_choice = input("Enter choice: ")
    selected_pair_config = FOREX_PAIRS.get(pair_choice)
    if not selected_pair_config:
        print("Invalid pair choice. Exiting.")
        return

    # Step 3: Select Action
    print("\nSelect Action:")
    print("1: Get Historical Data")
    print("2: Train Model")
    print("3: Run Backtest")
    print("4: Connect to IBKR (Paper Trading)")
    action_choice = input("Enter choice: ")

    # Execute Action
    if action_choice == "1":
        get_historical_data_for_pair(selected_pair_config, selected_model_type)
    elif action_choice == "2":
        train_model(selected_model_type, selected_pair_config)
    elif action_choice == "3":
        run_backtest(selected_model_type, selected_pair_config)
    elif action_choice == "4":
        connect_to_ibkr(selected_model_type, selected_pair_config)
    else:
        print("Invalid action choice. Exiting.")

if __name__ == "__main__":
    main_cli()