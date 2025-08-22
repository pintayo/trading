import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.simple.data_processor import DataProcessor
from .ensemble_model import EnsembleForexModel
from config.model_config import ADVANCED_CONFIG, get_model_paths # Import ADVANCED_CONFIG and get_model_paths for standalone run
from config.trading_config import TIMEFRAME # Import TIMEFRAME
import numpy as np

def train_advanced_model(config_params, pair_config):
    print(f"Starting Advanced Ensemble Training for {pair_config['name']} (Target: 70% Accuracy)...")
    
    # Get dynamic model and scaler paths
    paths = get_model_paths(model_type="advanced", pair_name=pair_config['name'])
    model_path = paths['model_path']
    scaler_path = paths['scaler_path']
    attention_path = paths['attention_path']
    cnn_path = paths['cnn_path']
    xgboost_path = paths['xgboost_path']

    # Load and process data using DataProcessor
    processor = DataProcessor(symbol=pair_config['symbol'], model_type="advanced", interval=TIMEFRAME)
    df = processor.load_data()
    
    # --- Refined Target Variable Definition (copied from simple/data_processor.py) ---
    # Calculate future close price
    df['future_close'] = df['close'].shift(-config_params['target_future_candles'])
    
    # Define thresholds
    upper_bound = df['close'] * (1 + config_params['target_percent_threshold'])
    lower_bound = df['close'] * (1 - config_params['target_percent_threshold'])
    
    # Assign target: 1 for up, 0 for down
    df['target'] = np.nan # Initialize with NaN
    df.loc[df['future_close'] >= upper_bound, 'target'] = 1
    df.loc[df['future_close'] <= lower_bound, 'target'] = 0
    
    # Drop rows where target is NaN (i.e., price stayed within the threshold)
    df = df.dropna(subset=['target'])
    df['target'] = df['target'].astype(int)
    # --- End Refined Target Variable Definition ---
    
    print(f"Dataset size after target definition: {len(df)} samples")
    
    # Initialize and train ensemble model
    ensemble_model = EnsembleForexModel()
    ensemble_model.train(df, target_col='target', paths=paths)
    
    # Save the trained model
    ensemble_model.save_model(paths=paths) # Pass paths to save_model
    
    print("🚀 Advanced Ensemble Training Complete!")
    print("Target: 70% accuracy for forex prediction")

if __name__ == "__main__":
    # For standalone testing, assume advanced model and USD/JPY
    test_pair_config = {
        "name": "USD/JPY",
        "symbol": "USDJPY=X",
        "ibkr_symbol": "USD",
        "ibkr_currency": "JPY",
        "ibkr_exchange": "IDEALPRO"
    }
    train_advanced_model(ADVANCED_CONFIG, test_pair_config)