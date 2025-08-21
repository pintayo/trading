import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data_processor import DataProcessor
from ensemble_model import EnsembleForexModel
from config.model_config import TARGET_PERCENT_THRESHOLD, TARGET_FUTURE_CANDLES
import numpy as np

def main():
    print("Starting Advanced Ensemble Training for 70% Accuracy Target...")
    
    # Load and process data
    processor = DataProcessor()
    df = processor.load_data()
    
    # --- Refined Target Variable Definition (copied from simple/data_processor.py) ---
    # Calculate future close price
    df['future_close'] = df['close'].shift(-TARGET_FUTURE_CANDLES)
    
    # Define thresholds
    upper_bound = df['close'] * (1 + TARGET_PERCENT_THRESHOLD)
    lower_bound = df['close'] * (1 - TARGET_PERCENT_THRESHOLD)
    
    # Assign target: 1 for up, 0 for down
    df['target'] = np.nan # Initialize with NaN
    df.loc[df['future_close'] >= upper_bound, 'target'] = 1
    df.loc[df['future_close'] <= lower_bound, 'target'] = 0
    
    # Drop rows where target is NaN (i.e., price stayed within the threshold)
    df = df.dropna(subset=['target'])
    df['target'] = df['target'].astype(int)
    # --- End Refined Target Variable Definition ---
    
    # Remove NaN values that might have been introduced by feature engineering later
    # (This is handled by ensemble_model.py's feature engineering)
    
    print(f"Dataset size after target definition: {len(df)} samples")
    
    # Initialize and train ensemble model
    ensemble_model = EnsembleForexModel()
    ensemble_model.train(df, target_col='target')
    
    # Save the trained model
    ensemble_model.save_model()
    
    print("🚀 Advanced Ensemble Training Complete!")
    print("Target: 70% accuracy for forex prediction")

if __name__ == "__main__":
    main()