import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data_processor import DataProcessor
from ensemble_model import EnsembleForexModel

def main():
    print("Starting Advanced Ensemble Training for 70% Accuracy Target...")
    
    # Load and process data
    processor = DataProcessor()
    df = processor.load_data()
    
    # Create target variable (1 if price goes up in next 4 hours, 0 if down)
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    # Remove NaN values
    df = df.dropna()
    
    print(f"Dataset size: {len(df)} samples")
    
    # Initialize and train ensemble model
    ensemble_model = EnsembleForexModel()
    ensemble_model.train(df, target_col='target')
    
    # Save the trained model
    ensemble_model.save_model()
    
    print("🚀 Advanced Ensemble Training Complete!")
    print("Target: 70% accuracy for forex prediction")

if __name__ == "__main__":
    main()