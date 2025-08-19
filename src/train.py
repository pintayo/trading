import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processor import DataProcessor
from lstm_model import USDJPYLSTMModel, ModelTrainer
import torch
import numpy as np
from config.model_config import MODEL_PATH

def main():
    print("Starting USD/JPY LSTM Model Training...")
    
    # Load and process data
    processor = DataProcessor()
    features, targets = processor.prepare_training_data()
    
    # Initialize trainer
    trainer = ModelTrainer(USDJPYLSTMModel())
    
    # Normalize features
    features_scaled = trainer.scaler.fit_transform(features.values)
    
    # Create sequences
    X, y = trainer.prepare_sequences(features_scaled, targets.values)
    
    # Split data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = trainer.split_data(X, y)
    
    # Convert to tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.FloatTensor(y_val)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)
    
    print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")
    
    # Train model
    trainer.train(X_train, y_train, X_val, y_val)
    
    # Save model and scaler
    torch.save(trainer.model.state_dict(), MODEL_PATH)
    trainer.save_scaler()
    
    # Evaluate on test set
    trainer.model.eval()
    with torch.no_grad():
        test_outputs = trainer.model(X_test)
        predictions = (test_outputs.squeeze() > 0.5).float()
        accuracy = (predictions == y_test).float().mean()
        
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Model saved to {MODEL_PATH}")
    print("Training complete!")

if __name__ == "__main__":
    main()