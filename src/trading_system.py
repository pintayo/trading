import torch
import pandas as pd
from lstm_model import USDJPYLSTMModel, ModelTrainer
from data_processor import DataProcessor

class TradingSystem:
    def __init__(self):
        self.model = USDJPYLSTMModel()
        self.trainer = ModelTrainer(self.model)
        self.processor = DataProcessor()
        
    def train_model(self):
        features, targets = self.processor.prepare_training_data()
        
        # Normalize features
        features_scaled = self.trainer.scaler.fit_transform(features)
        
        # Prepare sequences
        X, y = self.trainer.prepare_sequences(features_scaled, targets.values)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        # Train model
        self.trainer.train(X_tensor, y_tensor)
        
        # Save model
        torch.save(self.model.state_dict(), 'models/usdjpy_model.pth')
        print("Model trained and saved!")
        
    def backtest(self, start_date, end_date):
        # Simple backtesting logic
        features, targets = self.processor.prepare_training_data()
        # Implementation for backtesting performance
        pass

# Usage script
if __name__ == "__main__":
    system = TradingSystem()
    system.train_model()
