import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from .data_processor import DataProcessor
from .lstm_model import ForexLSTMModel, ModelTrainer
import torch
import numpy as np
from config.model_config import SIMPLE_CONFIG, get_model_paths # Import SIMPLE_CONFIG and get_model_paths for standalone run

def train_simple_model(config_params, pair_config):
    print(f"Starting Simple LSTM Model Training for {pair_config['name']}...")
    
    # Get dynamic model and scaler paths
    paths = get_model_paths(model_type="simple", pair_name=pair_config['name'])
    model_path = paths['model_path']
    scaler_path = paths['scaler_path']

    # Load and process data
    processor = DataProcessor()
    features, targets = processor.prepare_training_data()
    
    # Initialize trainer
    model = ForexLSTMModel(
        input_size=config_params['input_size'],
        hidden_size=config_params['hidden_size'],
        num_layers=config_params['num_layers'],
        output_size=config_params['output_size'],
        dropout=config_params['dropout']
    )
    trainer = ModelTrainer(model, learning_rate=config_params['learning_rate'])
    
    # Normalize features
    features_scaled = trainer.scaler.fit_transform(features.values)
    
    # Create sequences
    X, y = trainer.prepare_sequences(features_scaled, targets.values, sequence_length=config_params['sequence_length'])
    
    # Split data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = trainer.split_data(
        X, y, 
        train_split=config_params['train_split'],
        validation_split=config_params['validation_split'],
        test_split=config_params['test_split']
    )
    
    # Convert to tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.FloatTensor(y_val)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)
    
    print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")
    
    # Train model
    trainer.train(
        X_train, y_train, X_val, y_val,
        epochs=config_params['epochs'],
        model_path=model_path,
        patience=config_params['patience']
    )
    
    # Save model and scaler
    torch.save(trainer.model.state_dict(), model_path)
    trainer.save_scaler(scaler_path)
    
    # Evaluate on test set
    trainer.model.eval()
    with torch.no_grad():
        test_outputs = trainer.model(X_test)
        predictions = (test_outputs.squeeze() > 0.5).float()
        accuracy = (predictions == y_test).float().mean()
        
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Model saved to {model_path}")
    print("Training complete!")

if __name__ == "__main__":
    from config.trading_config import SYMBOL, IBKR_SYMBOL, IBKR_CURRENCY, IBKR_EXCHANGE
    test_pair_config = {
        "name": "USD/JPY",
        "symbol": SYMBOL,
        "ibkr_symbol": IBKR_SYMBOL,
        "ibkr_currency": IBKR_CURRENCY,
        "ibkr_exchange": IBKR_EXCHANGE
    }
    train_simple_model(SIMPLE_CONFIG, test_pair_config)