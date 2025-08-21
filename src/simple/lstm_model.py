import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
from config.model_config import *

class USDJPYLSTMModel(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, 
                 num_layers=NUM_LAYERS, output_size=OUTPUT_SIZE):
        super(USDJPYLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # IMPROVED ARCHITECTURE
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=DROPOUT, bidirectional=True)
        
        # Bidirectional doubles the output size
        lstm_output_size = hidden_size * 2
        
        # Multi-layer classifier
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(), 
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_size // 2, output_size),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Initialize hidden states for bidirectional LSTM
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size)
        
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Use the last output
        output = self.classifier(lstm_out[:, -1, :])
        return output

class ModelTrainer:
    def __init__(self, model=None, learning_rate=LEARNING_RATE):
        self.model = model or USDJPYLSTMModel()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.BCELoss()
        self.scaler = MinMaxScaler()
        
    def prepare_sequences(self, features, targets, sequence_length=SEQUENCE_LENGTH):
        X, y = [], []
        for i in range(len(features) - sequence_length):
            X.append(features[i:(i + sequence_length)])
            y.append(targets[i + sequence_length])
        return np.array(X), np.array(y)
    
    def split_data(self, X, y):
        """Split data into train/validation/test sets"""
        n = len(X)
        train_end = int(n * TRAIN_SPLIT)
        val_end = int(n * (TRAIN_SPLIT + VALIDATION_SPLIT))
        
        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=EPOCHS):
        self.model.train()
        best_val_loss = float('inf')
        patience_counter = 0 # Initialize patience counter
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            outputs = self.model(X_train)
            loss = self.criterion(outputs.squeeze(), y_train.float())
            loss.backward()
            self.optimizer.step()
            
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val)
                    val_loss = self.criterion(val_outputs.squeeze(), y_val.float())
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save(self.model.state_dict(), MODEL_PATH)
                        patience_counter = 0 # Reset patience if validation loss improves
                    else:
                        patience_counter += 1 # Increment patience if no improvement
                        
                    if patience_counter >= PATIENCE: # Check for early stopping
                        print(f"Early stopping at epoch {epoch} due to no improvement in validation loss.")
                        break
                        
                self.model.train()
                if epoch % 10 == 0: # Print every 10 epochs
                    print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
            elif epoch % 20 == 0:
                print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}')
    
    def save_scaler(self):
        with open(SCALER_PATH, 'wb') as f:
            pickle.dump(self.scaler, f)
    
    def load_scaler(self):
        with open(SCALER_PATH, 'rb') as f:
            self.scaler = pickle.load(f)