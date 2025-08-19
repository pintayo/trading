import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class USDJPYLSTMModel(nn.Module):
    def __init__(self, input_size=12, hidden_size=50, num_layers=2, output_size=1):
        super(USDJPYLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        lstm_out, _ = self.lstm(x, (h0, c0))
        output = self.fc(lstm_out[:, -1, :])
        return self.sigmoid(output)

class ModelTrainer:
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.BCELoss()
        self.scaler = MinMaxScaler()
        
    def prepare_sequences(self, features, targets, sequence_length=24):  # 24 = 4 days of 4-hour data
        X, y = [], []
        for i in range(len(features) - sequence_length):
            X.append(features[i:(i + sequence_length)])
            y.append(targets[i + sequence_length])
        return np.array(X), np.array(y)
    
    def train(self, X_train, y_train, epochs=100):
        self.model.train()
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            outputs = self.model(X_train)
            loss = self.criterion(outputs.squeeze(), y_train.float())
            loss.backward()
            self.optimizer.step()
            
            if epoch % 20 == 0:
                print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}')
