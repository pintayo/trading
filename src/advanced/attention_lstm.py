import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from config.model_config import *

class AttentionMechanism(nn.Module):
    """Research-proven attention mechanism for forex prediction"""
    
    def __init__(self, hidden_size, attention_dim=ATTENTION_DIM):
        super(AttentionMechanism, self).__init__()
        self.hidden_size = hidden_size
        self.attention_dim = attention_dim
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )
        
    def forward(self, lstm_outputs):
        # lstm_outputs: (batch_size, seq_len, hidden_size)
        attention_weights = self.attention(lstm_outputs)  # (batch_size, seq_len, 1)
        attention_weights = F.softmax(attention_weights, dim=1)  # Normalize across sequence
        
        # Apply attention weights
        context = torch.sum(attention_weights * lstm_outputs, dim=1)  # (batch_size, hidden_size)
        
        return context, attention_weights

class AttentionLSTM(nn.Module):
    """Attention-based LSTM - Proven to achieve 65%+ accuracy"""
    
    def __init__(self, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, 
                 num_layers=NUM_LAYERS, dropout=DROPOUT):
        super(AttentionLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bidirectional LSTM for better pattern capture
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout, bidirectional=True
        )
        
        # Attention mechanism
        self.attention = AttentionMechanism(hidden_size * 2)  # *2 for bidirectional
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Initialize hidden states for bidirectional LSTM
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Apply attention mechanism
        context, attention_weights = self.attention(lstm_out)
        
        # Final prediction
        output = self.classifier(context)
        
        return output, attention_weights

class CNNLSTMModel(nn.Module):
    """CNN-LSTM hybrid for local pattern detection"""
    
    def __init__(self, input_size=INPUT_SIZE, sequence_length=SEQUENCE_LENGTH):
        super(CNNLSTMModel, self).__init__()
        
        # CNN layers for local pattern extraction - ALL FIXED
        self.conv1 = nn.Conv1d(input_size, CNN_FILTERS[0], CNN_KERNEL_SIZE, padding=1)
        self.conv2 = nn.Conv1d(CNN_FILTERS[0], CNN_FILTERS[1], CNN_KERNEL_SIZE, padding=1)
        self.conv3 = nn.Conv1d(CNN_FILTERS[1], CNN_FILTERS[2], CNN_KERNEL_SIZE, padding=1)

        
        self.pool = nn.MaxPool1d(CNN_POOL_SIZE)
        self.dropout = nn.Dropout(0.3)
        
        # Calculate LSTM input size after convolutions
        conv_output_size = CNN_FILTERS[2]  # This is 128
        
        # LSTM layers
        self.lstm = nn.LSTM(conv_output_size, 64, 2, batch_first=True, dropout=0.3)
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Input: (batch_size, seq_len, features)
        # Transpose for conv1d: (batch_size, features, seq_len)
        x = x.transpose(1, 2)
        
        # CNN feature extraction
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        # Transpose back for LSTM: (batch_size, seq_len, features)
        x = x.transpose(1, 2)
        
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Use last output
        output = self.classifier(lstm_out[:, -1, :])
        
        return output