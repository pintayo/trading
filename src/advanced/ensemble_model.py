import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

from .attention_lstm import AttentionLSTM, CNNLSTMModel
from .advanced_features import AdvancedFeatureEngineering
from config.model_config import *

class EnsembleForexModel:
    """Research-backed ensemble model targeting 70% accuracy"""
    
    def __init__(self):
        self.attention_lstm = None
        self.cnn_lstm = None
        self.xgboost = xgb.XGBClassifier(**XGBOOST_PARAMS)
        
        self.feature_engineer = AdvancedFeatureEngineering()
        self.selected_features = None
        self.is_trained = False
        
    def prepare_lstm_data(self, features, sequence_length):
        """Prepare sequential data for LSTM models"""
        X, y = [], []
        for i in range(len(features) - sequence_length):
            X.append(features[i:(i + sequence_length)])
        return np.array(X)
    
    def prepare_xgboost_data(self, features):
        """Prepare flat data for XGBoost"""
        # Use last 12 periods as features for XGBoost (flattened)
        X = []
        for i in range(12, len(features)):
            # Flatten last 12 periods
            flat_features = features[i-12:i].flatten()
            X.append(flat_features)
        return np.array(X)
    
    def train(self, df, target_col='target', paths=None):
        """Train the complete ensemble model"""
        print("Starting ensemble model training...")
        
        # Feature engineering
        df_features = self.feature_engineer.create_advanced_features(df.copy())
        
        # Feature selection
        features_df, self.selected_features = self.feature_engineer.select_best_features(
            df_features, df_features[target_col]
        )
        
        # Get number of features
        num_features = len(self.selected_features)
        
        # Initialize models with dynamic input size
        self.attention_lstm = AttentionLSTM(input_size=num_features)
        self.cnn_lstm = CNNLSTMModel(input_size=num_features)
        
        # Prepare data
        features = features_df.values
        targets = df_features[target_col].values
        
        # Split data
        train_size = int(len(features) * ADVANCED_CONFIG['train_split'])
        val_size = int(len(features) * ADVANCED_CONFIG['validation_split'])
        
        # LSTM data preparation
        lstm_X = self.prepare_lstm_data(features, sequence_length=ADVANCED_CONFIG['sequence_length'])
        lstm_y = targets[ADVANCED_CONFIG['sequence_length']:]
        
        # XGBoost data preparation  
        xgb_X = self.prepare_xgboost_data(features)
        xgb_y = targets[12:]  # Align with XGBoost preparation
        
        # Align all datasets to same length
        min_length = min(len(lstm_X), len(xgb_X))
        lstm_X = lstm_X[:min_length]
        lstm_y = lstm_y[:min_length]
        xgb_X = xgb_X[:min_length]
        xgb_y = xgb_y[:min_length]
        
        # Train splits
        train_end = int(min_length * ADVANCED_CONFIG['train_split'])
        val_end = int(min_length * (ADVANCED_CONFIG['train_split'] + ADVANCED_CONFIG['validation_split']))
        
        # Train Attention LSTM
        print("Training Attention LSTM...")
        self._train_lstm_model(
            self.attention_lstm, 
            lstm_X[:train_end], lstm_y[:train_end],
            lstm_X[train_end:val_end], lstm_y[train_end:val_end],
            paths['attention_path']
        )
        
        # Train CNN-LSTM
        print("Training CNN-LSTM...")
        self._train_lstm_model(
            self.cnn_lstm,
            lstm_X[:train_end], lstm_y[:train_end],
            lstm_X[train_end:val_end], lstm_y[train_end:val_end], 
            paths['cnn_path']
        )
        
        # Train XGBoost
        print("Training XGBoost...")
        self.xgboost.fit(xgb_X[:train_end], xgb_y[:train_end], 
                         eval_set=[(xgb_X[train_end:val_end], xgb_y[train_end:val_end])], 
                         verbose=False)
        
        # Save XGBoost
        with open(paths['xgboost_path'], 'wb') as f:
            pickle.dump(self.xgboost, f)
        
        # Evaluate ensemble
        self._evaluate_ensemble(lstm_X[val_end:], xgb_X[val_end:], lstm_y[val_end:])
        
        self.is_trained = True
        print("Ensemble training complete!")
        
    def _train_lstm_model(self, model, X_train, y_train, X_val, y_val, save_path):
        """Train individual LSTM model"""
        num_features = len(self.selected_features)
        # Convert to tensors
        X_train = torch.FloatTensor(X_train).reshape(-1, ADVANCED_CONFIG['sequence_length'], num_features)
        y_train = torch.FloatTensor(y_train)
        X_val = torch.FloatTensor(X_val).reshape(-1, ADVANCED_CONFIG['sequence_length'], num_features)
        y_val = torch.FloatTensor(y_val)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=ADVANCED_CONFIG['learning_rate'], weight_decay=ADVANCED_CONFIG['weight_decay'])
        criterion = nn.BCELoss()
        
        best_val_loss = float('inf')
        patience = ADVANCED_CONFIG['patience']
        patience_counter = 0
        
        model.train()
        for epoch in range(ADVANCED_CONFIG['epochs']):
            optimizer.zero_grad()
            
            if isinstance(model, AttentionLSTM):
                outputs, _ = model(X_train)
            else:
                outputs = model(X_train)
                
            loss = criterion(outputs.squeeze(), y_train)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=ADVANCED_CONFIG['gradient_clipping']) # Gradient clipping
            optimizer.step()
            
            # Validation
            if epoch % 20 == 0:
                model.eval()
                with torch.no_grad():
                    if isinstance(model, AttentionLSTM):
                        val_outputs, _ = model(X_val)
                    else:
                        val_outputs = model(X_val)
                    val_loss = criterion(val_outputs.squeeze(), y_val)
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save(model.state_dict(), save_path)
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch}")
                        break
                        
                print(f'Epoch [{epoch}/{ADVANCED_CONFIG["epochs"]}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
                model.train()
    
    def _evaluate_ensemble(self, lstm_X, xgb_X, y_true):
        """Evaluate ensemble performance"""
        num_features = len(self.selected_features)
        # Ensure lstm_X is 3D for prediction
        if lstm_X.ndim == 2:
            lstm_X = lstm_X.reshape(-1, ADVANCED_CONFIG['sequence_length'], num_features)

        # Get predictions from all models
        predictions = self.predict(lstm_X, xgb_X, return_individual=True)
        
        ensemble_pred = predictions['ensemble']
        
        accuracy = accuracy_score(y_true, (ensemble_pred > 0.5).astype(int))
        print(f"\nEnsemble Test Accuracy: {accuracy:.4f}")
        print(f"Target reached: {'✅' if accuracy >= 0.70 else '❌'} (Target: 70%)")
        
        print("\nIndividual Model Accuracies:")
        for model_name, pred in predictions.items():
            if model_name != 'ensemble':
                acc = accuracy_score(y_true, (pred > 0.5).astype(int))
                print(f"{model_name}: {acc:.4f}")
    
    def predict(self, lstm_X, xgb_X, return_individual=False):
        """Generate ensemble predictions"""
        predictions = {}
        
        # Attention LSTM predictions
        self.attention_lstm.eval()
        with torch.no_grad():
            lstm_X_tensor = torch.FloatTensor(lstm_X)
            attn_pred, _ = self.attention_lstm(lstm_X_tensor)
            predictions['attention_lstm'] = attn_pred.squeeze().numpy()
        
        # CNN-LSTM predictions
        self.cnn_lstm.eval()
        with torch.no_grad():
            cnn_pred = self.cnn_lstm(lstm_X_tensor)
            predictions['cnn_lstm'] = cnn_pred.squeeze().numpy()
        
        # XGBoost predictions
        xgb_pred = self.xgboost.predict_proba(xgb_X)[:, 1]
        predictions['xgboost'] = xgb_pred
        
        # Ensemble prediction (weighted average)
        ensemble_pred = (
            predictions['attention_lstm'] * ENSEMBLE_WEIGHTS['attention_lstm'] +
            predictions['cnn_lstm'] * ENSEMBLE_WEIGHTS['cnn_lstm'] + 
            predictions['xgboost'] * ENSEMBLE_WEIGHTS['xgboost']
        )
        
        predictions['ensemble'] = ensemble_pred
        
        if return_individual:
            return predictions
        else:
            return ensemble_pred
    
    def save_model(self, paths):
        """Save complete ensemble model"""
        model_data = {
            'selected_features': self.selected_features,
            'feature_engineer': self.feature_engineer,
            'is_trained': self.is_trained
        }
        
        with open(paths['model_path'], 'wb') as f:
            pickle.dump(model_data, f)
        
        print("Ensemble model saved successfully!")
    
    def load_model(self, paths):
        """Load complete ensemble model"""
        # Load model states
        with open(paths['model_path'], 'rb') as f:
            model_data = pickle.load(f)
            self.selected_features = model_data['selected_features']
            self.feature_engineer = model_data['feature_engineer']
            self.is_trained = model_data['is_trained']
        
        num_features = len(self.selected_features)
        self.attention_lstm = AttentionLSTM(input_size=num_features)
        self.cnn_lstm = CNNLSTMModel(input_size=num_features)

        self.attention_lstm.load_state_dict(torch.load(paths['attention_path']))
        self.cnn_lstm.load_state_dict(torch.load(paths['cnn_path']))
        
        with open(paths['xgboost_path'], 'rb') as f:
            self.xgboost = pickle.load(f)
            
        
        print("Ensemble model loaded successfully!")