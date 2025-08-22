import os
# Model Configuration - Organized Structure

# =============================================================================
# Paths and Global Settings
# =============================================================================

# For view_model.py
SIMPLE_MODEL_PATH = 'models/simple_usdjpy_model.pth'
ADVANCED_MODEL_PATH = 'models/ensemble_model.pkl'

# =============================================================================
# RESEARCH-BACKED HYPERPARAMETERS
# =============================================================================

# XGBoost Optimization (Based on Latest Research)
XGBOOST_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',                # Better than logloss for imbalanced data
    'n_estimators': 1000,                # More trees with early stopping
    'max_depth': 8,                      # Deeper trees for complex patterns
    'learning_rate': 0.01,               # Lower LR for better generalization
    'subsample': 0.85,                   # Prevent overfitting
    'colsample_bytree': 0.8,            # Feature sampling
    'colsample_bylevel': 0.8,           # Additional regularization
    'min_child_weight': 3,               # Minimum samples per leaf
    'gamma': 0.1,                        # Minimum split loss
    'reg_alpha': 0.1,                    # L1 regularization
    'reg_lambda': 1.0,                   # L2 regularization
    'random_state': 42,
    'n_jobs': -1,                        # Use all CPU cores
    'early_stopping_rounds': 50,         # Stop if no improvement
    'use_label_encoder': False
}

# CNN Architecture (Optimized for Forex Patterns)
CNN_FILTERS = [64, 128, 256, 512]       # Progressive filter increase
CNN_KERNEL_SIZES = [3, 5, 7]           # Multi-scale convolutions
CNN_POOL_SIZE = 2
CNN_DROPOUT = 0.3

# Attention Mechanism (Research-Proven Settings)
ATTENTION_DIM = 128                      # Sufficient for complex attention
ATTENTION_HEADS = 8                      # Multi-head attention
ATTENTION_DROPOUT = 0.2

# Ensemble Optimization (Dynamic Weighting)
ENSEMBLE_WEIGHTS = {
    'attention_lstm': 0.45,              # Best for trend detection
    'xgboost': 0.35,                     # Best for feature interactions
    'cnn_lstm': 0.20                     # Best for local patterns
}


# Technical Indicators (used by simple and advanced)
INDICATORS = {
    "rsi_period": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "bb_period": 20,
    "bb_std": 2,
    "atr_period": 14,
    "stoch_period": 14,
    "adx_period": 14
}

# =============================================================================
# SIMPLE MODEL CONFIGURATION (Recommended Start)
# =============================================================================

SIMPLE_CONFIG = {
    'input_size': 27, # Updated to accommodate new features
    'hidden_size': 128, # Increased hidden size for more capacity
    'num_layers': 4, # Increased layers for more capacity
    'dropout': 0.5, # Increased dropout to combat overfitting
    'output_size': 1,
    'learning_rate': 0.00005, # Further reduced learning rate
    'batch_size': 32,
    'epochs': 1000, # Increased epochs significantly
    'sequence_length': 48, # Increased sequence length
    'train_split': 0.8,
    'validation_split': 0.1,
    'test_split': 0.1,
    'patience': 60, # Increased patience for early stopping
    'target_percent_threshold': 0.005, # Increased to 0.5% move for clearer signals
    'target_future_candles': 2, # Look 2 candles ahead (8 hours)
    'feature_columns': [
        'open', 'high', 'low', 'close', 'volume',
        'rsi', 'macd', 'macd_signal', 
        'bb_upper', 'bb_middle', 'bb_lower',
        'atr', 'bb_position',
        'price_change', 'volatility', 'rsi_momentum',
        'close_lag_1', 'close_lag_2', 'close_lag_3',
        'obv',
        'rolling_std_7', 'rolling_std_21',
        'stoch_k', 'stoch_d',
        'adx', 'plus_di', 'minus_di'
    ]
}

# =============================================================================
# ADVANCED MODEL CONFIGURATION (Research-Optimized for 65%+ Accuracy)
# =============================================================================

ADVANCED_CONFIG = {
    # Core Architecture (Research-Backed Optimization)
    'hidden_size': 256,                  # Increased capacity for complex patterns
    'num_layers': 4,                     # Deeper architecture for hierarchical learning
    'dropout': 0.4,                      # Higher dropout to prevent overfitting
    'output_size': 1,

    # Training Parameters (Proven Optimal)
    'learning_rate': 0.00005,            # Very low LR for stable convergence
    'batch_size': 64,                    # Balanced for GPU memory and stability
    'epochs': 500,                       # Extended training with early stopping
    'sequence_length': 72,               # 12 days of 4-hour data (proven optimal)
    'patience': 50,                      # Early stopping patience

    # Data Splits (Research-Optimized)
    'train_split': 0.65,                 # More validation data for better generalization
    'validation_split': 0.20,           # Increased validation for robust evaluation
    'test_split': 0.15,                  # Final test set

    # Advanced Features
    'gradient_clipping': 1.0,            # Prevent gradient explosion
    'weight_decay': 1e-5,                # L2 regularization

    # Ensemble Configuration (Research-Optimized Weights)
    'ensemble_weights': {
        'attention_lstm': 0.45,          # Increased weight for attention model
        'xgboost': 0.35,                 # Best for feature interactions
        'cnn_lstm': 0.20                 # CNN for local pattern detection
    },

    # Advanced Target Engineering
    'target_percent_threshold': 0.002,   # 0.2% move (optimal for 4H forex)
    'target_future_candles': 3,          # 12-hour prediction horizon
}


# Function to get dynamic model and scaler paths
def get_model_paths(model_type, pair_name):
    base_path = "models/"
    model_filename = f"{model_type}_{pair_name.replace('/', '')}_model.pth"
    scaler_filename = f"{model_type}_{pair_name.replace('/', '')}_scaler.pkl"
    
    # Special handling for ensemble model's internal paths
    if model_type == "advanced":
        attention_path = f"{base_path}attention_lstm_{pair_name.replace('/', '')}_model.pth"
        cnn_path = f"{base_path}cnn_lstm_{pair_name.replace('/', '')}_model.pth"
        xgboost_path = f"{base_path}xgboost_{pair_name.replace('/', '')}_model.pkl"
        return {
            "model_path": os.path.join(base_path, model_filename),
            "scaler_path": os.path.join(base_path, scaler_filename),
            "attention_path": attention_path,
            "cnn_path": cnn_path,
            "xgboost_path": xgboost_path
        }
    else:
        return {
            "model_path": os.path.join(base_path, model_filename),
            "scaler_path": os.path.join(base_path, scaler_filename)
        }