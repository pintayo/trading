# Model Configuration - Organized Structure

# =============================================================================
# Paths and Global Settings (FIXED)
# =============================================================================
# These variables have been added to fix runtime errors.
# You may want to review or tune these values.

# For view_model.py
SIMPLE_MODEL_PATH = 'models/simple_usdjpy_model.pth'
ADVANCED_MODEL_PATH = 'models/ensemble_model.pkl'

# For advanced/attention_lstm.py
ATTENTION_DIM = 64
CNN_FILTERS = [32, 64, 128]
CNN_KERNEL_SIZE = 3
CNN_POOL_SIZE = 2

# For advanced/ensemble_model.py
XGBOOST_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'n_estimators': 200,
    'max_depth': 4,
    'learning_rate': 0.05,
    'use_label_encoder': False
}
ENSEMBLE_WEIGHTS = {
    'attention_lstm': 0.4,
    'xgboost': 0.35,
    'cnn_lstm': 0.25
}
ATTENTION_PATH = 'models/attention_lstm_model.pth'
CNN_PATH = 'models/cnn_lstm_model.pth'
XGBOOST_PATH = 'models/xgboost_model.pkl'
ENSEMBLE_PATH = 'models/ensemble_model.pkl'


# Technical Indicators (used by simple and advanced)
INDICATORS = {
    "rsi_period": 14,

    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "bb_period": 20,
    "bb_std": 2,
    "atr_period": 14
}

# =============================================================================
# SIMPLE MODEL CONFIGURATION (Recommended Start)
# =============================================================================

# Inside SIMPLE_CONFIG add:
SIMPLE_CONFIG = {
    'input_size': 22, # Updated to accommodate new features
    'hidden_size': 128, # Increased hidden size for more capacity
    'num_layers': 4, # Increased layers for more capacity
    'dropout': 0.5, # Increased dropout to combat overfitting
    'output_size': 1,
    'learning_rate': 0.0001, # Further reduced learning rate
    'batch_size': 32,
    'epochs': 300, # Increased epochs significantly
    'sequence_length': 48, # Increased sequence length
    'train_split': 0.8,
    'validation_split': 0.1,
    'test_split': 0.1,
    'model_path': 'models/simple_usdjpy_model.pth',
    'scaler_path': 'models/simple_scaler.pkl',
    'patience': 60, # Increased patience for early stopping
    'target_percent_threshold': 0.002, # Increased to 0.2% move for clearer signals
    'target_future_candles': 2 # Look 2 candles ahead (8 hours)
}

# Inside ADVANCED_CONFIG add:
ADVANCED_CONFIG = {
    'input_size': 20, # Corrected to match select_best_features k=20
    'hidden_size': 128,
    'num_layers': 3,
    'dropout': 0.3,
    'output_size': 1,
    'learning_rate': 0.0003,
    'batch_size': 128,
    'epochs': 300,
    'sequence_length': 48,
    'train_split': 0.7,
    'validation_split': 0.15,
    'test_split': 0.15,
    'ensemble_weights': {
        'attention_lstm': 0.4,
        'xgboost': 0.35,
        'cnn_lstm': 0.25
    },
    'model_path': 'models/ensemble_model.pkl',
    'scaler_path': 'models/ensemble_scaler.pkl'
}

CURRENT_MODE = "SIMPLE"  # Change to "ADVANCED" for advanced mode

# Then expose them in the CURRENT_MODE block:
if CURRENT_MODE == "SIMPLE":
    INPUT_SIZE        = SIMPLE_CONFIG['input_size']
    HIDDEN_SIZE       = SIMPLE_CONFIG['hidden_size']
    NUM_LAYERS        = SIMPLE_CONFIG['num_layers']
    DROPOUT           = SIMPLE_CONFIG['dropout']
    OUTPUT_SIZE       = SIMPLE_CONFIG['output_size']
    LEARNING_RATE     = SIMPLE_CONFIG['learning_rate']
    BATCH_SIZE        = SIMPLE_CONFIG['batch_size']
    EPOCHS            = SIMPLE_CONFIG['epochs']
    SEQUENCE_LENGTH   = SIMPLE_CONFIG['sequence_length']
    TRAIN_SPLIT       = SIMPLE_CONFIG['train_split']
    VALIDATION_SPLIT  = SIMPLE_CONFIG['validation_split']
    TEST_SPLIT        = SIMPLE_CONFIG['test_split']
    MODEL_PATH        = SIMPLE_CONFIG['model_path']
    SCALER_PATH       = SIMPLE_CONFIG['scaler_path']
    PATIENCE          = SIMPLE_CONFIG['patience'] # Exposed PATIENCE for early stopping
    TARGET_PERCENT_THRESHOLD = SIMPLE_CONFIG['target_percent_threshold']
    TARGET_FUTURE_CANDLES = SIMPLE_CONFIG['target_future_candles']
else:
    INPUT_SIZE        = ADVANCED_CONFIG['input_size']
    HIDDEN_SIZE       = ADVANCED_CONFIG['hidden_size']
    NUM_LAYERS        = ADVANCED_CONFIG['num_layers']
    DROPOUT           = ADVANCED_CONFIG['dropout']
    OUTPUT_SIZE       = ADVANCED_CONFIG['output_size']
    LEARNING_RATE     = ADVANCED_CONFIG['learning_rate']
    BATCH_SIZE        = ADVANCED_CONFIG['batch_size']
    EPOCHS            = ADVANCED_CONFIG['epochs']
    SEQUENCE_LENGTH   = ADVANCED_CONFIG['sequence_length']
    TRAIN_SPLIT       = ADVANCED_CONFIG['train_split']
    VALIDATION_SPLIT  = ADVANCED_CONFIG['validation_split']
    TEST_SPLIT        = ADVANCED_CONFIG['test_split']
    MODEL_PATH        = ADVANCED_CONFIG['model_path']
    SCALER_PATH       = ADVANCED_CONFIG['scaler_path']