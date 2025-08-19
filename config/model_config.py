# Model Configuration - Organized Structure

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
    'input_size': 16,
    'hidden_size': 50,
    'num_layers': 2,
    'dropout': 0.2,
    'output_size': 1,
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 80,
    'sequence_length': 24,
    'train_split': 0.8,
    'validation_split': 0.1,
    'test_split': 0.1,
    'model_path': 'models/simple_usdjpy_model.pth',
    'scaler_path': 'models/simple_scaler.pkl'
}

# Inside ADVANCED_CONFIG add:
ADVANCED_CONFIG = {
    'input_size': 25,
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