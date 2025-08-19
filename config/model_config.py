# LSTM Model Configuration

# Model Architecture
# because Your features are:
# open, high, low, close, volume (5 features)
# rsi, macd, macd_signal, bb_upper, bb_middle, bb_lower (6 features)
# atr, bb_position, price_change, volatility (4 features)
INPUT_SIZE = 15
HIDDEN_SIZE = 50
NUM_LAYERS = 2
OUTPUT_SIZE = 1
DROPOUT = 0.2

# Training Parameters
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 80
TRAIN_SPLIT = 0.8
VALIDATION_SPLIT = 0.1
TEST_SPLIT = 0.1

# Technical Indicators
INDICATORS = {
    "rsi_period": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "bb_period": 20,
    "bb_std": 2,
    "atr_period": 14
}

# Model Paths
MODEL_PATH = "models/usdjpy_model.pth"
SCALER_PATH = "models/scaler.pkl"
