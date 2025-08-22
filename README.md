# Pin Trading System

## Overview
The Pin Trading System is a Python-based algorithmic trading framework designed for forex markets. It features both a "simple" LSTM model and an "advanced" ensemble model, integrated with Interactive Brokers (IBKR) for paper trading and equipped with a backtesting framework. The system is built with reusability and modularity in mind, allowing for dynamic selection of model types, forex pairs, and actions via a command-line interface.

## Features
- **Dynamic Data Collection:** Fetches historical forex data using Yahoo Finance for various currency pairs.
- **Comprehensive Feature Engineering:** Includes a wide range of technical indicators, lagged features, and volatility measures for both simple and advanced models.
- **LSTM & Ensemble Models:**
    - **Simple Model:** A basic LSTM neural network for price prediction.
    - **Advanced Model:** An ensemble combining Attention LSTM, CNN-LSTM, and XGBoost for potentially higher accuracy.
- **Refined Target Variable:** Predicts significant percentage price moves over a future window to reduce noise.
- **Robust Backtesting Framework:** Simulates trading strategies on historical data, providing detailed performance metrics (PnL, win rate, drawdown).
- **IBKR Paper Trading Integration:** Connects to IBKR TWS/Gateway for live data fetching and simulated trade execution.
- **Menu-Driven CLI (`main.py`):** A central interface for selecting model types, forex pairs, and actions (data collection, training, backtesting, live trading).
- **Dynamic Model/Scaler Management:** Automatically saves and loads models and scalers based on selected model type and currency pair.
- **Risk Management:** Basic position sizing, stop-loss, and take-profit mechanisms.

## Project Structure
```
.
├── config/                 # Configuration files (model parameters, trading rules)
├── data/                   # Stores historical data (e.g., usdjpy_data.db)
├── logs/                   # Stores trading logs
├── models/                 # Stores trained model (.pth) and scaler (.pkl) files
├── src/
│   ├── advanced/           # Components for the advanced ensemble model
│   └── simple/             # Components for the simple LSTM model and core functionalities
├── main.py                 # Main menu-driven CLI script
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Prerequisites
- Python 3.8+
- Interactive Brokers Trader Workstation (TWS) or IB Gateway running for live/paper trading.
- **Important:** Ensure your IBKR paper trading account has the necessary market data subscriptions for the forex pairs you want to trade.

## Installation
1.  Navigate to the project root directory in your terminal.
2.  Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
To run the trading system's command-line interface:
```bash
python3 main.py
```

Follow the on-screen prompts to select:
1.  **Model Type:** Simple or Advanced.
2.  **Forex Pair:** Choose from predefined pairs (USD/JPY, AUD/NZD, EUR/CHF).
3.  **Action:**
    *   `1: Get Historical Data`: Downloads historical data for the selected pair.
    *   `2: Train Model`: Trains the selected model type on the data for the chosen pair.
    *   `3: Run Backtest`: Runs a backtest of the selected model on historical data.
    *   `4: Connect to IBKR (Paper Trading)`: Attempts to connect to IBKR and run the trading system in paper mode.

## Future Improvements
This project is a robust foundation, but here are key areas for future development and improvement:

1.  **Advanced Model Integration (Live Trading):**
    *   Currently, the live trading system (`src/simple/trading_system.py`) only uses the simple model. It needs to be updated to dynamically use the advanced model when selected.
2.  **Backtesting Enhancements:**
    *   **Slippage & Commissions:** Implement realistic slippage and commission costs in the backtester for more accurate performance evaluation.
    *   **Multiple Open Positions:** Enhance the backtester to manage multiple concurrent open trades.
    *   **More Complex Exit Strategies:** Implement time-based exits, profit-target re-evaluation, or trailing stops.
3.  **Continuous Learning Implementation:**
    *   Fully integrate and test the `src/simple/continuous_learning.py` module to enable the system to adapt and improve over time based on live trading performance.
4.  **Error Handling & Logging:**
    *   Implement more comprehensive logging throughout the system for easier debugging and monitoring of live operations.
    *   Add more specific error handling for various API responses and data issues.
5.  **Deployment & Monitoring:**
    *   Develop a strategy for running the system 24/7 (e.g., on a dedicated server, cloud instance).
    *   Implement monitoring tools to track performance, system health, and alerts.
6.  **Strategy Optimization:**
    *   **Hyperparameter Tuning:** Conduct more systematic hyperparameter searches for both simple and advanced models.
    *   **Feature Engineering:** Explore more advanced feature sets or alternative data sources.
    *   **Strategy Diversification:** Investigate other trading strategies (e.g., mean reversion, arbitrage) and asset classes.
7.  **Unit Tests:**
    *   Add unit tests for critical functions and classes to ensure code correctness and prevent regressions.
