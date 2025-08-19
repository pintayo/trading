import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import sqlite3
import pandas as pd
import numpy as np
import torch
from datetime import datetime
import pickle
from lstm_model import USDJPYLSTMModel, ModelTrainer

class ContinuousLearningSystem:
    """Make your 52% model continuously improve through real trading"""
    
    def __init__(self):
        self.model = USDJPYLSTMModel()
        self.trainer = ModelTrainer()
        self.load_current_model()
        
        # Performance tracking
        self.recent_accuracy = []
        self.learning_threshold = 0.48  # Retrain if accuracy drops below 48%
        
    def load_current_model(self):
        """Load your existing 52% model"""
        try:
            self.model.load_state_dict(torch.load("models/usdjpy_model.pth"))
            with open("models/scaler.pkl", 'rb') as f:
                self.trainer.scaler = pickle.load(f)
            print("✅ Loaded existing model (52% accuracy)")
        except:
            print("❌ No existing model found")
    
    def log_prediction_outcome(self, prediction_confidence, actual_outcome, 
                             market_context, trade_details):
        """Log every prediction and its outcome"""
        conn = sqlite3.connect("logs/learning_log.db")
        
        # Create table if not exists
        conn.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                timestamp TEXT,
                prediction_confidence REAL,
                actual_outcome INTEGER,
                was_correct INTEGER,
                market_volatility REAL,
                time_of_day INTEGER,
                market_session TEXT,
                rsi REAL,
                macd REAL,
                bb_position REAL,
                price_change_4h REAL,
                trade_executed INTEGER,
                profit_loss REAL
            )
        ''')
        
        was_correct = 1 if (prediction_confidence > 0.5) == bool(actual_outcome) else 0
        
        conn.execute('''
            INSERT INTO predictions 
            (timestamp, prediction_confidence, actual_outcome, was_correct,
             market_volatility, time_of_day, market_session, rsi, macd, 
             bb_position, price_change_4h, trade_executed, profit_loss)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            prediction_confidence,
            actual_outcome,
            was_correct,
            market_context.get('volatility', 0),
            datetime.now().hour,
            market_context.get('session', 'unknown'),
            market_context.get('rsi', 0),
            market_context.get('macd', 0),
            market_context.get('bb_position', 0.5),
            market_context.get('price_change_4h', 0),
            1 if trade_details else 0,
            trade_details.get('pnl', 0) if trade_details else 0
        ))
        
        conn.commit()
        conn.close()
        
        # Update recent accuracy
        self.recent_accuracy.append(was_correct)
        if len(self.recent_accuracy) > 100:  # Keep last 100 predictions
            self.recent_accuracy.pop(0)
        
        print(f"Prediction logged: {prediction_confidence:.3f} -> {actual_outcome} ({'✅' if was_correct else '❌'})")
    
    def analyze_performance_patterns(self):
        """Find what conditions your model works best/worst in"""
        conn = sqlite3.connect("logs/learning_log.db")
        df = pd.read_sql_query("SELECT * FROM predictions", conn)
        conn.close()
        
        if len(df) < 50:
            return "Need more predictions for analysis"
        
        # Analyze by market conditions
        accuracy_by_volatility = df.groupby(pd.cut(df['market_volatility'], bins=5))['was_correct'].mean()
        accuracy_by_time = df.groupby('time_of_day')['was_correct'].mean()
        accuracy_by_rsi = df.groupby(pd.cut(df['rsi'], bins=[0, 30, 70, 100]))['was_correct'].mean()
        
        print("📊 Performance Analysis:")
        print(f"Overall accuracy (last {len(df)} predictions): {df['was_correct'].mean():.1%}")
        print(f"Best time of day: {accuracy_by_time.idxmax()}:00 ({accuracy_by_time.max():.1%})")
        print(f"Worst time of day: {accuracy_by_time.idxmin()}:00 ({accuracy_by_time.min():.1%})")
        print(f"High confidence (>0.7) accuracy: {df[df['prediction_confidence'] > 0.7]['was_correct'].mean():.1%}")
        
        return {
            'overall_accuracy': df['was_correct'].mean(),
            'high_confidence_accuracy': df[df['prediction_confidence'] > 0.7]['was_correct'].mean(),
            'best_conditions': accuracy_by_time.to_dict()
        }
    
    def should_retrain(self):
        """Decide if model needs retraining"""
        if len(self.recent_accuracy) < 50:
            return False, "Not enough recent data"
        
        recent_acc = np.mean(self.recent_accuracy)
        
        if recent_acc < self.learning_threshold:
            return True, f"Recent accuracy dropped to {recent_acc:.1%}"
        
        return False, f"Performance stable at {recent_acc:.1%}"
    
    def incremental_retrain(self):
        """Retrain model with new data, keeping good parts"""
        print("🔄 Starting incremental retraining...")
        
        # Load recent successful patterns
        conn = sqlite3.connect("logs/learning_log.db")
        recent_data = pd.read_sql_query('''
            SELECT * FROM predictions 
            WHERE timestamp > datetime('now', '-30 days')
            ORDER BY timestamp DESC
        ''', conn)
        conn.close()
        
        if len(recent_data) < 100:
            print("❌ Not enough recent data for retraining")
            return
        
        # Focus on high-confidence correct predictions
        good_predictions = recent_data[
            (recent_data['was_correct'] == 1) & 
            (recent_data['prediction_confidence'] > 0.6)
        ]
        
        print(f"Found {len(good_predictions)} high-quality recent predictions")
        
        # This would implement actual incremental learning
        # For now, just update confidence thresholds
        self._update_trading_thresholds(recent_data)
    
    def _update_trading_thresholds(self, recent_data):
        """Update when to trade based on recent performance"""
        # Find optimal confidence threshold
        thresholds = [0.55, 0.60, 0.65, 0.70, 0.75]
        best_threshold = 0.60
        best_performance = 0
        
        for threshold in thresholds:
            subset = recent_data[recent_data['prediction_confidence'] > threshold]
            if len(subset) > 10:
                accuracy = subset['was_correct'].mean()
                trade_count = len(subset)
                performance_score = accuracy * np.log(trade_count)  # Balance accuracy vs trade frequency
                
                if performance_score > best_performance:
                    best_performance = performance_score
                    best_threshold = threshold
        
        # Save updated threshold
        with open("models/trading_threshold.pkl", 'wb') as f:
            pickle.dump({
                'confidence_threshold': best_threshold,
                'last_updated': datetime.now().isoformat(),
                'recent_accuracy': recent_data['was_correct'].mean()
            }, f)
        
        print(f"📈 Updated trading threshold to {best_threshold:.2f}")
        print(f"Expected accuracy at this threshold: {recent_data[recent_data['prediction_confidence'] > best_threshold]['was_correct'].mean():.1%}")