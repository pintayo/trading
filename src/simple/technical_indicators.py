import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np
from config.model_config import INDICATORS

class TechnicalIndicators:
    
    @staticmethod
    def rsi(data, period=INDICATORS["rsi_period"]):
        """Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(data, fast=INDICATORS["macd_fast"], slow=INDICATORS["macd_slow"], signal=INDICATORS["macd_signal"]):
        """MACD Indicator"""
        ema_fast = data.ewm(span=fast).mean()
        ema_slow = data.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(data, period=INDICATORS["bb_period"], std=INDICATORS["bb_std"]):
        """Bollinger Bands"""
        sma = data.rolling(window=period).mean()
        std_dev = data.rolling(window=period).std()
        upper = sma + (std_dev * std)
        lower = sma - (std_dev * std)
        return upper, sma, lower
    
    @staticmethod
    def atr(high, low, close, period=INDICATORS["atr_period"]):
        """Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    @staticmethod
    def stochastic_oscillator(high, low, close, period=14):
        """Stochastic Oscillator"""
        low_min = low.rolling(window=period).min()
        high_max = high.rolling(window=period).max()
        k = 100 * (close - low_min) / (high_max - low_min)
        d = k.rolling(window=3).mean()
        return k, d

    @staticmethod
    def adx(high, low, close, period=14):
        """Average Directional Index"""
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0

        tr1 = pd.DataFrame(high - low)
        tr2 = pd.DataFrame(abs(high - close.shift(1)))
        tr3 = pd.DataFrame(abs(low - close.shift(1)))
        frames = [tr1, tr2, tr3]
        tr = pd.concat(frames, axis = 1, join = 'inner').max(axis = 1)
        atr = tr.rolling(window=period).mean()

        plus_di = 100 * (plus_dm.ewm(alpha = 1/period).mean() / atr)
        minus_di = abs(100 * (minus_dm.ewm(alpha = 1/period).mean() / atr))
        dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
        adx = ((dx.shift(1) * (period - 1)) + dx) / period
        return adx, plus_di, minus_di

    def add_all_indicators(self, df):
        """Add core technical indicators for simple model"""
        # RSI
        df['rsi'] = self.rsi(df['close'])
        
        # MACD
        df['macd'], df['macd_signal'], _ = self.macd(df['close'])
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = self.bollinger_bands(df['close'])
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # ATR
        df['atr'] = self.atr(df['high'], df['low'], df['close'])

        # Stochastic Oscillator
        df['stoch_k'], df['stoch_d'] = self.stochastic_oscillator(df['high'], df['low'], df['close'])

        # ADX
        df['adx'], df['plus_di'], df['minus_di'] = self.adx(df['high'], df['low'], df['close'])
        
        return df