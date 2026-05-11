"""
Technical Indicators Calculator

This module calculates technical analysis indicators including:
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Moving Averages (SMA for multiple periods)
- Bollinger Bands
- Volume features
- Consecutive candles count

Usage:
    from modules.features import TechnicalIndicators

    indicators = TechnicalIndicators(config={
        'rsi_period': 14,
        'ma_periods': [9, 20, 50]
    })

    df_with_indicators = indicators.calculate(df)
"""

import pandas as pd
import numpy as np
from typing import List
from .base import BaseFeature


class TechnicalIndicators(BaseFeature):
    """
    Calculate technical analysis indicators for stock price data

    This class implements common technical indicators used in trading analysis.
    All indicators are calculated using vectorized pandas operations for efficiency.
    """

    DEFAULT_CONFIG = {
        'rsi_period': 14,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'ma_periods': [9, 20, 50],
        'bb_period': 20,
        'bb_std': 2.0,
        'volume_ma_period': 20,
        'consecutive_lookback': 5
    }

    def get_default_config(self) -> dict:
        """Get default configuration"""
        return self.DEFAULT_CONFIG.copy()

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators and add them as new columns

        Args:
            df (pd.DataFrame): Input DataFrame with OHLCV columns

        Returns:
            pd.DataFrame: DataFrame with added indicator columns
        """
        # Validate input
        self.validate_input(df)

        # Make a copy to avoid modifying original
        df = df.copy()

        # Calculate all indicators
        df = self._calculate_rsi(df)
        df = self._calculate_macd(df)
        df = self._calculate_moving_averages(df)
        df = self._calculate_bollinger_bands(df)
        df = self._calculate_volume_features(df)
        df = self._calculate_consecutive_candles(df)

        return df

    def get_feature_names(self) -> List[str]:
        """
        Get list of feature column names this calculator adds

        Returns:
            list: List of feature column names
        """
        feature_names = [
            # RSI
            'RSI',

            # MACD
            'MACD',
            'MACD_Signal',
            'MACD_Hist',

            # Bollinger Bands
            'BB_Upper',
            'BB_Middle',
            'BB_Lower',
            'BB_Position',

            # Volume
            'Volume_SMA',
            'Volume_Ratio',

            # Consecutive candles
            'Consecutive_Count',
            'Consecutive_Direction'
        ]

        # Add moving averages
        for period in self.config['ma_periods']:
            feature_names.append(f'SMA_{period}')
            feature_names.append(f'MA_Distance_{period}')

        return feature_names

    def _calculate_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Relative Strength Index (RSI)

        RSI = 100 - (100 / (1 + RS))
        where RS = average_gain / average_loss

        Args:
            df (pd.DataFrame): Input DataFrame

        Returns:
            pd.DataFrame: DataFrame with RSI column added
        """
        period = self.config['rsi_period']
        prices = df['Close']

        # Calculate price changes
        delta = prices.diff()

        # Separate gains and losses
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        # Calculate rolling averages
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()

        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        df['RSI'] = rsi

        return df

    def _calculate_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate MACD (Moving Average Convergence Divergence)

        MACD = EMA(fast) - EMA(slow)
        Signal = EMA(MACD, signal_period)
        Histogram = MACD - Signal

        Args:
            df (pd.DataFrame): Input DataFrame

        Returns:
            pd.DataFrame: DataFrame with MACD columns added
        """
        fast = self.config['macd_fast']
        slow = self.config['macd_slow']
        signal = self.config['macd_signal']

        prices = df['Close']

        # Calculate EMAs
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()

        # Calculate MACD line
        macd = ema_fast - ema_slow

        # Calculate signal line
        macd_signal = macd.ewm(span=signal, adjust=False).mean()

        # Calculate histogram
        macd_hist = macd - macd_signal

        df['MACD'] = macd
        df['MACD_Signal'] = macd_signal
        df['MACD_Hist'] = macd_hist

        return df

    def _calculate_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Simple Moving Averages and distance from price

        For each period, calculates:
        - SMA_{period}: Simple moving average
        - MA_Distance_{period}: Percentage distance from price to MA

        Args:
            df (pd.DataFrame): Input DataFrame

        Returns:
            pd.DataFrame: DataFrame with MA columns added
        """
        prices = df['Close']

        for period in self.config['ma_periods']:
            # Calculate SMA
            sma = prices.rolling(window=period, min_periods=period).mean()
            df[f'SMA_{period}'] = sma

            # Calculate distance as percentage
            # distance = ((price - ma) / ma) * 100
            distance = ((prices - sma) / sma) * 100
            df[f'MA_Distance_{period}'] = distance

        return df

    def _calculate_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Bollinger Bands

        Middle Band = SMA(period)
        Upper Band = Middle + (std * multiplier)
        Lower Band = Middle - (std * multiplier)
        BB Position = (Close - Lower) / (Upper - Lower)

        Args:
            df (pd.DataFrame): Input DataFrame

        Returns:
            pd.DataFrame: DataFrame with Bollinger Bands columns added
        """
        period = self.config['bb_period']
        std_multiplier = self.config['bb_std']

        prices = df['Close']

        # Calculate middle band (SMA)
        bb_middle = prices.rolling(window=period, min_periods=period).mean()

        # Calculate standard deviation
        bb_std = prices.rolling(window=period, min_periods=period).std()

        # Calculate upper and lower bands
        bb_upper = bb_middle + (bb_std * std_multiplier)
        bb_lower = bb_middle - (bb_std * std_multiplier)

        # Calculate position within bands (0-1, where 0.5 is middle)
        bb_position = (prices - bb_lower) / (bb_upper - bb_lower)

        df['BB_Upper'] = bb_upper
        df['BB_Middle'] = bb_middle
        df['BB_Lower'] = bb_lower
        df['BB_Position'] = bb_position

        return df

    def _calculate_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volume-based features

        - Volume_SMA: Simple moving average of volume
        - Volume_Ratio: Current volume / Average volume

        Args:
            df (pd.DataFrame): Input DataFrame

        Returns:
            pd.DataFrame: DataFrame with volume feature columns added
        """
        period = self.config['volume_ma_period']
        volume = df['Volume']

        # Calculate volume moving average
        volume_sma = volume.rolling(window=period, min_periods=period).mean()

        # Calculate volume ratio
        volume_ratio = volume / volume_sma

        df['Volume_SMA'] = volume_sma
        df['Volume_Ratio'] = volume_ratio

        return df

    def _calculate_consecutive_candles(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate consecutive same-color candles count

        Counts how many consecutive candles in the same direction (up/down)
        from the most recent date backwards, capped at lookback.

        Up day (green): today's Close >= yesterday's Close
        Down day (red):  today's Close <  yesterday's Close

        Args:
            df (pd.DataFrame): Input DataFrame

        Returns:
            pd.DataFrame: DataFrame with consecutive candle columns added
        """
        lookback = self.config['consecutive_lookback']

        # Determine candle direction using Close-to-Close (today vs yesterday Close).
        # More robust than Open-to-Close: free data sources (e.g. Stooq) sometimes
        # have adjusted Open prices that differ from actual traded open.
        # Red (-1): today's Close < yesterday's Close; Green (1): otherwise.
        close = df['Close'].values
        prev_close = np.empty_like(close)
        prev_close[0] = close[0]   # first bar: no prior close, treat as flat (green)
        prev_close[1:] = close[:-1]
        direction = np.where(close < prev_close, -1, 1)
        n = len(direction)

        # Single pass over numpy int array to compute streak length,
        # capped at lookback.  ~500x faster than the nested .iloc loop.
        streak = np.ones(n, dtype=int)
        for k in range(1, n):
            if direction[k] == direction[k - 1]:
                streak[k] = min(streak[k - 1] + 1, lookback)
            # else: stays 1 (already initialised)

        df['Consecutive_Count'] = streak
        df['Consecutive_Direction'] = direction

        return df
