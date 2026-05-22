"""
Pattern-Based Trading Strategies

This module implements pattern-based trading strategies identified from
SPY expiry date analysis.

Strategies:
1. RSIReversalStrategy: Oversold reversal (RSI < 30 + consecutive down candles)
2. ConsecutiveCandleStrategy: Trend exhaustion (multiple consecutive candles)
3. MADistanceStrategy: Mean reversion (price far from moving average)
4. VolumeMACD_ComboStrategy: Volume surge + MACD confirmation

Usage:
    from modules.models import RSIReversalStrategy

    strategy = RSIReversalStrategy(config={'rsi_threshold': 30})
    signal = strategy.predict(df_with_features, expiry_date)
"""

import pandas as pd
from typing import Dict
from .base import BaseStrategy


class RSIReversalStrategy(BaseStrategy):
    """
    RSI Reversal Strategy

    Pattern: Oversold conditions suggest reversal
    Signal: Long when RSI < threshold AND consecutive down candles

    Logic:
    - RSI < 30: Oversold condition
    - 3+ consecutive red candles: Selling pressure exhausted
    - Expected: Mean reversion bounce
    """

    DEFAULT_CONFIG = {
        'rsi_threshold': 30,
        'min_consecutive': 3,
        'lookback_days': 1  # Days before expiry to check
    }

    def get_default_config(self) -> Dict:
        return self.DEFAULT_CONFIG.copy()

    def get_required_features(self) -> list:
        return ['RSI', 'Consecutive_Count', 'Consecutive_Direction']

    def predict(self, df: pd.DataFrame, expiry_date: pd.Timestamp) -> int:
        """
        Generate signal based on RSI reversal pattern

        Returns:
            1: Long (expect bounce)
            0: Neutral
        """
        try:
            # Validate data
            if not self.validate_data(df):
                return 0

            # Get lookback date (1 day before expiry)
            lookback = self.config['lookback_days']

            # Get RSI value
            rsi = self._get_value_at_date(df, 'RSI', expiry_date, lookback)
            if rsi is None:
                return 0

            # Get consecutive candles
            consec_count = self._get_value_at_date(df, 'Consecutive_Count', expiry_date, lookback)
            consec_dir = self._get_value_at_date(df, 'Consecutive_Direction', expiry_date, lookback)

            if consec_count is None or consec_dir is None:
                return 0

            # Check pattern: RSI oversold + consecutive down candles
            rsi_oversold = rsi < self.config['rsi_threshold']
            enough_consecutive = consec_count >= self.config['min_consecutive']
            down_direction = consec_dir == -1  # Red candles

            if rsi_oversold and enough_consecutive and down_direction:
                return 1  # Long signal (expect bounce)

            return 0

        except Exception as e:
            return 0

    def get_metadata(self) -> Dict:
        return {
            'name': 'RSIReversalStrategy',
            'version': '1.0',
            'description': 'Long when RSI oversold with consecutive down candles',
            'config': self.config
        }


class ConsecutiveCandleStrategy(BaseStrategy):
    """
    Consecutive Candle Exhaustion Strategy

    Pattern: Extended trends tend to reverse
    Signal: Counter-trend when too many consecutive candles

    Logic:
    - 4+ consecutive candles in same direction: Trend exhaustion
    - Expected: Mean reversion
    """

    DEFAULT_CONFIG = {
        'min_consecutive': 4,
        'use_ma_filter': True,  # Only trade if price above/below MA
        'ma_column': 'SMA_20',
        'lookback_days': 1
    }

    def get_default_config(self) -> Dict:
        return self.DEFAULT_CONFIG.copy()

    def get_required_features(self) -> list:
        base_features = ['Consecutive_Count', 'Consecutive_Direction', 'Close']
        if self.config['use_ma_filter']:
            base_features.append(self.config['ma_column'])
        return base_features

    def predict(self, df: pd.DataFrame, expiry_date: pd.Timestamp) -> int:
        """
        Generate signal based on consecutive candle exhaustion

        Returns:
            1: Long (expect up after down exhaustion)
            -1: Short (expect down after up exhaustion)
            0: Neutral
        """
        try:
            if not self.validate_data(df):
                return 0

            lookback = self.config['lookback_days']

            # Get consecutive candles
            consec_count = self._get_value_at_date(df, 'Consecutive_Count', expiry_date, lookback)
            consec_dir = self._get_value_at_date(df, 'Consecutive_Direction', expiry_date, lookback)

            if consec_count is None or consec_dir is None:
                return 0

            # Check if enough consecutive candles
            if consec_count < self.config['min_consecutive']:
                return 0

            # Optional: MA filter
            if self.config['use_ma_filter']:
                close = self._get_value_at_date(df, 'Close', expiry_date, lookback)
                ma = self._get_value_at_date(df, self.config['ma_column'], expiry_date, lookback)

                if close is None or ma is None:
                    return 0

                # For long: consecutive down + price above MA
                # For short: consecutive up + price below MA
                if consec_dir == -1:  # Down candles
                    if close > ma:
                        return 1  # Long (expect reversal up)
                else:  # Up candles
                    if close < ma:
                        return -1  # Short (expect reversal down)

                return 0
            else:
                # Without MA filter: simple reversal
                if consec_dir == -1:
                    return 1  # Long after down exhaustion
                else:
                    return -1  # Short after up exhaustion

        except Exception as e:
            return 0

    def get_metadata(self) -> Dict:
        return {
            'name': 'ConsecutiveCandleStrategy',
            'version': '1.0',
            'description': 'Counter-trend signal on consecutive candle exhaustion',
            'config': self.config
        }


class MADistanceStrategy(BaseStrategy):
    """
    MA Distance Mean Reversion Strategy

    Pattern: Price far from moving average tends to revert
    Signal: Counter-trend when price deviates significantly from MA

    Logic:
    - Price > +X% from MA: Overbought, expect pullback
    - Price < -X% from MA: Oversold, expect bounce
    """

    DEFAULT_CONFIG = {
        'ma_column': 'MA_Distance_20',
        'upper_threshold': 5.0,  # % above MA for short
        'lower_threshold': -5.0,  # % below MA for long
        'use_rsi_filter': True,
        'rsi_upper': 60,
        'rsi_lower': 40,
        'lookback_days': 1
    }

    def get_default_config(self) -> Dict:
        return self.DEFAULT_CONFIG.copy()

    def get_required_features(self) -> list:
        features = [self.config['ma_column']]
        if self.config['use_rsi_filter']:
            features.append('RSI')
        return features

    def predict(self, df: pd.DataFrame, expiry_date: pd.Timestamp) -> int:
        """
        Generate signal based on MA distance

        Returns:
            1: Long (price below MA, expect bounce)
            -1: Short (price above MA, expect pullback)
            0: Neutral
        """
        try:
            if not self.validate_data(df):
                return 0

            lookback = self.config['lookback_days']

            # Get MA distance
            ma_dist = self._get_value_at_date(df, self.config['ma_column'], expiry_date, lookback)
            if ma_dist is None:
                return 0

            # Check RSI filter if enabled
            if self.config['use_rsi_filter']:
                rsi = self._get_value_at_date(df, 'RSI', expiry_date, lookback)
                if rsi is None:
                    return 0

                # Long signal: price far below MA + RSI not too low
                if ma_dist < self.config['lower_threshold']:
                    if rsi < self.config['rsi_lower']:
                        return 1

                # Short signal: price far above MA + RSI not too high
                elif ma_dist > self.config['upper_threshold']:
                    if rsi > self.config['rsi_upper']:
                        return -1

            else:
                # Without RSI filter
                if ma_dist < self.config['lower_threshold']:
                    return 1  # Long
                elif ma_dist > self.config['upper_threshold']:
                    return -1  # Short

            return 0

        except Exception as e:
            return 0

    def get_metadata(self) -> Dict:
        return {
            'name': 'MADistanceStrategy',
            'version': '1.0',
            'description': 'Mean reversion based on distance from moving average',
            'config': self.config
        }


class VolumeMACD_ComboStrategy(BaseStrategy):
    """
    Volume + MACD Combo Strategy

    Pattern: Volume surge + MACD confirmation = strong trend
    Signal: Trend-following when volume and momentum align

    Logic:
    - High volume ratio (>2x average): Institutional activity
    - MACD histogram positive/negative: Momentum direction
    - Expected: Trend continuation
    """

    DEFAULT_CONFIG = {
        'volume_ratio_threshold': 2.0,
        'macd_hist_threshold': 0,  # MACD histogram threshold
        'use_price_filter': True,
        'lookback_days': 1
    }

    def get_default_config(self) -> Dict:
        return self.DEFAULT_CONFIG.copy()

    def get_required_features(self) -> list:
        return ['Volume_Ratio', 'MACD_Hist']

    def predict(self, df: pd.DataFrame, expiry_date: pd.Timestamp) -> int:
        """
        Generate signal based on volume and MACD

        Returns:
            1: Long (volume surge + positive MACD)
            -1: Short (volume surge + negative MACD)
            0: Neutral
        """
        try:
            if not self.validate_data(df):
                return 0

            lookback = self.config['lookback_days']

            # Get volume ratio
            vol_ratio = self._get_value_at_date(df, 'Volume_Ratio', expiry_date, lookback)
            if vol_ratio is None:
                return 0

            # Get MACD histogram
            macd_hist = self._get_value_at_date(df, 'MACD_Hist', expiry_date, lookback)
            if macd_hist is None:
                return 0

            # Check volume surge
            volume_surge = vol_ratio > self.config['volume_ratio_threshold']

            if not volume_surge:
                return 0

            # Check MACD direction
            if macd_hist > self.config['macd_hist_threshold']:
                return 1  # Long (momentum up + volume)
            elif macd_hist < -self.config['macd_hist_threshold']:
                return -1  # Short (momentum down + volume)

            return 0

        except Exception as e:
            return 0

    def get_metadata(self) -> Dict:
        return {
            'name': 'VolumeMACD_ComboStrategy',
            'version': '1.0',
            'description': 'Trend following with volume surge and MACD confirmation',
            'config': self.config
        }
