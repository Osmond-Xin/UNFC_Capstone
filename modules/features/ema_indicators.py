"""
EMA Indicators Calculator

This module calculates EMA-based indicators for the EMA signal strategy:
- EMA_20: 20-day Exponential Moving Average
- EMA_SLOPE_20: EMA slope (uptrend indicator)
- EMA_DIST_PCT_20: Percentage distance from price to EMA
- EMA_CROSS_20: Price-EMA cross indicator
- CANDLE_BODY_PCT: Candle body percentage (close-open)/open
- AVG_BODY_20: 20-day average absolute candle body size
- ADX_14: Average Directional Index (optional, when adx_period is set)

Usage:
    from modules.features import EMAIndicators

    indicators = EMAIndicators(config={'ema_period': 20})
    df_with_indicators = indicators.calculate(df)

    # With ADX trend filter:
    indicators = EMAIndicators(config={'ema_period': 5, 'adx_period': 14})
    df_with_indicators = indicators.calculate(df)
"""

import pandas as pd
import numpy as np
from typing import List
from .base import BaseFeature


class EMAIndicators(BaseFeature):
    """
    Calculate EMA-based technical indicators for stock price data

    This class implements indicators specifically designed for the
    EMA crossover/bounce trading strategy.
    """

    DEFAULT_CONFIG = {
        'ema_period': 20,
        'slope_lookback': 5,
        'body_avg_period': 20,
    }

    def get_default_config(self) -> dict:
        """Get default configuration"""
        return self.DEFAULT_CONFIG.copy()

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all EMA indicators and add them as new columns

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
        df = self._calculate_ema(df)
        df = self._calculate_ema_slope(df)
        df = self._calculate_ema_distance(df)
        df = self._calculate_ema_cross(df)
        df = self._calculate_candle_body(df)
        df = self._calculate_avg_body(df)

        # Calculate ADX if adx_period is configured
        if self.config.get('adx_period') is not None:
            df = self._calculate_adx(df)

        return df

    def get_feature_names(self) -> List[str]:
        """
        Get list of feature column names this calculator adds

        Returns:
            list: List of feature column names
        """
        period = self.config['ema_period']
        names = [
            f'EMA_{period}',
            f'EMA_SLOPE_{period}',
            f'EMA_DIST_PCT_{period}',
            f'EMA_CROSS_{period}',
            'CANDLE_BODY_PCT',
            f'AVG_BODY_{period}',
        ]
        adx_period = self.config.get('adx_period')
        if adx_period is not None:
            names.append(f'ADX_{adx_period}')
        return names

    def _calculate_ema(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Exponential Moving Average

        Args:
            df (pd.DataFrame): Input DataFrame

        Returns:
            pd.DataFrame: DataFrame with EMA column added
        """
        period = self.config['ema_period']
        prices = df['Close']

        ema = prices.ewm(span=period, adjust=False).mean()
        df[f'EMA_{period}'] = ema

        return df

    def _calculate_ema_slope(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate EMA slope (rate of change) to determine trend direction

        Slope = (EMA_today - EMA_n_days_ago) / EMA_n_days_ago * 100

        Positive slope indicates uptrend, negative indicates downtrend.

        Args:
            df (pd.DataFrame): Input DataFrame with EMA calculated

        Returns:
            pd.DataFrame: DataFrame with EMA_SLOPE column added
        """
        period = self.config['ema_period']
        lookback = self.config['slope_lookback']

        ema_col = f'EMA_{period}'
        ema = df[ema_col]

        # Calculate slope as percentage change over lookback period
        ema_shifted = ema.shift(lookback)
        slope = ((ema - ema_shifted) / ema_shifted) * 100

        df[f'EMA_SLOPE_{period}'] = slope

        return df

    def _calculate_ema_distance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate percentage distance from Close price to EMA

        Distance = (Close - EMA) / EMA * 100

        Positive: price above EMA
        Negative: price below EMA

        Args:
            df (pd.DataFrame): Input DataFrame with EMA calculated

        Returns:
            pd.DataFrame: DataFrame with EMA_DIST_PCT column added
        """
        period = self.config['ema_period']
        ema_col = f'EMA_{period}'

        distance = ((df['Close'] - df[ema_col]) / df[ema_col]) * 100
        df[f'EMA_DIST_PCT_{period}'] = distance

        return df

    def _calculate_ema_cross(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect price crossing EMA

        Cross signals:
         1: Price crossed above EMA (bullish cross)
        -1: Price crossed below EMA (bearish cross)
         0: No cross

        A candle is considered crossing when:
        - Low <= EMA <= High (candle touches EMA)

        Args:
            df (pd.DataFrame): Input DataFrame with EMA calculated

        Returns:
            pd.DataFrame: DataFrame with EMA_CROSS column added
        """
        period = self.config['ema_period']
        ema_col = f'EMA_{period}'

        ema = df[ema_col]
        high = df['High']
        low = df['Low']
        close = df['Close']
        prev_close = close.shift(1)

        # Candle touches EMA (low <= ema <= high)
        touches_ema = (low <= ema) & (ema <= high)

        # Determine cross direction based on close relative to previous close
        # If close > prev_close and touches EMA, bullish touch
        # If close < prev_close and touches EMA, bearish touch
        cross = pd.Series(0, index=df.index)
        cross[touches_ema & (close > prev_close)] = 1   # Bullish cross/touch
        cross[touches_ema & (close < prev_close)] = -1  # Bearish cross/touch

        df[f'EMA_CROSS_{period}'] = cross

        return df

    def _calculate_candle_body(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate candle body as percentage of open price

        Body_PCT = (Close - Open) / Open * 100

        Positive: green candle (bullish)
        Negative: red candle (bearish)

        Args:
            df (pd.DataFrame): Input DataFrame

        Returns:
            pd.DataFrame: DataFrame with CANDLE_BODY_PCT column added
        """
        body_pct = ((df['Close'] - df['Open']) / df['Open']) * 100
        df['CANDLE_BODY_PCT'] = body_pct

        return df

    def _calculate_avg_body(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate rolling average of absolute candle body size

        This is used to compare individual candles against the average
        to identify unusually large moves (kill candles).

        Args:
            df (pd.DataFrame): Input DataFrame with CANDLE_BODY_PCT calculated

        Returns:
            pd.DataFrame: DataFrame with AVG_BODY column added
        """
        period = self.config['ema_period']

        # Use absolute body size for averaging
        abs_body = df['CANDLE_BODY_PCT'].abs()
        avg_body = abs_body.rolling(window=period, min_periods=period).mean()

        df[f'AVG_BODY_{period}'] = avg_body

        return df

    @staticmethod
    def _wilder_smooth(series: pd.Series, n: int) -> pd.Series:
        """Wilder smoothing (used by ADX calculation)."""
        vals = series.values.astype(float)
        out = np.full(len(vals), np.nan)

        # Find the first position with n consecutive non-NaN values
        valid_mask = ~np.isnan(vals)
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) < n:
            return pd.Series(out, index=series.index)

        first_pos = valid_indices[n - 1]
        out[first_pos] = np.nanmean(vals[valid_indices[:n]])

        for i in range(first_pos + 1, len(vals)):
            if np.isnan(vals[i]):
                continue
            out[i] = (out[i - 1] * (n - 1) + vals[i]) / n

        return pd.Series(out, index=series.index)

    def _calculate_adx(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Average Directional Index (ADX) using Wilder smoothing.

        ADX measures trend strength regardless of direction.
        Values >= 20 indicate a trending market; < 20 indicates ranging/choppy.

        Args:
            df (pd.DataFrame): Input DataFrame with OHLCV columns

        Returns:
            pd.DataFrame: DataFrame with ADX_{period} column added
        """
        period = self.config['adx_period']
        high, low, close = df['High'], df['Low'], df['Close']

        # True Range
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)

        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = pd.Series(0.0, index=close.index)
        minus_dm = pd.Series(0.0, index=close.index)
        plus_dm[(up_move > 0) & (up_move > down_move)] = up_move
        minus_dm[(down_move > 0) & (down_move > up_move)] = down_move

        # Wilder-smoothed ATR, +DI, -DI
        atr = self._wilder_smooth(tr, period)
        plus_di = (self._wilder_smooth(plus_dm, period) / atr) * 100
        minus_di = (self._wilder_smooth(minus_dm, period) / atr) * 100

        # DX → ADX
        di_sum = plus_di + minus_di
        di_diff = (plus_di - minus_di).abs()
        dx = pd.Series(np.nan, index=close.index)
        mask = di_sum > 0
        dx[mask] = (di_diff[mask] / di_sum[mask]) * 100

        df[f'ADX_{period}'] = self._wilder_smooth(dx, period)

        return df
