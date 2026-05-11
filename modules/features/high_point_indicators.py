"""
High Point Indicators

This module detects whether a stock is at a technical high point by
calculating overbought sub-scores and resistance/support levels.

Must run AFTER TechnicalIndicators in the pipeline, as it depends
on RSI, BB_Position, MA_Distance_20, and MACD_Hist columns.

Usage:
    from modules.features import TechnicalIndicators, HighPointIndicators, FeaturePipeline

    pipeline = FeaturePipeline([TechnicalIndicators(), HighPointIndicators()])
    df = pipeline.calculate(df)
"""

import pandas as pd
import numpy as np
from typing import List
from .base import BaseFeature
from ..config.high_point_params import (
    HEIGHT_SCORE_WEIGHTS,
    HEIGHT_SCORE_THRESHOLDS,
    RESISTANCE_CONFIG,
    calculate_height_score,
)


class HighPointIndicators(BaseFeature):
    """
    Calculate high point detection indicators.

    Produces two groups of columns:
    A. Overbought sub-scores (depend on TechnicalIndicators):
       RSI_OB_SCORE, BB_OB_SCORE, MA_OB_SCORE, MACD_OB_SCORE, HEIGHT_SCORE
    B. Resistance/support levels (from OHLCV only):
       RESISTANCE_LEVEL, RESISTANCE_DIST_PCT, SUPPORT_LEVEL, SUPPORT_DIST_PCT,
       ROUND_RESISTANCE, ROUND_DIST_PCT
    """

    DEFAULT_CONFIG = {
        'pivot_lookback': RESISTANCE_CONFIG['pivot_lookback'],
        'min_prominence_pct': RESISTANCE_CONFIG['min_prominence_pct'],
        'max_age_bars': RESISTANCE_CONFIG['max_age_bars'],
        'macd_hist_lookback': HEIGHT_SCORE_THRESHOLDS['macd_hist_lookback'],
    }

    def get_default_config(self) -> dict:
        return self.DEFAULT_CONFIG.copy()

    def get_required_columns(self) -> List[str]:
        return ['Open', 'High', 'Low', 'Close', 'Volume',
                'RSI', 'BB_Position', 'MA_Distance_20', 'MACD_Hist',
                'SMA_9', 'SMA_20', 'SMA_50']

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        self.validate_input(df)
        df = df.copy()

        # A. Overbought sub-scores
        df = self._calculate_overbought_scores(df)

        # B. Resistance / support levels
        df = self._calculate_resistance_support(df)
        df = self._calculate_round_resistance(df)

        # C. Trend context
        df = self._calculate_trend_context(df)

        return df

    def get_feature_names(self) -> List[str]:
        return [
            'RSI_OB_SCORE',
            'BB_OB_SCORE',
            'MA_OB_SCORE',
            'MACD_OB_SCORE',
            'HEIGHT_SCORE',
            'RESISTANCE_LEVEL',
            'RESISTANCE_DIST_PCT',
            'SUPPORT_LEVEL',
            'SUPPORT_DIST_PCT',
            'ROUND_RESISTANCE',
            'ROUND_DIST_PCT',
            'TREND_SMA_ALIGN',
            'TREND_SCORE',
            'TREND_PHASE',
        ]

    # -----------------------------------------------------------------
    # A. Overbought sub-scores
    # -----------------------------------------------------------------
    def _calculate_overbought_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        rsi_start = HEIGHT_SCORE_THRESHOLDS['rsi_ob_start']
        rsi_full = HEIGHT_SCORE_THRESHOLDS['rsi_ob_full']
        rsi_range = rsi_full - rsi_start

        # RSI_OB_SCORE: (RSI - 50) / 20 clamped 0-1
        df['RSI_OB_SCORE'] = ((df['RSI'] - rsi_start) / rsi_range).clip(0, 1)

        # BB_OB_SCORE: (BB_Position - 0.5) * 2 clamped 0-1
        bb_start = HEIGHT_SCORE_THRESHOLDS['bb_ob_start']
        df['BB_OB_SCORE'] = ((df['BB_Position'] - bb_start) * 2).clip(0, 1)

        # MA_OB_SCORE: MA_Distance_20 / 5 clamped 0-1
        ma_full = HEIGHT_SCORE_THRESHOLDS['ma_dist_full']
        df['MA_OB_SCORE'] = (df['MA_Distance_20'] / ma_full).clip(0, 1)

        # MACD_OB_SCORE: MACD_Hist normalized by rolling 20-day max
        lookback = self.config['macd_hist_lookback']
        macd_hist = df['MACD_Hist']
        rolling_max = macd_hist.abs().rolling(window=lookback, min_periods=1).max()
        # Avoid divide-by-zero
        safe_max = rolling_max.replace(0, np.nan)
        df['MACD_OB_SCORE'] = (macd_hist / safe_max).clip(0, 1).fillna(0)

        # HEIGHT_SCORE: weighted sum
        df['HEIGHT_SCORE'] = (
            HEIGHT_SCORE_WEIGHTS['rsi_overbought'] * df['RSI_OB_SCORE'] +
            HEIGHT_SCORE_WEIGHTS['bb_position'] * df['BB_OB_SCORE'] +
            HEIGHT_SCORE_WEIGHTS['ma_distance'] * df['MA_OB_SCORE'] +
            HEIGHT_SCORE_WEIGHTS['macd_momentum'] * df['MACD_OB_SCORE']
        ).clip(0, 1)

        return df

    # -----------------------------------------------------------------
    # B. Resistance / Support via pivot detection
    # -----------------------------------------------------------------
    def _calculate_resistance_support(self, df: pd.DataFrame) -> pd.DataFrame:
        lookback = self.config['pivot_lookback']
        min_prom = self.config['min_prominence_pct'] / 100.0
        max_age = self.config['max_age_bars']

        highs = df['High'].values
        lows = df['Low'].values
        closes = df['Close'].values
        n = len(df)

        resistance_levels = np.full(n, np.nan)
        resistance_dists = np.full(n, np.nan)
        support_levels = np.full(n, np.nan)
        support_dists = np.full(n, np.nan)

        # Pre-detect all pivot highs and lows (no look-ahead)
        # A bar at index i is a pivot high if High[i] is the max of
        # High[i-lookback : i+1] (only past data).
        # We check it once we have enough data on both sides, but
        # to avoid look-ahead we only confirm pivots for bars where
        # we have lookback bars BEFORE and lookback bars AFTER,
        # where "after" is still in the past relative to current bar.

        # For each bar t, find confirmed pivots up to bar t-lookback
        pivot_highs = []  # list of (index, price)
        pivot_lows = []   # list of (index, price)

        for t in range(n):
            # Check if bar (t - lookback) is a pivot.
            # Bar c = t - lookback has lookback bars before it and
            # lookback bars after it (bars c-lookback..c..c+lookback = t).
            c = t - lookback
            if c >= lookback:
                # Check pivot high at c
                window_start = c - lookback
                window_end = c + lookback + 1  # exclusive
                window_highs = highs[window_start:window_end]
                if highs[c] == window_highs.max():
                    # Check prominence: must be min_prom above the lowest
                    # trough in the window
                    window_lows_min = lows[window_start:window_end].min()
                    if window_lows_min > 0:
                        prominence = (highs[c] - window_lows_min) / window_lows_min
                        if prominence >= min_prom:
                            pivot_highs.append((c, highs[c]))

                # Check pivot low at c
                window_lows = lows[window_start:window_end]
                if lows[c] == window_lows.min():
                    window_highs_max = highs[window_start:window_end].max()
                    if lows[c] > 0:
                        prominence = (window_highs_max - lows[c]) / lows[c]
                        if prominence >= min_prom:
                            pivot_lows.append((c, lows[c]))

            # Find nearest resistance (pivot high above current close)
            close = closes[t]
            if close > 0:
                best_resistance = np.nan
                for idx, price in reversed(pivot_highs):
                    if t - idx > max_age:
                        break
                    if price > close:
                        if np.isnan(best_resistance) or price < best_resistance:
                            best_resistance = price
                resistance_levels[t] = best_resistance
                if not np.isnan(best_resistance):
                    resistance_dists[t] = (best_resistance - close) / close * 100

                # Find nearest support (pivot low below current close)
                best_support = np.nan
                for idx, price in reversed(pivot_lows):
                    if t - idx > max_age:
                        break
                    if price < close:
                        if np.isnan(best_support) or price > best_support:
                            best_support = price
                support_levels[t] = best_support
                if not np.isnan(best_support):
                    support_dists[t] = (close - best_support) / close * 100

        df['RESISTANCE_LEVEL'] = resistance_levels
        df['RESISTANCE_DIST_PCT'] = resistance_dists
        df['SUPPORT_LEVEL'] = support_levels
        df['SUPPORT_DIST_PCT'] = support_dists

        return df

    # -----------------------------------------------------------------
    # C. Trend context via SMA alignment
    # -----------------------------------------------------------------
    def _calculate_trend_context(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate trend context from SMA alignment.

        TREND_SMA_ALIGN: +1 (bullish SMA9>SMA20>SMA50), -1 (bearish), 0 (mixed)
        TREND_SCORE: 0-1 based on price position relative to SMAs
        TREND_PHASE: 'strong_up', 'up', 'neutral', 'down', 'strong_down'
        """
        close = df['Close']
        sma9 = df['SMA_9']
        sma20 = df['SMA_20']
        sma50 = df['SMA_50']

        # SMA alignment: +1 if SMA9 > SMA20 > SMA50 (bullish stack)
        bullish_align = (sma9 > sma20) & (sma20 > sma50)
        bearish_align = (sma9 < sma20) & (sma20 < sma50)
        df['TREND_SMA_ALIGN'] = np.where(bullish_align, 1, np.where(bearish_align, -1, 0))

        # Trend score: how many SMAs is price above (0-1)
        above_sma9 = (close > sma9).astype(float)
        above_sma20 = (close > sma20).astype(float)
        above_sma50 = (close > sma50).astype(float)
        df['TREND_SCORE'] = ((above_sma9 + above_sma20 + above_sma50) / 3.0).clip(0, 1)

        # Trend phase classification
        def classify_trend(row):
            align = row['TREND_SMA_ALIGN']
            score = row['TREND_SCORE']
            if align == 1 and score >= 0.9:
                return 'strong_up'
            elif align == 1 or score >= 0.67:
                return 'up'
            elif align == -1 and score <= 0.1:
                return 'strong_down'
            elif align == -1 or score <= 0.33:
                return 'down'
            else:
                return 'neutral'

        df['TREND_PHASE'] = df.apply(classify_trend, axis=1)

        return df

    # -----------------------------------------------------------------
    # Round number resistance
    # -----------------------------------------------------------------
    def _calculate_round_resistance(self, df: pd.DataFrame) -> pd.DataFrame:
        round_levels = RESISTANCE_CONFIG['round_levels']

        def next_round(price):
            if price <= 0 or np.isnan(price):
                return np.nan
            for threshold, step in round_levels:
                if price < threshold:
                    return np.ceil(price / step) * step
            # Fallback (should not reach here due to inf)
            return np.ceil(price / 10) * 10

        closes = df['Close']
        round_res = closes.apply(next_round)
        df['ROUND_RESISTANCE'] = round_res
        df['ROUND_DIST_PCT'] = np.where(
            closes > 0,
            (round_res - closes) / closes * 100,
            np.nan
        )

        return df
