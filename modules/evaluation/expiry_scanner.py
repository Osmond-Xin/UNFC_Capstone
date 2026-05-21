"""
Expiry Scanner

This module scans current stocks for pattern matches ahead of
upcoming expiry dates. It generates a watchlist of stocks that
match high-probability patterns.

Usage:
    from modules.evaluation import ExpiryScanner

    scanner = ExpiryScanner()
    watchlist = scanner.scan(stock_data, next_expiry)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from .predictive_patterns import PredictivePatterns


class ExpiryScanner:
    """
    Scan stocks for pattern matches before expiry dates

    This class scans current stock data to identify stocks that
    match predictive patterns, generating a watchlist for trading.
    """

    def __init__(self, patterns: PredictivePatterns = None):
        """
        Initialize the expiry scanner

        Args:
            patterns (PredictivePatterns, optional): Pattern definitions to use.
                If None, uses default patterns.
        """
        self.pattern_matcher = patterns if patterns else PredictivePatterns()

    def scan(
        self,
        stock_data: Dict[str, pd.DataFrame],
        next_expiry: pd.Timestamp = None,
        patterns_to_check: List[str] = None
    ) -> pd.DataFrame:
        """
        Scan all stocks for pattern matches

        Args:
            stock_data (dict): Dictionary mapping ticker -> DataFrame with features
            next_expiry (pd.Timestamp, optional): Next expiry date (for display)
            patterns_to_check (list, optional): Specific patterns to check.
                If None, checks all patterns.

        Returns:
            pd.DataFrame: Watchlist with columns:
                - Ticker: Stock symbol
                - Pattern: Matched pattern name(s)
                - Current_RSI: Current RSI value
                - Consecutive_Candles: Consecutive candle count
                - Candle_Direction: 'green' or 'red'
                - MA9_Dist, MA20_Dist, MA50_Dist: MA distances
                - Expected_Direction: 'Bullish' or 'Bearish'
                - Signal_Strength: Number of patterns matched
        """
        if patterns_to_check is None:
            patterns_to_check = list(self.pattern_matcher.patterns.keys())

        watchlist = []

        for ticker, df in stock_data.items():
            if df.empty or len(df) < 60:
                continue

            # Get latest data
            latest_row = df.iloc[-1]
            latest_date = df.index[-1]

            # Extract current indicators
            stock_indicators = self._extract_current_indicators(df, latest_row)

            # Check patterns
            matched_patterns = []
            expected_directions = []

            for pattern_name in patterns_to_check:
                if pattern_name not in self.pattern_matcher.patterns:
                    continue

                pattern_def = self.pattern_matcher.patterns[pattern_name]

                if pattern_def.condition(stock_indicators):
                    matched_patterns.append(pattern_name)
                    expected_directions.append(pattern_def.expected_direction)

            if matched_patterns:
                # Determine overall expected direction
                if len(expected_directions) > 0:
                    avg_direction = np.mean(expected_directions)
                    overall_direction = 'Bullish' if avg_direction > 0 else 'Bearish'
                else:
                    overall_direction = 'Neutral'

                watchlist.append({
                    'Ticker': ticker,
                    'Pattern': ', '.join(matched_patterns),
                    'Patterns_Matched': len(matched_patterns),
                    'Current_RSI': stock_indicators.get('RSI', np.nan),
                    'Consecutive_Candles': stock_indicators.get('Consecutive_Candles', 0),
                    'Candle_Direction': stock_indicators.get('Candle_Direction', 'unknown'),
                    'MA9_Dist': stock_indicators.get('MA9_dist', np.nan),
                    'MA20_Dist': stock_indicators.get('MA20_dist', np.nan),
                    'MA50_Dist': stock_indicators.get('MA50_dist', np.nan),
                    'Volume_Ratio': stock_indicators.get('Volume_Ratio', np.nan),
                    'Expected_Direction': overall_direction,
                    'Latest_Close': latest_row.get('Close', np.nan),
                    'Data_Date': latest_date.strftime('%Y-%m-%d')
                })

        if not watchlist:
            return pd.DataFrame()

        watchlist_df = pd.DataFrame(watchlist)
        # Sort by number of patterns matched
        # Use mergesort for stability (preserves input order for ties)
        # This allows the VolatilityFilter order (Winners first) to propagate
        watchlist_df = watchlist_df.sort_values('Patterns_Matched', ascending=False, kind='mergesort')
        watchlist_df = watchlist_df.reset_index(drop=True)

        return watchlist_df

    def scan_by_pattern(
        self,
        stock_data: Dict[str, pd.DataFrame],
        pattern_name: str
    ) -> pd.DataFrame:
        """
        Scan stocks for a specific pattern

        Args:
            stock_data (dict): Dictionary mapping ticker -> DataFrame with features
            pattern_name (str): Pattern name to scan for

        Returns:
            pd.DataFrame: Stocks matching the pattern
        """
        return self.scan(stock_data, patterns_to_check=[pattern_name])

    def get_bullish_watchlist(
        self,
        stock_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Get watchlist of stocks with bullish patterns

        Args:
            stock_data (dict): Dictionary mapping ticker -> DataFrame with features

        Returns:
            pd.DataFrame: Stocks with bullish patterns
        """
        # Bullish patterns
        bullish_patterns = [
            'Oversold_Reversal',
            'Strong_Uptrend',
            'Deep_Pullback',
            'Overbought_Momentum',
            'Volume_Surge_Reversal'
        ]

        watchlist = self.scan(stock_data, patterns_to_check=bullish_patterns)

        if not watchlist.empty:
            watchlist = watchlist[watchlist['Expected_Direction'] == 'Bullish']

        return watchlist

    def get_bearish_watchlist(
        self,
        stock_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Get watchlist of stocks with bearish patterns

        Args:
            stock_data (dict): Dictionary mapping ticker -> DataFrame with features

        Returns:
            pd.DataFrame: Stocks with bearish patterns
        """
        # Bearish patterns (Bearish_Exhaustion disabled for observation)
        bearish_patterns = []  # ['Bearish_Exhaustion']

        watchlist = self.scan(stock_data, patterns_to_check=bearish_patterns)

        if not watchlist.empty:
            watchlist = watchlist[watchlist['Expected_Direction'] == 'Bearish']

        return watchlist

    def _extract_current_indicators(
        self,
        df: pd.DataFrame,
        latest_row: pd.Series
    ) -> Dict:
        """
        Extract current technical indicators from stock data

        Args:
            df (pd.DataFrame): Stock data with features
            latest_row (pd.Series): Latest row of data

        Returns:
            dict: Current indicator values
        """
        indicators = {}

        # RSI
        if 'RSI' in df.columns:
            indicators['RSI'] = latest_row['RSI']

        # Consecutive candles
        if 'Consecutive_Count' in df.columns:
            indicators['Consecutive_Candles'] = latest_row['Consecutive_Count']

        if 'Consecutive_Direction' in df.columns:
            direction = latest_row['Consecutive_Direction']
            if direction == 1:
                indicators['Candle_Direction'] = 'green'
            elif direction == -1:
                indicators['Candle_Direction'] = 'red'
            else:
                indicators['Candle_Direction'] = 'unknown'

        # MA distances
        for period in [9, 20, 50]:
            col_name = f'MA_Distance_{period}'
            if col_name in df.columns:
                indicators[f'MA{period}_dist'] = latest_row[col_name]

        # Volume ratio
        if 'Volume_Ratio' in df.columns:
            indicators['Volume_Ratio'] = latest_row['Volume_Ratio']

        # MACD
        if 'MACD' in df.columns:
            indicators['MACD'] = latest_row['MACD']

        if 'MACD_Hist' in df.columns:
            indicators['MACD_Hist'] = latest_row['MACD_Hist']

        # BB Position
        if 'BB_Position' in df.columns:
            indicators['BB_Position'] = latest_row['BB_Position']

        # High point indicators
        if 'HEIGHT_SCORE' in df.columns:
            indicators['HEIGHT_SCORE'] = latest_row['HEIGHT_SCORE']

        if 'RESISTANCE_DIST_PCT' in df.columns:
            indicators['RESISTANCE_DIST_PCT'] = latest_row['RESISTANCE_DIST_PCT']

        if 'RESISTANCE_LEVEL' in df.columns:
            indicators['RESISTANCE_LEVEL'] = latest_row['RESISTANCE_LEVEL']

        return indicators

    def generate_report(
        self,
        watchlist: pd.DataFrame,
        next_expiry: pd.Timestamp,
        include_details: bool = True
    ) -> str:
        """
        Generate a formatted watchlist report

        Args:
            watchlist (pd.DataFrame): Watchlist from scan()
            next_expiry (pd.Timestamp): Next expiry date
            include_details (bool): Whether to include detailed indicators

        Returns:
            str: Formatted report string
        """
        lines = []
        lines.append("=" * 70)
        lines.append(f"EXPIRY DAY WATCHLIST - {next_expiry.strftime('%Y-%m-%d (%A)')}")
        lines.append("=" * 70)

        if watchlist.empty:
            lines.append("\nNo stocks match any patterns currently.")
            return "\n".join(lines)

        # Group by expected direction
        for direction in ['Bullish', 'Bearish']:
            direction_stocks = watchlist[watchlist['Expected_Direction'] == direction]

            if direction_stocks.empty:
                continue

            lines.append(f"\n{'📈' if direction == 'Bullish' else '📉'} {direction.upper()} SIGNALS ({len(direction_stocks)} stocks)")
            lines.append("-" * 50)

            for _, row in direction_stocks.iterrows():
                lines.append(f"\n  {row['Ticker']}:")
                lines.append(f"    Pattern(s): {row['Pattern']}")

                if include_details:
                    lines.append(f"    RSI: {row['Current_RSI']:.1f}")
                    lines.append(f"    Consecutive {row['Candle_Direction']} candles: {row['Consecutive_Candles']}")
                    lines.append(f"    MA20 Distance: {row['MA20_Dist']:+.2f}%")
                    lines.append(f"    Volume Ratio: {row['Volume_Ratio']:.2f}x")

        lines.append("\n" + "=" * 70)
        lines.append(f"Total: {len(watchlist)} stocks with pattern matches")

        return "\n".join(lines)
