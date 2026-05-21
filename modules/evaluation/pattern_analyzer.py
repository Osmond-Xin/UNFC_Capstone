"""
Pattern Analyzer

This module analyzes technical patterns for top/bottom performers
on expiry dates. It extracts technical indicators from the day before
expiry to identify predictive patterns.

Usage:
    from modules.evaluation import PatternAnalyzer

    analyzer = PatternAnalyzer()
    patterns = analyzer.analyze_patterns(stock_data, performers_df, expiry_date)
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List


class PatternAnalyzer:
    """
    Analyze technical patterns for stocks before expiry dates

    This class examines technical indicators on the day before expiry
    for top and bottom performers to identify predictive patterns.
    """

    def __init__(self):
        """Initialize the pattern analyzer"""
        pass

    def analyze_patterns(
        self,
        stock_data: Dict[str, pd.DataFrame],
        performers_df: pd.DataFrame,
        expiry_date: pd.Timestamp,
        category: str = 'Top'
    ) -> pd.DataFrame:
        """
        Analyze technical patterns for a set of performers

        Args:
            stock_data (dict): Dictionary mapping ticker -> DataFrame with features
            performers_df (pd.DataFrame): Performance data (from PerformanceCalculator)
            expiry_date (pd.Timestamp): The expiry date being analyzed
            category (str): Category label ('Top' or 'Bottom')

        Returns:
            pd.DataFrame: Pattern analysis results with columns:
                - Date: Expiry date
                - Ticker: Stock symbol
                - Open, Close, Change_Pct: Performance data
                - RSI: RSI value on day before expiry
                - Consecutive_Count: Consecutive candle count
                - Candle_Direction: 'green' or 'red'
                - MA9_dist, MA20_dist, MA50_dist: Distance from MAs
                - Category: 'Top' or 'Bottom'
        """
        results = []

        for _, row in performers_df.iterrows():
            ticker = row['ticker']

            if ticker not in stock_data:
                continue

            df = stock_data[ticker]
            pattern_data = self._extract_pattern_data(df, expiry_date, row, category)

            if pattern_data is not None:
                results.append(pattern_data)

        if not results:
            return pd.DataFrame()

        return pd.DataFrame(results)

    def analyze_all_performers(
        self,
        stock_data: Dict[str, pd.DataFrame],
        top_performers: pd.DataFrame,
        bottom_performers: pd.DataFrame,
        expiry_date: pd.Timestamp
    ) -> pd.DataFrame:
        """
        Analyze patterns for both top and bottom performers

        Args:
            stock_data (dict): Dictionary mapping ticker -> DataFrame with features
            top_performers (pd.DataFrame): Top performers from PerformanceCalculator
            bottom_performers (pd.DataFrame): Bottom performers from PerformanceCalculator
            expiry_date (pd.Timestamp): The expiry date being analyzed

        Returns:
            pd.DataFrame: Combined pattern analysis for all performers
        """
        top_patterns = self.analyze_patterns(
            stock_data, top_performers, expiry_date, category='Top'
        )
        bottom_patterns = self.analyze_patterns(
            stock_data, bottom_performers, expiry_date, category='Bottom'
        )

        # Combine results
        all_patterns = pd.concat([top_patterns, bottom_patterns], ignore_index=True)

        return all_patterns

    def analyze_multi_expiry(
        self,
        stock_data: Dict[str, pd.DataFrame],
        performance_by_expiry: Dict[str, pd.DataFrame],
        expiry_dates: List[pd.Timestamp],
        top_n: int = 10,
        bottom_n: int = 10
    ) -> pd.DataFrame:
        """
        Analyze patterns across multiple expiry dates

        Args:
            stock_data (dict): Dictionary mapping ticker -> DataFrame with features
            performance_by_expiry (dict): Performance data for each expiry date
            expiry_dates (list): List of expiry dates
            top_n (int): Number of top performers to analyze per expiry
            bottom_n (int): Number of bottom performers to analyze per expiry

        Returns:
            pd.DataFrame: Combined pattern analysis for all expiry dates
        """
        all_patterns = []

        for expiry_date in expiry_dates:
            expiry_str = expiry_date.strftime('%Y-%m-%d')

            if expiry_str not in performance_by_expiry:
                continue

            perf_df = performance_by_expiry[expiry_str]

            if perf_df.empty:
                continue

            # Get top and bottom performers
            top_df = perf_df.head(top_n)
            bottom_df = perf_df.tail(bottom_n)

            # Analyze patterns
            patterns = self.analyze_all_performers(
                stock_data, top_df, bottom_df, expiry_date
            )

            if not patterns.empty:
                all_patterns.append(patterns)

        if not all_patterns:
            return pd.DataFrame()

        return pd.concat(all_patterns, ignore_index=True)

    def _extract_pattern_data(
        self,
        df: pd.DataFrame,
        expiry_date: pd.Timestamp,
        performance_row: pd.Series,
        category: str
    ) -> Optional[Dict]:
        """
        Extract technical pattern data for a single stock

        Args:
            df (pd.DataFrame): Stock data with features calculated
            expiry_date (pd.Timestamp): Expiry date
            performance_row (pd.Series): Performance data for this stock
            category (str): 'Top' or 'Bottom'

        Returns:
            dict: Pattern data, or None if extraction fails
        """
        try:
            # Find the day before expiry
            expiry_date_only = expiry_date.date()

            # Get all dates before expiry
            df_before = df[df.index.map(lambda x: x.date()) < expiry_date_only]

            if df_before.empty:
                return None

            # Get the last trading day before expiry
            prev_row = df_before.iloc[-1]
            prev_date = df_before.index[-1]

            # Extract pattern data
            pattern_data = {
                'Date': expiry_date.strftime('%Y-%m-%d'),
                'Ticker': performance_row['ticker'],
                'Open': performance_row.get('open_price', np.nan),
                'Close': performance_row.get('close_price', np.nan),
                'Change_Pct': performance_row.get('return_pct', np.nan),
                'Category': category,
                'Prev_Date': prev_date.strftime('%Y-%m-%d'),
            }

            # Extract RSI
            if 'RSI' in df.columns:
                pattern_data['RSI'] = prev_row['RSI']
            else:
                pattern_data['RSI'] = np.nan

            # Extract consecutive candles
            if 'Consecutive_Count' in df.columns:
                pattern_data['Consecutive_Candles'] = prev_row['Consecutive_Count']
            else:
                pattern_data['Consecutive_Candles'] = np.nan

            if 'Consecutive_Direction' in df.columns:
                direction = prev_row['Consecutive_Direction']
                # Convert 1/-1 to 'green'/'red'
                if direction == 1:
                    pattern_data['Candle_Direction'] = 'green'
                elif direction == -1:
                    pattern_data['Candle_Direction'] = 'red'
                else:
                    pattern_data['Candle_Direction'] = 'unknown'
            else:
                pattern_data['Candle_Direction'] = 'unknown'

            # Extract MA distances
            for period in [9, 20, 50]:
                col_name = f'MA_Distance_{period}'
                output_name = f'MA{period}_dist'
                if col_name in df.columns:
                    pattern_data[output_name] = prev_row[col_name]
                else:
                    pattern_data[output_name] = np.nan

            # Extract MACD
            if 'MACD' in df.columns:
                pattern_data['MACD'] = prev_row['MACD']
            else:
                pattern_data['MACD'] = np.nan

            if 'MACD_Hist' in df.columns:
                pattern_data['MACD_Hist'] = prev_row['MACD_Hist']
            else:
                pattern_data['MACD_Hist'] = np.nan

            # Extract Volume Ratio
            if 'Volume_Ratio' in df.columns:
                pattern_data['Volume_Ratio'] = prev_row['Volume_Ratio']
            else:
                pattern_data['Volume_Ratio'] = np.nan

            # Extract BB Position
            if 'BB_Position' in df.columns:
                pattern_data['BB_Position'] = prev_row['BB_Position']
            else:
                pattern_data['BB_Position'] = np.nan

            # Extract High Point indicators
            if 'HEIGHT_SCORE' in df.columns:
                pattern_data['HEIGHT_SCORE'] = prev_row['HEIGHT_SCORE']
            else:
                pattern_data['HEIGHT_SCORE'] = np.nan

            if 'RESISTANCE_DIST_PCT' in df.columns:
                pattern_data['RESISTANCE_DIST_PCT'] = prev_row['RESISTANCE_DIST_PCT']
            else:
                pattern_data['RESISTANCE_DIST_PCT'] = np.nan

            return pattern_data

        except Exception as e:
            return None

    def get_pattern_summary(
        self,
        all_patterns: pd.DataFrame
    ) -> Dict:
        """
        Generate summary statistics for pattern data

        Args:
            all_patterns (pd.DataFrame): Combined pattern data

        Returns:
            dict: Summary statistics
        """
        if all_patterns.empty:
            return {}

        summary = {
            'total_samples': len(all_patterns),
            'top_count': len(all_patterns[all_patterns['Category'] == 'Top']),
            'bottom_count': len(all_patterns[all_patterns['Category'] == 'Bottom']),
            'unique_tickers': all_patterns['Ticker'].nunique(),
            'unique_dates': all_patterns['Date'].nunique(),
        }

        # Calculate means by category
        for category in ['Top', 'Bottom']:
            cat_data = all_patterns[all_patterns['Category'] == category]
            if not cat_data.empty:
                summary[f'{category.lower()}_mean_rsi'] = cat_data['RSI'].mean()
                summary[f'{category.lower()}_mean_change'] = cat_data['Change_Pct'].mean()
                if 'Consecutive_Candles' in cat_data.columns:
                    summary[f'{category.lower()}_mean_consecutive'] = cat_data['Consecutive_Candles'].mean()

        return summary
