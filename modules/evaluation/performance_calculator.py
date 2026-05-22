"""
Performance Calculator

This module calculates stock performance on expiry dates.
It identifies top/bottom performers and computes returns.

Usage:
    from modules.evaluation import PerformanceCalculator

    calculator = PerformanceCalculator()

    # Calculate returns for single expiry date
    performance_df = calculator.calculate_returns(stock_data, expiry_date)

    # Get top and bottom performers
    top, bottom = calculator.get_top_bottom(performance_df, top_n=10, bottom_n=10)
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List


class PerformanceCalculator:
    """
    Calculate stock performance on expiry dates

    This class computes returns for stocks on specific dates (typically expiry dates)
    and provides utilities to identify top and bottom performers.
    """

    def __init__(self):
        """Initialize the performance calculator"""
        pass

    def calculate_returns(
        self,
        stock_data: Dict[str, pd.DataFrame],
        expiry_date: pd.Timestamp,
        return_type: str = 'intraday'
    ) -> pd.DataFrame:
        """
        Calculate returns for all stocks on a specific date

        Args:
            stock_data (dict): Dictionary mapping ticker -> DataFrame
            expiry_date (pd.Timestamp): Date to calculate returns for
            return_type (str): Type of return to calculate
                - 'intraday': (Close - Open) / Open
                - 'close_to_close': (Close[t] - Close[t-1]) / Close[t-1]

        Returns:
            pd.DataFrame: Performance data with columns:
                - ticker: Stock symbol
                - return_pct: Return percentage
                - open_price: Open price on expiry date
                - close_price: Close price on expiry date
                - volume: Volume on expiry date
        """
        results = []

        expiry_date_str = expiry_date.strftime('%Y-%m-%d')

        for ticker, df in stock_data.items():
            try:
                # Find date by matching date part (handles timezone differences)
                date_to_use = self._find_date_by_datepart(df.index, expiry_date)

                if date_to_use is None:
                    # Try to find closest date within tolerance
                    date_to_use = self._find_closest_date(df.index, expiry_date)
                    if date_to_use is None:
                        continue

                # Get price data for the date
                row = df.loc[date_to_use]

                # Calculate return based on type
                if return_type == 'intraday':
                    # Intraday: (Close - Open) / Open
                    return_pct = ((row['Close'] - row['Open']) / row['Open']) * 100

                elif return_type == 'close_to_close':
                    # Close-to-close: need previous day's close
                    idx = df.index.get_loc(date_to_use)
                    if idx == 0:
                        continue  # No previous day

                    prev_close = df.iloc[idx - 1]['Close']
                    return_pct = ((row['Close'] - prev_close) / prev_close) * 100

                else:
                    raise ValueError(f"Unknown return_type: {return_type}")

                results.append({
                    'ticker': ticker,
                    'return_pct': return_pct,
                    'open_price': row['Open'],
                    'close_price': row['Close'],
                    'volume': row['Volume'],
                    'date_used': date_to_use.strftime('%Y-%m-%d')
                })

            except Exception as e:
                # Skip stocks with errors
                continue

        # Create DataFrame from results
        if not results:
            return pd.DataFrame()

        perf_df = pd.DataFrame(results)

        # Sort by return (descending)
        perf_df = perf_df.sort_values('return_pct', ascending=False)
        perf_df = perf_df.reset_index(drop=True)

        return perf_df

    def get_top_bottom(
        self,
        performance_df: pd.DataFrame,
        top_n: int = 10,
        bottom_n: int = 10
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get top and bottom performers from performance DataFrame

        Args:
            performance_df (pd.DataFrame): Performance data from calculate_returns()
            top_n (int): Number of top performers to return
            bottom_n (int): Number of bottom performers to return

        Returns:
            tuple: (top_performers, bottom_performers) as DataFrames
        """
        if performance_df.empty:
            return pd.DataFrame(), pd.DataFrame()

        # Already sorted by return_pct (descending) from calculate_returns()
        top_performers = performance_df.head(top_n).copy()
        bottom_performers = performance_df.tail(bottom_n).copy()

        # Reverse bottom performers for ascending order
        bottom_performers = bottom_performers.iloc[::-1].reset_index(drop=True)

        return top_performers, bottom_performers

    def calculate_multi_expiry(
        self,
        stock_data: Dict[str, pd.DataFrame],
        expiry_dates: List[pd.Timestamp],
        return_type: str = 'intraday'
    ) -> Dict[str, pd.DataFrame]:
        """
        Calculate returns for multiple expiry dates

        Args:
            stock_data (dict): Dictionary mapping ticker -> DataFrame
            expiry_dates (list): List of expiry dates
            return_type (str): Type of return to calculate

        Returns:
            dict: Dictionary mapping expiry_date_str -> performance_df
        """
        results = {}

        for expiry_date in expiry_dates:
            expiry_str = expiry_date.strftime('%Y-%m-%d')
            perf_df = self.calculate_returns(stock_data, expiry_date, return_type)

            if not perf_df.empty:
                results[expiry_str] = perf_df

        return results

    def calculate_aggregate_stats(
        self,
        performance_df: pd.DataFrame
    ) -> Dict:
        """
        Calculate aggregate statistics for performance data

        Args:
            performance_df (pd.DataFrame): Performance data

        Returns:
            dict: Statistics including mean, median, std, etc.
        """
        if performance_df.empty:
            return {}

        returns = performance_df['return_pct']

        stats = {
            'count': len(returns),
            'mean_return': returns.mean(),
            'median_return': returns.median(),
            'std_return': returns.std(),
            'min_return': returns.min(),
            'max_return': returns.max(),
            'positive_count': (returns > 0).sum(),
            'negative_count': (returns < 0).sum(),
            'positive_pct': ((returns > 0).sum() / len(returns)) * 100,
            'avg_positive_return': returns[returns > 0].mean() if (returns > 0).any() else 0,
            'avg_negative_return': returns[returns < 0].mean() if (returns < 0).any() else 0
        }

        return stats

    def _find_date_by_datepart(
        self,
        dates: pd.DatetimeIndex,
        target_date: pd.Timestamp
    ) -> Optional[pd.Timestamp]:
        """
        Find exact date match by comparing date part only (ignores timezone)

        This method handles timezone differences between expiry dates
        (which may be timezone-naive) and DataFrame indices (which may be UTC).

        Args:
            dates (pd.DatetimeIndex): Available dates (may have timezone)
            target_date (pd.Timestamp): Target date to find (may be timezone-naive)

        Returns:
            pd.Timestamp: Matching date from the index, or None if not found
        """
        # Vectorized O(log n) lookup via searchsorted on normalized index
        # Strip timezone from both sides to avoid tz-naive vs tz-aware comparison failure
        target_ts_norm = pd.Timestamp(target_date.date())
        normalized = dates.normalize()
        normalized_naive = normalized.tz_localize(None) if normalized.tz is not None else normalized
        loc = normalized_naive.searchsorted(target_ts_norm)
        if loc < len(normalized_naive) and normalized_naive[loc] == target_ts_norm:
            return dates[loc]
        return None

    def _find_closest_date(
        self,
        dates: pd.DatetimeIndex,
        target_date: pd.Timestamp,
        max_days: int = 5
    ) -> Optional[pd.Timestamp]:
        """
        Find the closest date to target in the index

        Args:
            dates (pd.DatetimeIndex): Available dates
            target_date (pd.Timestamp): Target date to find
            max_days (int): Maximum days difference allowed

        Returns:
            pd.Timestamp: Closest date, or None if none within max_days
        """
        # Calculate differences — strip timezone to avoid tz-naive/tz-aware TypeError
        dates_naive = dates.tz_localize(None) if dates.tz is not None else dates
        target_naive = target_date.tz_localize(None) if target_date.tzinfo is not None else target_date
        diffs = (dates_naive - target_naive).days.abs()

        # Find minimum difference
        min_diff_idx = diffs.argmin()
        min_diff = diffs[min_diff_idx]

        if min_diff <= max_days:
            return dates[min_diff_idx]
        else:
            return None

    def get_return_distribution(
        self,
        performance_df: pd.DataFrame,
        bins: int = 20
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get return distribution for histogram plotting

        Args:
            performance_df (pd.DataFrame): Performance data
            bins (int): Number of histogram bins

        Returns:
            tuple: (hist_values, bin_edges)
        """
        if performance_df.empty:
            return np.array([]), np.array([])

        returns = performance_df['return_pct'].values
        hist, edges = np.histogram(returns, bins=bins)

        return hist, edges
