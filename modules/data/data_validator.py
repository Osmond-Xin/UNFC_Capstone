"""
Data Validator for Stock Data Quality Checks

This module ensures data quality before analysis by checking:
- Completeness (required columns present)
- Price/volume outliers (extreme values)
- Single-day extreme price changes (>50% daily change)
- Date continuity (no large gaps)
- Sufficient data points
- OHLC logic consistency

Note: Price level outliers are only reported when they affect more than 5% of
rows, which catches likely cache corruption while avoiding one-off legitimate
large moves.

Usage:
    from modules.data import DataValidator

    validator = DataValidator()
    is_valid, errors = validator.validate(df, 'AAPL')

    if not is_valid:
        print(f"Validation errors: {errors}")
"""

import pandas as pd
import numpy as np
from typing import Tuple, List


class DataValidator:
    """
    Data quality validator for stock price data

    Validates:
    - Required columns (Open, High, Low, Close, Volume)
    - Data completeness
    - Outlier detection
    - Date continuity
    - Sufficient data points
    """

    REQUIRED_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume']
    MIN_RECORDS = 30  # Minimum number of records required

    def __init__(self, min_records=30):
        """
        Initialize the data validator

        Args:
            min_records (int): Minimum number of records required (default: 30)
        """
        self.min_records = min_records

    def validate(self, df: pd.DataFrame, ticker: str) -> Tuple[bool, List[str]]:
        """
        Comprehensive data validation

        Args:
            df (pd.DataFrame): DataFrame with stock data
            ticker (str): Ticker symbol (for error messages)

        Returns:
            tuple: (is_valid, error_messages)
                - is_valid (bool): True if all checks pass
                - error_messages (list): List of validation error messages
        """
        errors = []

        # Check 1: Required columns
        if not self.check_completeness(df):
            missing = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
            errors.append(f"Missing required columns: {missing}")

        # Check 2: Sufficient data points
        if len(df) < self.min_records:
            errors.append(f"Insufficient data: {len(df)} records (minimum: {self.min_records})")

        # Check 3: Empty data
        if df.empty:
            errors.append("DataFrame is empty")
            return False, errors

        # Check 4: Check for NaN in critical columns
        for col in self.REQUIRED_COLUMNS:
            if col in df.columns:
                nan_count = df[col].isna().sum()
                if nan_count > 0:
                    errors.append(f"Column '{col}' has {nan_count} NaN values")

        # Check 5: Date continuity
        continuity_check = self.check_date_continuity(df)
        if not continuity_check:
            errors.append("Large date gaps detected (>10 trading days)")

        # Check 6: Outliers and anomalies
        outlier_errors = self.check_outliers(df)
        if outlier_errors:
            errors.extend(outlier_errors)

        # Check 7: Logical consistency (High >= Low, etc.)
        ohlc_issues = self._check_ohlc_logic_detailed(df)
        if ohlc_issues:
            errors.append(
                f"OHLC logic data quality issues detected "
                f"({ohlc_issues['count']} rows affected, {ohlc_issues['percent']:.1f}%)"
            )

        is_valid = len(errors) == 0
        return is_valid, errors

    def check_completeness(self, df: pd.DataFrame) -> bool:
        """
        Check if all required columns are present

        Args:
            df (pd.DataFrame): DataFrame to check

        Returns:
            bool: True if all required columns present
        """
        return all(col in df.columns for col in self.REQUIRED_COLUMNS)

    def check_outliers(self, df: pd.DataFrame) -> List[str]:
        """
        Check for data anomalies

        Checks:
        1. Extreme volume outliers (using IQR method)
        2. Extreme close-price level outliers (using IQR method)
        3. Single-day extreme price changes (>50% daily change)

        Args:
            df (pd.DataFrame): DataFrame to check

        Returns:
            list: List of anomaly error messages (empty if no issues)
        """
        errors = []

        # Check 1: Volume outliers (IQR method is appropriate for volume)
        if 'Volume' in df.columns:
            Q1 = df['Volume'].quantile(0.25)
            Q3 = df['Volume'].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR

            outliers = ((df['Volume'] < lower_bound) | (df['Volume'] > upper_bound)).sum()

            if outliers > 0:
                outlier_pct = (outliers / len(df)) * 100
                if outlier_pct > 5:  # Only report if >5% are outliers
                    errors.append(f"Volume has {outliers} outliers ({outlier_pct:.1f}%)")

        # Check 2: Close price level outliers (likely split/cache errors when frequent)
        if 'Close' in df.columns:
            Q1 = df['Close'].quantile(0.25)
            Q3 = df['Close'].quantile(0.75)
            IQR = Q3 - Q1

            if IQR > 0:
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                outliers = ((df['Close'] < lower_bound) | (df['Close'] > upper_bound)).sum()

                if outliers > 0:
                    outlier_pct = (outliers / len(df)) * 100
                    if outlier_pct > 5:
                        errors.append(f"Close has {outliers} outliers ({outlier_pct:.1f}%)")

        # Check 3: Single-day extreme price changes (likely data errors)
        if 'Close' in df.columns:
            # Calculate daily percentage change
            daily_pct_change = df['Close'].pct_change(fill_method=None).abs()

            # Flag changes >50% in a single day as suspicious
            extreme_changes = (daily_pct_change > 0.5).sum()

            if extreme_changes > 0:
                extreme_pct = (extreme_changes / len(df)) * 100
                if extreme_pct > 1:  # Only report if >1% of days have extreme changes
                    errors.append(f"Extreme daily price changes: {extreme_changes} days with >50% change ({extreme_pct:.1f}%)")

        return errors

    def check_date_continuity(self, df: pd.DataFrame) -> bool:
        """
        Check for large gaps in date sequence

        Args:
            df (pd.DataFrame): DataFrame with DatetimeIndex

        Returns:
            bool: True if no large gaps (>10 trading days)
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            return True  # Cannot check, assume OK

        # Calculate differences between consecutive dates
        date_diffs = df.index.to_series().diff()

        # Trading days: expect max ~3-4 days for weekends/holidays
        # Flag if gap >10 days (likely data issue)
        max_gap = date_diffs.max()

        if pd.isna(max_gap):
            return True

        return max_gap.days <= 10

    def _check_ohlc_logic(self, df: pd.DataFrame) -> bool:
        """
        Check OHLC logical consistency (legacy method, kept for compatibility)

        Rules:
        - High >= Low
        - Close between Low and High
        - Open between Low and High

        Args:
            df (pd.DataFrame): DataFrame to check

        Returns:
            bool: True if all records pass logic check
        """
        if not all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
            return True  # Cannot check, assume OK

        # Check High >= Low
        high_low_check = (df['High'] >= df['Low']).all()

        # Check Close within [Low, High]
        close_check = ((df['Close'] >= df['Low']) & (df['Close'] <= df['High'])).all()

        # Check Open within [Low, High]
        open_check = ((df['Open'] >= df['Low']) & (df['Open'] <= df['High'])).all()

        return high_low_check and close_check and open_check

    def _check_ohlc_logic_detailed(self, df: pd.DataFrame) -> dict:
        """
        Check OHLC logical consistency with detailed reporting

        Rules:
        - High >= Low
        - Close between Low and High
        - Open between Low and High

        Args:
            df (pd.DataFrame): DataFrame to check

        Returns:
            dict: Details about OHLC issues, or None if no issues
                  {'count': int, 'percent': float, 'examples': list}
        """
        if not all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
            return None  # Cannot check

        # Find problematic rows
        high_low_issues = df['High'] < df['Low']
        close_issues = (df['Close'] < df['Low']) | (df['Close'] > df['High'])
        open_issues = (df['Open'] < df['Low']) | (df['Open'] > df['High'])

        all_issues = high_low_issues | close_issues | open_issues
        issue_count = all_issues.sum()

        if issue_count == 0:
            return None

        issue_pct = (issue_count / len(df)) * 100

        return {
            'count': issue_count,
            'percent': issue_pct,
            'total_rows': len(df)
        }

    def get_data_summary(self, df: pd.DataFrame) -> dict:
        """
        Get summary statistics for the data

        Args:
            df (pd.DataFrame): DataFrame to summarize

        Returns:
            dict: Summary statistics
        """
        summary = {
            'num_records': len(df),
            'date_range': None,
            'columns': list(df.columns),
            'missing_values': {},
            'price_range': {}
        }

        if isinstance(df.index, pd.DatetimeIndex):
            summary['date_range'] = {
                'start': df.index.min().strftime('%Y-%m-%d'),
                'end': df.index.max().strftime('%Y-%m-%d')
            }

        # Check missing values
        for col in df.columns:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                summary['missing_values'][col] = nan_count

        # Price statistics
        if 'Close' in df.columns:
            summary['price_range'] = {
                'min': float(df['Close'].min()),
                'max': float(df['Close'].max()),
                'mean': float(df['Close'].mean())
            }

        return summary
