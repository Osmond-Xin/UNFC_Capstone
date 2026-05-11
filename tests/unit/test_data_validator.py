"""
Unit tests for DataValidator

Tests data quality validation logic.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from modules.data import DataValidator


class TestDataValidator(unittest.TestCase):
    """Test cases for DataValidator"""

    def setUp(self):
        """Create test data"""
        self.validator = DataValidator(min_records=30)

        # Create valid test data
        dates = pd.date_range('2025-01-01', periods=100, freq='D')
        self.valid_df = pd.DataFrame({
            'Open': np.random.uniform(100, 110, 100),
            'High': np.random.uniform(110, 120, 100),
            'Low': np.random.uniform(90, 100, 100),
            'Close': np.random.uniform(95, 115, 100),
            'Volume': np.random.randint(1000000, 10000000, 100)
        }, index=dates)

        # Ensure OHLC logic is correct
        for i in range(len(self.valid_df)):
            low = self.valid_df.iloc[i]['Low']
            high = self.valid_df.iloc[i]['High']
            self.valid_df.iloc[i, self.valid_df.columns.get_loc('High')] = max(high, low + 5)
            self.valid_df.iloc[i, self.valid_df.columns.get_loc('Close')] = np.random.uniform(low, high)
            self.valid_df.iloc[i, self.valid_df.columns.get_loc('Open')] = np.random.uniform(low, high)

    def test_check_completeness_valid(self):
        """Test completeness check with valid data"""
        self.assertTrue(self.validator.check_completeness(self.valid_df))

    def test_check_completeness_missing_column(self):
        """Test completeness check with missing column"""
        df = self.valid_df.drop(columns=['Volume'])
        self.assertFalse(self.validator.check_completeness(df))

    def test_validate_valid_data(self):
        """Test validation with completely valid data"""
        is_valid, errors = self.validator.validate(self.valid_df, 'TEST')
        self.assertTrue(is_valid, f"Validation failed with errors: {errors}")
        self.assertEqual(len(errors), 0)

    def test_validate_insufficient_data(self):
        """Test validation with insufficient data points"""
        small_df = self.valid_df.head(10)  # Only 10 records
        is_valid, errors = self.validator.validate(small_df, 'TEST')

        self.assertFalse(is_valid)
        self.assertTrue(any('Insufficient data' in err for err in errors))

    def test_validate_empty_dataframe(self):
        """Test validation with empty DataFrame"""
        empty_df = pd.DataFrame()
        is_valid, errors = self.validator.validate(empty_df, 'TEST')

        self.assertFalse(is_valid)
        self.assertTrue(any('empty' in err.lower() for err in errors))

    def test_validate_with_nan_values(self):
        """Test validation with NaN values"""
        df_with_nan = self.valid_df.copy()
        df_with_nan.loc[df_with_nan.index[5], 'Close'] = np.nan

        is_valid, errors = self.validator.validate(df_with_nan, 'TEST')

        self.assertFalse(is_valid)
        self.assertTrue(any('NaN' in err for err in errors))

    def test_check_date_continuity(self):
        """Test date continuity check"""
        # Valid continuous dates
        self.assertTrue(self.validator.check_date_continuity(self.valid_df))

        # Create a large gap (50 days gap from Jan 31 to March 1)
        dates_part1 = pd.date_range('2025-01-01', periods=31, freq='D')
        dates_part2 = pd.date_range('2025-03-21', periods=30, freq='D')  # 49 day gap
        dates_with_gap = dates_part1.tolist() + dates_part2.tolist()

        df_with_gap = pd.DataFrame({
            'Open': np.random.uniform(100, 110, len(dates_with_gap)),
            'High': np.random.uniform(110, 120, len(dates_with_gap)),
            'Low': np.random.uniform(90, 100, len(dates_with_gap)),
            'Close': np.random.uniform(95, 115, len(dates_with_gap)),
            'Volume': np.random.randint(1000000, 10000000, len(dates_with_gap))
        }, index=pd.DatetimeIndex(dates_with_gap))

        # Sort index to ensure proper ordering
        df_with_gap = df_with_gap.sort_index()

        # Should detect the gap (49 days > 10 day threshold)
        self.assertFalse(self.validator.check_date_continuity(df_with_gap))

    def test_ohlc_logic(self):
        """Test OHLC logical consistency checks"""
        # Valid data should pass
        is_valid, _ = self.validator.validate(self.valid_df, 'TEST')
        self.assertTrue(is_valid)

        # Create invalid OHLC data (High < Low)
        invalid_df = self.valid_df.copy()
        invalid_df.iloc[0, invalid_df.columns.get_loc('High')] = 50
        invalid_df.iloc[0, invalid_df.columns.get_loc('Low')] = 100

        is_valid, errors = self.validator.validate(invalid_df, 'TEST')
        self.assertFalse(is_valid)
        self.assertTrue(any('OHLC logic' in err for err in errors))

    def test_get_data_summary(self):
        """Test data summary generation"""
        summary = self.validator.get_data_summary(self.valid_df)

        self.assertEqual(summary['num_records'], 100)
        self.assertIn('columns', summary)
        self.assertIn('price_range', summary)
        self.assertIn('date_range', summary)

        # Check price range
        self.assertIn('min', summary['price_range'])
        self.assertIn('max', summary['price_range'])
        self.assertIn('mean', summary['price_range'])

    def test_check_outliers(self):
        """Test outlier detection"""
        # Valid data should have no outliers
        errors = self.validator.check_outliers(self.valid_df)
        self.assertEqual(len(errors), 0)

        # Add extreme outliers
        df_with_outliers = self.valid_df.copy()
        for i in range(10):  # Add 10 outliers (10% of data)
            df_with_outliers.iloc[i, df_with_outliers.columns.get_loc('Close')] = 10000  # Extreme value

        errors = self.validator.check_outliers(df_with_outliers)
        # Should detect outliers
        self.assertGreater(len(errors), 0)


if __name__ == '__main__':
    unittest.main()
