"""
Unit tests for ExpiryCalculator

Tests the SPY options expiry date calculation logic.
"""

import unittest
import pandas as pd
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from modules.data import ExpiryCalculator


class TestExpiryCalculator(unittest.TestCase):
    """Test cases for ExpiryCalculator"""

    def test_calculate_third_friday_known_dates(self):
        """Test calculation with known third Friday dates"""
        # Test cases with known third Fridays
        test_cases = [
            (2025, 11, '2025-11-21'),  # November 2025
            (2025, 12, '2025-12-19'),  # December 2025
            (2026, 1, '2026-01-16'),   # January 2026
            (2024, 6, '2024-06-21'),   # June 2024
        ]

        for year, month, expected_date in test_cases:
            with self.subTest(year=year, month=month):
                result = ExpiryCalculator.calculate_third_friday(year, month)
                expected = pd.Timestamp(expected_date)
                self.assertEqual(result.date(), expected.date(),
                               f"Failed for {year}-{month}")

    def test_generate_expiry_dates(self):
        """Test generating multiple expiry dates in a range"""
        start_date = '2025-01-01'
        end_date = '2025-12-31'

        expiry_dates = ExpiryCalculator.generate_expiry_dates(start_date, end_date)

        # Should have 12 months
        self.assertEqual(len(expiry_dates), 12)

        # Most should be Fridays (weekday 4), but some may be Thursday due to holiday adjustments
        # For example, April 2025: Good Friday (April 18) → expiry moves to Thursday April 17
        fridays = sum(1 for expiry in expiry_dates if expiry.weekday() == 4)
        thursdays = sum(1 for expiry in expiry_dates if expiry.weekday() == 3)

        # Most should be Fridays, a few might be Thursdays
        self.assertGreaterEqual(fridays, 10, "Should have at least 10 Fridays")
        self.assertLessEqual(thursdays, 2, "Should have at most 2 Thursday adjustments")

        # Check first and last
        self.assertEqual(expiry_dates[0].strftime('%Y-%m-%d'), '2025-01-17')
        self.assertEqual(expiry_dates[-1].strftime('%Y-%m-%d'), '2025-12-19')

    def test_get_next_expiry(self):
        """Test getting next expiry date"""
        # Test with a specific reference date
        reference = '2025-11-01'
        next_expiry = ExpiryCalculator.get_next_expiry(reference)

        # Next expiry after Nov 1 should be Nov 21
        self.assertEqual(next_expiry.strftime('%Y-%m-%d'), '2025-11-21')

        # Test when reference is after current month's expiry
        reference = '2025-11-25'
        next_expiry = ExpiryCalculator.get_next_expiry(reference)

        # Next expiry after Nov 25 should be Dec 19
        self.assertEqual(next_expiry.strftime('%Y-%m-%d'), '2025-12-19')

    def test_get_previous_expiry(self):
        """Test getting previous expiry date"""
        reference = '2025-11-25'
        prev_expiry = ExpiryCalculator.get_previous_expiry(reference)

        # Previous expiry before Nov 25 should be Nov 21
        self.assertEqual(prev_expiry.strftime('%Y-%m-%d'), '2025-11-21')

        # Test when reference is before current month's expiry
        reference = '2025-11-10'
        prev_expiry = ExpiryCalculator.get_previous_expiry(reference)

        # Previous expiry before Nov 10 should be Oct 17
        self.assertEqual(prev_expiry.strftime('%Y-%m-%d'), '2025-10-17')

    def test_is_expiry_date(self):
        """Test checking if a date is an expiry date"""
        # Known expiry date
        self.assertTrue(ExpiryCalculator.is_expiry_date('2025-11-21'))

        # Not an expiry date
        self.assertFalse(ExpiryCalculator.is_expiry_date('2025-11-20'))

        # With tolerance
        self.assertTrue(ExpiryCalculator.is_expiry_date('2025-11-20', tolerance_days=1))

    def test_days_to_expiry(self):
        """Test calculating days to expiry"""
        current = '2025-11-15'
        expiry = '2025-11-21'

        days = ExpiryCalculator.days_to_expiry(current, expiry)
        self.assertEqual(days, 6)

        # Negative days (past expiry)
        days = ExpiryCalculator.days_to_expiry('2025-11-25', '2025-11-21')
        self.assertEqual(days, -4)

    def test_get_expiry_for_month(self):
        """Test getting expiry for a specific month"""
        expiry = ExpiryCalculator.get_expiry_for_month(2025, 11)
        self.assertEqual(expiry.strftime('%Y-%m-%d'), '2025-11-21')

    def test_get_expiry_info(self):
        """Test getting detailed expiry information"""
        info = ExpiryCalculator.get_expiry_info('2025-11-21')

        self.assertEqual(info['year'], 2025)
        self.assertEqual(info['month'], 11)
        self.assertEqual(info['month_name'], 'November')
        self.assertEqual(info['day_of_week'], 'Friday')
        self.assertTrue(info['is_third_friday'])

    def test_year_boundary(self):
        """Test expiry calculation across year boundaries"""
        # December to January
        dec_expiry = ExpiryCalculator.calculate_third_friday(2025, 12)
        jan_expiry = ExpiryCalculator.calculate_third_friday(2026, 1)

        self.assertEqual(dec_expiry.year, 2025)
        self.assertEqual(jan_expiry.year, 2026)

        # Next expiry from late December should be January
        next_expiry = ExpiryCalculator.get_next_expiry('2025-12-25')
        self.assertEqual(next_expiry.year, 2026)
        self.assertEqual(next_expiry.month, 1)


if __name__ == '__main__':
    unittest.main()
