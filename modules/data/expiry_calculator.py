"""
SPY Options Expiry Date Calculator

This module calculates SPY monthly options expiry dates (third Friday of each month).
Handles holiday adjustments and provides various utility methods.

Usage:
    from modules.data import ExpiryCalculator

    # Calculate third Friday of a specific month
    expiry = ExpiryCalculator.calculate_third_friday(2025, 11)

    # Generate expiry dates for a date range
    expiry_dates = ExpiryCalculator.generate_expiry_dates('2024-01-01', '2025-12-31')

    # Get next expiry date
    next_expiry = ExpiryCalculator.get_next_expiry()
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional
import calendar


class ExpiryCalculator:
    """
    Calculator for SPY monthly options expiry dates

    SPY monthly options expire on the third Friday of each month.
    If the third Friday is a market holiday, expiry is typically on Thursday.
    """

    # US market holidays (major holidays that close the market)
    # Note: This is a simplified list. For production, use trading calendar library
    KNOWN_HOLIDAYS = [
        '2024-01-01',  # New Year's Day
        '2024-01-15',  # MLK Day
        '2024-02-19',  # Presidents Day
        '2024-03-29',  # Good Friday
        '2024-05-27',  # Memorial Day
        '2024-06-19',  # Juneteenth
        '2024-07-04',  # Independence Day
        '2024-09-02',  # Labor Day
        '2024-11-28',  # Thanksgiving
        '2024-12-25',  # Christmas
        '2025-01-01',  # New Year's Day
        '2025-01-20',  # MLK Day
        '2025-02-17',  # Presidents Day
        '2025-04-18',  # Good Friday
        '2025-05-26',  # Memorial Day
        '2025-06-19',  # Juneteenth
        '2025-07-04',  # Independence Day
        '2025-09-01',  # Labor Day
        '2025-11-27',  # Thanksgiving
        '2025-12-25',  # Christmas
        '2026-01-01',  # New Year's Day
        '2026-01-19',  # MLK Day
        '2026-02-16',  # Presidents Day
        '2026-04-03',  # Good Friday
        '2026-05-25',  # Memorial Day
        '2026-06-19',  # Juneteenth
        '2026-07-03',  # Independence Day (observed)
        '2026-09-07',  # Labor Day
        '2026-11-26',  # Thanksgiving
        '2026-12-25',  # Christmas
    ]

    @staticmethod
    def calculate_third_friday(year: int, month: int) -> pd.Timestamp:
        """
        Calculate the third Friday of a given month

        Args:
            year (int): Year
            month (int): Month (1-12)

        Returns:
            pd.Timestamp: Third Friday of the month
        """
        # Get the first day of the month
        first_day = datetime(year, month, 1)

        # Find the first Friday
        # weekday(): Monday=0, Sunday=6, so Friday=4
        days_until_friday = (4 - first_day.weekday()) % 7
        first_friday = first_day + timedelta(days=days_until_friday)

        # Third Friday is 14 days after first Friday
        third_friday = first_friday + timedelta(days=14)

        # Check for holiday adjustment
        third_friday_ts = pd.Timestamp(third_friday)
        if ExpiryCalculator._is_holiday(third_friday_ts):
            # If Friday is a holiday, expiry is typically Thursday
            third_friday_ts = third_friday_ts - timedelta(days=1)

        return third_friday_ts

    @staticmethod
    def generate_expiry_dates(
        start_date: str,
        end_date: str,
        include_weeklies: bool = False
    ) -> List[pd.Timestamp]:
        """
        Generate all expiry dates within a date range

        Args:
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
            include_weeklies (bool): If True, include weekly expiries (default: False, monthly only)

        Returns:
            list: List of expiry dates as pd.Timestamp objects, sorted chronologically
        """
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        expiry_dates = []

        # Iterate through each month in the range
        current_date = start
        while current_date <= end:
            year = current_date.year
            month = current_date.month

            # Calculate third Friday
            third_friday = ExpiryCalculator.calculate_third_friday(year, month)

            # Only include if within date range
            if start <= third_friday <= end:
                expiry_dates.append(third_friday)

            # Move to next month
            if month == 12:
                current_date = datetime(year + 1, 1, 1)
            else:
                current_date = datetime(year, month + 1, 1)

        return sorted(expiry_dates)

    @staticmethod
    def get_next_expiry(reference_date: Optional[str] = None) -> pd.Timestamp:
        """
        Get the next expiry date after a reference date

        Args:
            reference_date (str, optional): Reference date in 'YYYY-MM-DD' format
                                           If None, uses today

        Returns:
            pd.Timestamp: Next expiry date
        """
        if reference_date is None:
            ref = pd.Timestamp.now()
        else:
            ref = pd.to_datetime(reference_date)

        # Start from current month
        year = ref.year
        month = ref.month

        # Check current month's expiry
        third_friday = ExpiryCalculator.calculate_third_friday(year, month)

        if third_friday > ref:
            return third_friday

        # Move to next month
        if month == 12:
            year += 1
            month = 1
        else:
            month += 1

        return ExpiryCalculator.calculate_third_friday(year, month)

    @staticmethod
    def get_previous_expiry(reference_date: Optional[str] = None) -> pd.Timestamp:
        """
        Get the previous expiry date before a reference date

        Args:
            reference_date (str, optional): Reference date in 'YYYY-MM-DD' format
                                           If None, uses today

        Returns:
            pd.Timestamp: Previous expiry date
        """
        if reference_date is None:
            ref = pd.Timestamp.now()
        else:
            ref = pd.to_datetime(reference_date)

        # Start from current month
        year = ref.year
        month = ref.month

        # Check current month's expiry
        third_friday = ExpiryCalculator.calculate_third_friday(year, month)

        if third_friday < ref:
            return third_friday

        # Move to previous month
        if month == 1:
            year -= 1
            month = 12
        else:
            month -= 1

        return ExpiryCalculator.calculate_third_friday(year, month)

    @staticmethod
    def is_expiry_date(date: str, tolerance_days: int = 0) -> bool:
        """
        Check if a given date is an expiry date

        Args:
            date (str): Date to check in 'YYYY-MM-DD' format
            tolerance_days (int): Number of days tolerance (for holiday adjustments)

        Returns:
            bool: True if the date is an expiry date
        """
        check_date = pd.to_datetime(date)
        year = check_date.year
        month = check_date.month

        # Calculate the expiry for this month
        expiry = ExpiryCalculator.calculate_third_friday(year, month)

        # Check if dates match (within tolerance)
        date_diff = abs((check_date - expiry).days)
        return date_diff <= tolerance_days

    @staticmethod
    def get_expiry_for_month(year: int, month: int) -> pd.Timestamp:
        """
        Get expiry date for a specific month

        Args:
            year (int): Year
            month (int): Month (1-12)

        Returns:
            pd.Timestamp: Expiry date
        """
        return ExpiryCalculator.calculate_third_friday(year, month)

    @staticmethod
    def days_to_expiry(date: str, expiry_date: str) -> int:
        """
        Calculate number of days between a date and expiry

        Args:
            date (str): Current date in 'YYYY-MM-DD' format
            expiry_date (str): Expiry date in 'YYYY-MM-DD' format

        Returns:
            int: Number of days to expiry (negative if past expiry)
        """
        current = pd.to_datetime(date)
        expiry = pd.to_datetime(expiry_date)

        return (expiry - current).days

    @staticmethod
    def _is_holiday(date: pd.Timestamp) -> bool:
        """
        Check if a date is a market holiday

        Args:
            date (pd.Timestamp): Date to check

        Returns:
            bool: True if the date is a known holiday
        """
        date_str = date.strftime('%Y-%m-%d')
        return date_str in ExpiryCalculator.KNOWN_HOLIDAYS

    @staticmethod
    def get_expiry_info(expiry_date: str) -> dict:
        """
        Get detailed information about an expiry date

        Args:
            expiry_date (str): Expiry date in 'YYYY-MM-DD' format

        Returns:
            dict: Information about the expiry date
        """
        expiry = pd.to_datetime(expiry_date)

        info = {
            'date': expiry.strftime('%Y-%m-%d'),
            'year': expiry.year,
            'month': expiry.month,
            'month_name': expiry.strftime('%B'),
            'day_of_week': expiry.strftime('%A'),
            'is_third_friday': ExpiryCalculator.is_expiry_date(expiry_date),
            'days_from_today': ExpiryCalculator.days_to_expiry(
                pd.Timestamp.now().strftime('%Y-%m-%d'),
                expiry_date
            )
        }

        return info

    # =========================================================================
    # Weekly Expiry Methods
    # =========================================================================

    @staticmethod
    def get_weekly_expiry_dates(
        start_date: str,
        end_date: str,
        day_of_week: str = 'friday'
    ) -> List[pd.Timestamp]:
        """
        Get weekly expiry dates within a date range

        SPY weekly options expire on Monday, Wednesday, and Friday.
        This method returns all expiry dates for the specified day of week.

        Args:
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
            day_of_week (str): 'monday', 'wednesday', 'friday', or 'all'

        Returns:
            List[pd.Timestamp]: List of weekly expiry dates
        """
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        # Map day names to weekday numbers (Monday=0, Sunday=6)
        day_map = {
            'monday': 0,
            'wednesday': 2,
            'friday': 4
        }

        if day_of_week.lower() == 'all':
            target_days = [0, 2, 4]  # Mon, Wed, Fri
        elif day_of_week.lower() in day_map:
            target_days = [day_map[day_of_week.lower()]]
        else:
            raise ValueError(
                f"Invalid day_of_week: {day_of_week}. "
                "Must be 'monday', 'wednesday', 'friday', or 'all'"
            )

        expiry_dates = []

        # Generate all dates in range
        current = start
        while current <= end:
            if current.weekday() in target_days:
                # Check if it's a holiday
                if ExpiryCalculator._is_holiday(current):
                    # Adjust to previous trading day
                    adjusted = current - timedelta(days=1)
                    while ExpiryCalculator._is_holiday(adjusted) or adjusted.weekday() >= 5:
                        adjusted -= timedelta(days=1)
                    expiry_dates.append(adjusted)
                else:
                    expiry_dates.append(current)
            current += timedelta(days=1)

        return sorted(list(set(expiry_dates)))  # Remove duplicates and sort

    @staticmethod
    def get_next_weekly_expiry(
        reference_date: Optional[str] = None,
        day_of_week: str = 'friday'
    ) -> pd.Timestamp:
        """
        Get the next weekly expiry date after a reference date

        Args:
            reference_date (str, optional): Reference date. If None, uses today.
            day_of_week (str): 'monday', 'wednesday', 'friday', or 'all'

        Returns:
            pd.Timestamp: Next weekly expiry date
        """
        if reference_date is None:
            ref = pd.Timestamp.now().normalize()
        else:
            ref = pd.to_datetime(reference_date)

        # Look ahead up to 14 days to find next expiry
        end_date = ref + timedelta(days=14)

        expiries = ExpiryCalculator.get_weekly_expiry_dates(
            ref.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d'),
            day_of_week
        )

        # Find first expiry after reference date
        for expiry in expiries:
            if expiry > ref:
                return expiry

        # If none found, extend search
        end_date = ref + timedelta(days=30)
        expiries = ExpiryCalculator.get_weekly_expiry_dates(
            ref.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d'),
            day_of_week
        )

        for expiry in expiries:
            if expiry > ref:
                return expiry

        raise ValueError("Could not find next weekly expiry")

    @staticmethod
    def is_monthly_opex_week(date: str) -> bool:
        """
        Check if a date falls in the week of monthly OpEx (third Friday)

        Args:
            date (str): Date to check in 'YYYY-MM-DD' format

        Returns:
            bool: True if date is in the monthly OpEx week
        """
        check_date = pd.to_datetime(date)
        year = check_date.year
        month = check_date.month

        # Get the third Friday of the month
        third_friday = ExpiryCalculator.calculate_third_friday(year, month)

        # Calculate the Monday of the OpEx week
        days_since_monday = third_friday.weekday()
        opex_week_start = third_friday - timedelta(days=days_since_monday)
        opex_week_end = opex_week_start + timedelta(days=6)

        return opex_week_start <= check_date <= opex_week_end

    @staticmethod
    def get_expiry_type(date: str) -> str:
        """
        Determine the type of expiry for a given date

        Args:
            date (str): Date to check in 'YYYY-MM-DD' format

        Returns:
            str: 'monthly' if third Friday, 'weekly' if other expiry day, 'none' otherwise
        """
        check_date = pd.to_datetime(date)

        # Check if it's a monthly expiry (third Friday)
        if ExpiryCalculator.is_expiry_date(date, tolerance_days=1):
            return 'monthly'

        # Check if it's a weekly expiry day (Mon, Wed, Fri)
        if check_date.weekday() in [0, 2, 4]:
            if not ExpiryCalculator._is_holiday(check_date):
                return 'weekly'

        return 'none'

    @staticmethod
    def get_previous_trading_day(
        date: str,
        lookback: int = 1
    ) -> pd.Timestamp:
        """
        Get the previous trading day(s) before a given date

        Args:
            date (str): Reference date in 'YYYY-MM-DD' format
            lookback (int): Number of trading days to look back

        Returns:
            pd.Timestamp: Previous trading day
        """
        current = pd.to_datetime(date) - timedelta(days=1)
        count = 0

        while count < lookback:
            # Skip weekends
            while current.weekday() >= 5:
                current -= timedelta(days=1)

            # Skip holidays
            while ExpiryCalculator._is_holiday(current):
                current -= timedelta(days=1)
                # Check weekend again after holiday adjustment
                while current.weekday() >= 5:
                    current -= timedelta(days=1)

            count += 1
            if count < lookback:
                current -= timedelta(days=1)

        return current
