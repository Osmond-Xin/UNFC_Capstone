"""
Data Loader for S&P 500 Stock Data

This module provides unified interface to load stock data from cache.
It handles single ticker loading, batch loading, and S&P 500 list management.

Features:
- Auto-update: Automatically detects if cache is outdated and updates via Alpaca API
- Date filtering: Load data for specific date ranges
- Batch loading: Efficiently load multiple tickers

Usage:
    from modules.data import DataLoader

    loader = DataLoader()

    # Load single ticker (auto-updates cache if needed)
    df = loader.load_ticker('AAPL')

    # Load multiple tickers
    data_dict = loader.load_sp500_batch(['AAPL', 'GOOGL', 'MSFT'])

    # Disable auto-update
    loader = DataLoader(auto_update=False)

    # Get S&P 500 ticker list
    tickers = loader.get_sp500_tickers()
"""

import os
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional


class DataLoader:
    """
    Unified data loader for S&P 500 stock data from cache

    Features:
    - Auto-update: Checks cache freshness and updates via Alpaca if needed
    - Respects market holidays (only updates on trading days)
    - Rate-limited Alpaca API calls
    """

    def __init__(self, cache_dir='cache/constituent_data', auto_update=False):
        """
        Initialize the data loader

        Args:
            cache_dir (str): Path to cache directory
            auto_update (bool): If True, automatically update cache when outdated
        """
        self.cache_dir = cache_dir
        self.auto_update = auto_update
        self._update_checked = False  # Only check once per session

    def _check_and_update_cache(self):
        """
        Check if cache needs updating and trigger incremental update if needed.
        Only runs once per DataLoader session.
        """
        if self._update_checked or not self.auto_update:
            return

        self._update_checked = True

        try:
            # Get cache last date
            cache_last_date = self._get_cache_last_date()

            if cache_last_date is None:
                print("⚠️  No cache found, please run CacheManager.update_cache() first")
                return

            # Get expected last trading day
            expected_date = self._get_last_trading_day()

            cache_date = datetime.strptime(cache_last_date, '%Y-%m-%d').date()

            if cache_date >= expected_date:
                # Cache is up to date
                return

            # Cache is outdated, trigger incremental update
            days_behind = (expected_date - cache_date).days
            print(f"\n📅 Cache Status:")
            print(f"   Cache last date: {cache_last_date}")
            print(f"   Expected date: {expected_date}")
            print(f"   Days behind: {days_behind}")

            # Import CacheManager here to avoid circular import
            from .cache_manager import CacheManager

            print(f"\n🔄 Auto-updating cache via Alpaca API...")
            manager = CacheManager()
            manager.incremental_update()

        except Exception as e:
            print(f"⚠️  Auto-update check failed: {e}")
            # Don't block data loading if update fails

    def _get_cache_last_date(self) -> Optional[str]:
        """
        Get the latest date in cache by sampling a few files

        Returns:
            str: Latest date in YYYY-MM-DD format, or None
        """
        if not os.path.exists(self.cache_dir):
            return None

        cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.csv')]
        if not cache_files:
            return None

        latest_date = None

        # Sample first 5 files
        for filename in cache_files[:5]:
            try:
                filepath = os.path.join(self.cache_dir, filename)
                df = pd.read_csv(filepath, usecols=['Date'])
                if df.empty:
                    continue
                df['Date'] = pd.to_datetime(df['Date'])
                file_max = df['Date'].max()
                if latest_date is None or file_max > latest_date:
                    latest_date = file_max
            except Exception:
                continue

        if latest_date:
            return latest_date.strftime('%Y-%m-%d')
        return None

    def _get_last_trading_day(self) -> datetime.date:
        """
        Get the expected last trading day (yesterday if weekday, last Friday if weekend)

        Returns:
            date: Expected last trading day
        """
        today = datetime.now().date()

        # If before market open (9:30 AM ET), use previous day
        # Simplified: just use yesterday
        yesterday = today - timedelta(days=1)

        # Adjust for weekends
        weekday = yesterday.weekday()
        if weekday == 5:  # Saturday -> Friday
            return yesterday - timedelta(days=1)
        elif weekday == 6:  # Sunday -> Friday
            return yesterday - timedelta(days=2)

        return yesterday

    def load_ticker(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        validate: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Load data for a single ticker from cache

        Args:
            ticker (str): Stock ticker symbol
            start_date (str, optional): Start date in 'YYYY-MM-DD' format
            end_date (str, optional): End date in 'YYYY-MM-DD' format
            validate (bool): If True, validate data quality (default: True)

        Returns:
            pd.DataFrame: Stock data with DatetimeIndex and OHLCV columns
            None: If ticker not found or data is invalid
        """
        # Check and update cache if needed (only once per session)
        self._check_and_update_cache()

        cache_file = os.path.join(self.cache_dir, f"{ticker}.csv")

        if not os.path.exists(cache_file):
            print(f"⚠ Ticker {ticker} not found in cache")
            return None

        try:
            # Read CSV
            df = pd.read_csv(cache_file)

            # Convert Date to datetime and set as index (tz-naive — CSVs store plain dates)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')

            # Filter date range if specified
            if start_date is not None:
                df = df[df.index >= pd.to_datetime(start_date)]
            if end_date is not None:
                df = df[df.index <= pd.to_datetime(end_date)]

            # Validate data if requested
            if validate:
                from .data_validator import DataValidator
                validator = DataValidator()
                is_valid, errors = validator.validate(df, ticker)
                if not is_valid:
                    print(f"⚠ Data validation failed for {ticker}:")
                    for error in errors:
                        print(f"  - {error}")
                    return None

            return df

        except Exception as e:
            print(f"✗ Error loading {ticker}: {e}")
            return None

    def load_sp500_batch(
        self,
        tickers: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        validate: bool = True,
        skip_missing: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Load multiple tickers in batch

        Args:
            tickers (list): List of ticker symbols
            start_date (str, optional): Start date in 'YYYY-MM-DD' format
            end_date (str, optional): End date in 'YYYY-MM-DD' format
            validate (bool): If True, validate data quality
            skip_missing (bool): If True, skip missing tickers instead of raising error

        Returns:
            dict: Dictionary mapping ticker -> DataFrame
        """
        # Check and update cache if needed (only once per session)
        self._check_and_update_cache()

        data = {}
        failed = []

        print(f"Loading {len(tickers)} tickers...")

        for ticker in tickers:
            df = self.load_ticker(ticker, start_date, end_date, validate)
            if df is not None:
                data[ticker] = df
            else:
                failed.append(ticker)
                if not skip_missing:
                    raise ValueError(f"Failed to load ticker: {ticker}")

        print(f"✓ Loaded {len(data)} tickers successfully")
        if failed:
            print(f"⚠ Failed to load {len(failed)} tickers: {failed[:5]}...")

        return data

    def load_all_sp500(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        validate: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Load all S&P 500 stocks from cache

        Args:
            start_date (str, optional): Start date in 'YYYY-MM-DD' format
            end_date (str, optional): End date in 'YYYY-MM-DD' format
            validate (bool): If True, validate data quality

        Returns:
            dict: Dictionary mapping ticker -> DataFrame
        """
        tickers = self.get_sp500_tickers()
        return self.load_sp500_batch(tickers, start_date, end_date, validate)

    def get_sp500_tickers(self) -> List[str]:
        """
        Get list of all S&P 500 ticker symbols available in cache

        Returns:
            list: List of ticker symbols
        """
        if not os.path.exists(self.cache_dir):
            print(f"⚠ Cache directory not found: {self.cache_dir}")
            return []

        try:
            cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.csv')]
            tickers = [f.replace('.csv', '') for f in cache_files]
            tickers.sort()
            return tickers

        except Exception as e:
            print(f"✗ Error reading cache directory: {e}")
            return []

    def get_ticker_info(self, ticker: str) -> Optional[Dict]:
        """
        Get metadata about a ticker's cached data

        Args:
            ticker (str): Stock ticker symbol

        Returns:
            dict: Metadata including date range, number of records, etc.
            None: If ticker not found
        """
        cache_file = os.path.join(self.cache_dir, f"{ticker}.csv")

        if not os.path.exists(cache_file):
            return None

        try:
            df = pd.read_csv(cache_file)
            df['Date'] = pd.to_datetime(df['Date'], utc=True)

            info = {
                'ticker': ticker,
                'num_records': len(df),
                'start_date': df['Date'].min().strftime('%Y-%m-%d'),
                'end_date': df['Date'].max().strftime('%Y-%m-%d'),
                'file_path': cache_file,
                'file_size_kb': os.path.getsize(cache_file) / 1024
            }

            return info

        except Exception as e:
            print(f"✗ Error getting info for {ticker}: {e}")
            return None
