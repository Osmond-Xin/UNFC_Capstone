"""
Smart Cache Manager for S&P 500 Stock Data - Stooq + Alpaca Version

This module provides intelligent cache management for S&P 500 stock data:
- Full download: Downloads from Stooq.com (free, no API key required)
- Incremental update: Uses Alpaca API to append new daily data

Features:
- Full download from Stooq.com with resume capability
- Incremental updates via Alpaca API (free tier supports stocks)
- Automatic detection of which mode to use
- Configuration via .env file for Alpaca credentials

Usage:
    # As a standalone script
    python modules/data/cache_manager.py

    # Or import and use programmatically
    from modules.data import CacheManager

    # Auto update (smart detection)
    manager = CacheManager()
    manager.update_cache()

    # Force full download
    manager.update_cache(force_full=True)
"""

import os
import sys
import time
import requests
import pandas as pd
import io
from datetime import datetime, timedelta
from typing import List, Optional, Dict
from dotenv import load_dotenv

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo


class CacheManager:
    """
    Cache manager using Stooq (full download) + Alpaca (incremental update)

    Configuration:
    - Stooq: Free, no API key, with polite rate limiting (1s delay)
    - Alpaca: Requires API key from .env file
    - Stock list source: cache/sp500_list.csv
    """

    # Configuration constants
    CACHE_DIR = 'cache/constituent_data'
    SP500_LIST_FILE = 'cache/sp500_list.csv'
    STOOQ_BASE_URL = "https://stooq.com/q/d/l/"
    STOOQ_REQUEST_DELAY = 1.0  # seconds between requests

    # Alpaca API rate limits: 200 requests/minute (free tier)
    # Safe delay: 60s / 200 = 0.3s, use 0.5s for safety margin
    ALPACA_REQUEST_DELAY = 0.5  # seconds between requests
    ALPACA_BATCH_SIZE = 50      # requests per batch
    ALPACA_BATCH_DELAY = 5      # extra delay after each batch

    # US market timezone and close time
    US_EASTERN_TZ = 'America/New_York'
    MARKET_CLOSE_HOUR = 16  # 4:00 PM ET

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the cache manager

        Args:
            cache_dir (str, optional): Custom cache directory path
        """
        # Load environment variables
        load_dotenv()

        # Determine cache directory
        if cache_dir:
            self.cache_dir = cache_dir
        else:
            self.cache_dir = self.CACHE_DIR

        # Always set sp500_list_file to default path
        self.sp500_list_file = self.SP500_LIST_FILE

        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)

        # Load Alpaca credentials
        self.alpaca_api_key = os.getenv('ALPACA_API_KEY')
        self.alpaca_secret_key = os.getenv('ALPACA_SECRET_KEY')
        # Use Data API endpoint for historical data (not trading endpoint)
        self.alpaca_data_endpoint = 'https://data.alpaca.markets'

        # Statistics
        self.stats = {
            'downloaded': 0,
            'failed': 0,
            'skipped': 0,
            'total': 0
        }

    def update_cache(self, force_full: bool = False) -> bool:
        """
        Main update function with automatic detection

        Args:
            force_full (bool): Force full download even if cache exists

        Returns:
            bool: True if update successful, False otherwise
        """
        print("=" * 70)
        print("CACHE UPDATE MANAGER - Stooq + Alpaca")
        print("=" * 70)

        # Check if full download is needed
        needs_full = self._needs_full_download()

        if force_full:
            print("\n🔄 Force full download requested")
            return self.full_download()
        elif needs_full:
            print("\n🔄 Full download required (using Stooq.com)")
            return self.full_download()
        else:
            print("\n📥 Incremental update mode (using Alpaca API)")
            return self.incremental_update()

    def _needs_full_download(self) -> bool:
        """
        Determine if full download is needed

        Returns:
            bool: True if full download needed, False otherwise
        """
        # Check if cache directory is empty
        if not os.path.exists(self.cache_dir):
            print("   ⚠️  Cache directory does not exist")
            return True

        cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.csv')]

        if len(cache_files) == 0:
            print("   ⚠️  Cache is empty")
            return True

        # Check if stock list count matches cache count
        tickers = self._load_sp500_list()

        if len(cache_files) < len(tickers) * 0.9:  # Allow 10% missing
            print(f"   ⚠️  Cache incomplete: {len(cache_files)} files vs {len(tickers)} tickers")
            return True

        print(f"   ✓ Cache exists: {len(cache_files)} files")
        return False

    def _load_sp500_list(self) -> List[str]:
        """
        Load S&P 500 ticker list from cache/sp500_list.csv

        Returns:
            list: List of ticker symbols
        """
        if not os.path.exists(self.sp500_list_file):
            print(f"❌ Error: S&P 500 list file not found: {self.sp500_list_file}")
            sys.exit(1)

        try:
            df = pd.read_csv(self.sp500_list_file)

            # Extract Symbol column
            if 'Symbol' not in df.columns:
                print(f"❌ Error: 'Symbol' column not found in {self.sp500_list_file}")
                sys.exit(1)

            tickers = df['Symbol'].dropna().tolist()

            # Remove empty strings
            tickers = [t.strip() for t in tickers if t.strip()]

            print(f"📊 Loaded {len(tickers)} tickers from {self.sp500_list_file}")

            return tickers

        except Exception as e:
            print(f"❌ Error loading S&P 500 list: {e}")
            sys.exit(1)

    def full_download(self) -> bool:
        """
        Download all historical data from Stooq.com

        Returns:
            bool: True if successful, False otherwise
        """
        print("\n" + "=" * 70)
        print("FULL DOWNLOAD MODE - Stooq.com")
        print("=" * 70)

        # Load ticker list
        tickers = self._load_sp500_list()
        self.stats['total'] = len(tickers)

        print(f"\n📋 Configuration:")
        print(f"   Data source: Stooq.com (free, no API key)")
        print(f"   Request delay: {self.STOOQ_REQUEST_DELAY}s")
        print(f"   Total stocks: {len(tickers)}")

        # Estimate time
        estimated_time = len(tickers) * self.STOOQ_REQUEST_DELAY
        print(f"   Estimated time: {int(estimated_time // 60)} minutes {int(estimated_time % 60)} seconds")

        print(f"\n📥 Starting download...\n")

        # Download each ticker
        start_time = time.time()

        for idx, ticker in enumerate(tickers):
            progress = (idx + 1) / len(tickers) * 100
            print(f"[{idx + 1}/{len(tickers)}] ({progress:.1f}%) {ticker:<6} ", end="")
            sys.stdout.flush()

            success = self._download_ticker_stooq(ticker)

            if success == 'skipped':
                self.stats['skipped'] += 1
                print("✓ (cached)")
            elif success:
                self.stats['downloaded'] += 1
                print("✓")
            else:
                self.stats['failed'] += 1
                print("✗")

            # Rate limiting
            if idx < len(tickers) - 1:
                time.sleep(self.STOOQ_REQUEST_DELAY)

        # Print summary
        elapsed_time = time.time() - start_time
        self._print_summary(elapsed_time)

        return self.stats['failed'] == 0

    def incremental_update(self, tickers: Optional[List[str]] = None) -> bool:
        """
        Update cache with new data using Alpaca API (primary) or Stooq (fallback).

        Each ticker is updated independently based on its own last cached date.

        Args:
            tickers (list, optional): List of tickers to update.
                If None, updates all S&P 500 tickers.

        Returns:
            bool: True if successful, False otherwise
        """
        print("\n" + "=" * 70)
        print("INCREMENTAL UPDATE MODE - Alpaca API")
        print("=" * 70)

        # Load ticker list if not provided
        if tickers is None:
            tickers = self._load_sp500_list()
        else:
            print(f"📊 Updating {len(tickers)} specified tickers")

        self.stats['total'] = len(tickers)

        # Get the latest date in cache (for display only)
        global_last_date = self._get_cache_last_date()

        if global_last_date is None:
            print("\n⚠️  No valid cache found, switching to full download")
            return self.full_download()

        latest_available = self._get_latest_available_date()

        eastern_tz = ZoneInfo(self.US_EASTERN_TZ)
        now_eastern = datetime.now(eastern_tz)
        market_closed = now_eastern.hour >= self.MARKET_CLOSE_HOUR

        print(f"\n📋 Update Configuration:")
        print(f"   Global cache reference date: {global_last_date}")
        print(f"   Current US Eastern time: {now_eastern.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"   Market closed: {'Yes' if market_closed else 'No (closes at 4:00 PM ET)'}")
        print(f"   Update end date: {latest_available}")
        print(f"   Note: Each ticker updated based on its own last date")
        print(f"   Stocks to check: {len(tickers)}")
        print(f"\n📥 Starting incremental update...\n")

        # Update each ticker
        start_time = time.time()

        for idx, ticker in enumerate(tickers):
            progress = (idx + 1) / len(tickers) * 100
            print(f"[{idx + 1}/{len(tickers)}] ({progress:.1f}%) {ticker:<6} ", end="")
            sys.stdout.flush()

            success = self._download_ticker_alpaca_incremental(ticker)

            if success:
                self.stats['downloaded'] += 1
                print("✓")
            elif success is None:
                self.stats['skipped'] += 1
                print("⊘ (up to date)")
            else:
                self.stats['failed'] += 1
                print("✗")

            # Small delay to avoid rate limiting
            if idx < len(tickers) - 1:
                time.sleep(0.1)

        # Print summary
        elapsed_time = time.time() - start_time
        self._print_summary(elapsed_time)

        return self.stats['failed'] == 0

    def _convert_ticker_to_stooq(self, ticker: str) -> str:
        """
        Convert standard ticker to Stooq format (e.g. BRK-B -> BRK.B.US)

        Args:
            ticker (str): Standard ticker symbol

        Returns:
            str: Stooq-formatted ticker
        """
        # Replace dash with dot
        stooq_ticker = ticker.replace('-', '.')
        # Append .US suffix
        return f"{stooq_ticker}.US"

    def _download_ticker_stooq(self, ticker: str, fallback_to_alpaca: bool = True):
        """
        Download full historical data for a single ticker from Stooq.com

        Args:
            ticker (str): Stock ticker symbol
            fallback_to_alpaca (bool): If True, try Alpaca API when Stooq fails

        Returns:
            bool or str: True if downloaded, 'skipped' if cached, False if failed
        """
        # Check if file already exists (resume capability)
        output_file = os.path.join(self.cache_dir, f"{ticker}.csv")
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            try:
                # Verify it's a valid CSV
                pd.read_csv(output_file, nrows=1)
                return 'skipped'
            except Exception:
                pass  # If invalid, re-download

        stooq_ticker = self._convert_ticker_to_stooq(ticker)
        params = {
            's': stooq_ticker,
            'i': 'd'  # daily interval
        }

        stooq_failed = False
        try:
            response = requests.get(self.STOOQ_BASE_URL, params=params, timeout=15)

            if response.status_code != 200:
                stooq_failed = True
            else:
                # Check if content is valid CSV
                content = response.content.decode('utf-8')
                if "No data" in content or len(content) < 50:
                    stooq_failed = True
                else:
                    # Parse CSV
                    df = pd.read_csv(io.StringIO(content))

                    if df.empty:
                        stooq_failed = True
                    else:
                        # Normalize column names
                        df.columns = [c.capitalize() for c in df.columns]

                        required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                        if not all(col in df.columns for col in required_cols):
                            stooq_failed = True
                        else:
                            # Ensure Date format YYYY-MM-DD
                            df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')

                            # Add Symbol column
                            df.insert(1, 'Symbol', ticker)

                            # Save to file
                            df.to_csv(output_file, index=False)

                            return True

        except Exception:
            stooq_failed = True

        # Fallback to Alpaca API if Stooq failed
        if stooq_failed and fallback_to_alpaca:
            if self.alpaca_api_key and self.alpaca_secret_key:
                print(" [Stooq failed, trying Alpaca]", end="")
                return self._download_ticker_alpaca_full(ticker)
            else:
                print(" [No data, Alpaca not configured]", end="")
                return False
        elif stooq_failed:
            print(" [No data]", end="")
            return False

        return False

    def _download_ticker_alpaca_full(self, ticker: str) -> bool:
        """
        Download full historical data for a single ticker using Alpaca API.
        Used as fallback when Stooq fails.

        Args:
            ticker (str): Stock ticker symbol

        Returns:
            bool: True if successful, False if failed
        """
        output_file = os.path.join(self.cache_dir, f"{ticker}.csv")

        # Download from 2015-01-01 to latest available date
        start_date = '2015-01-01'
        end_date = self._get_latest_available_date()

        try:
            url = f"{self.alpaca_data_endpoint}/v2/stocks/{ticker}/bars"

            headers = {
                'APCA-API-KEY-ID': self.alpaca_api_key,
                'APCA-API-SECRET-KEY': self.alpaca_secret_key
            }

            all_bars = []
            page_token = None

            # Paginate through all results
            while True:
                params = {
                    'start': start_date,
                    'end': end_date,
                    'timeframe': '1Day',
                    'adjustment': 'raw',
                    'feed': 'iex',
                    'limit': 10000
                }
                if page_token:
                    params['page_token'] = page_token

                response = requests.get(url, headers=headers, params=params, timeout=30)

                if response.status_code == 429:
                    # Rate limit - wait and retry
                    time.sleep(10)
                    response = requests.get(url, headers=headers, params=params, timeout=30)

                if response.status_code != 200:
                    print(f" [Alpaca HTTP {response.status_code}]", end="")
                    return False

                data = response.json()

                if 'bars' in data and data['bars']:
                    all_bars.extend(data['bars'])

                # Check for next page
                page_token = data.get('next_page_token')
                if not page_token:
                    break

            if not all_bars:
                print(" [No Alpaca data]", end="")
                return False

            # Convert to DataFrame
            df = pd.DataFrame(all_bars)

            # Rename columns to match our format
            column_mapping = {
                't': 'Date',
                'o': 'Open',
                'h': 'High',
                'l': 'Low',
                'c': 'Close',
                'v': 'Volume'
            }
            df = df.rename(columns=column_mapping)

            # Convert timestamp to date
            df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')

            # Add Symbol column
            df.insert(1, 'Symbol', ticker)

            # Select only needed columns
            df = df[['Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume']]

            # Sort by date
            df = df.sort_values('Date')

            # Save to file
            df.to_csv(output_file, index=False)

            return True

        except Exception as e:
            print(f" [Alpaca error: {str(e)[:20]}]", end="")
            return False

    def _download_ticker_stooq_incremental(self, ticker: str) -> Optional[bool]:
        """
        Download incremental data for a single ticker using Stooq date-range query.

        Stooq is the same source as the full download, providing consolidated
        market volume. Using Stooq for incremental updates keeps the volume
        data consistent with the historical data. Stooq supports date range
        parameters: ?s=AAPL.US&d1=YYYYMMDD&d2=YYYYMMDD&i=d

        Args:
            ticker (str): Stock ticker symbol

        Returns:
            True if new data was appended, None if already up to date, False on error
        """
        cache_file = os.path.join(self.cache_dir, f"{ticker}.csv")

        if not os.path.exists(cache_file):
            return self._download_ticker_stooq(ticker)

        try:
            df_dates = pd.read_csv(cache_file, usecols=['Date'])
            if df_dates.empty:
                return self._download_ticker_stooq(ticker)
            df_dates['Date'] = pd.to_datetime(df_dates['Date'], utc=True)
            ticker_last_date = df_dates['Date'].max().strftime('%Y-%m-%d')
        except Exception as e:
            print(f" [Error reading cache: {str(e)[:20]}]", end="")
            return False

        last_date_obj = datetime.strptime(ticker_last_date, '%Y-%m-%d')
        start_date = (last_date_obj + timedelta(days=1)).strftime('%Y-%m-%d')
        end_date = self._get_latest_available_date()

        if start_date > end_date:
            return None

        stooq_ticker = self._convert_ticker_to_stooq(ticker)
        params = {
            's':  stooq_ticker,
            'd1': start_date.replace('-', ''),   # YYYYMMDD
            'd2': end_date.replace('-', ''),
            'i':  'd',
        }

        try:
            response = requests.get(self.STOOQ_BASE_URL, params=params, timeout=15)
            if response.status_code != 200:
                print(f" [Stooq HTTP {response.status_code}]", end="")
                return False

            content = response.content.decode('utf-8')
            if 'No data' in content or len(content) < 50:
                # No new trading days available yet
                return None

            new_df = pd.read_csv(io.StringIO(content))
            if new_df.empty:
                return None

            new_df.columns = [c.capitalize() for c in new_df.columns]
            required = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            if not all(c in new_df.columns for c in required):
                print(f" [Stooq unexpected columns]", end="")
                return False

            new_df['Date'] = pd.to_datetime(new_df['Date']).dt.strftime('%Y-%m-%d')
            new_df.insert(1, 'Symbol', ticker)
            new_df = new_df[['Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume']]

            existing_df = pd.read_csv(cache_file)
            if 'Symbol' not in existing_df.columns:
                existing_df['Symbol'] = ticker
            existing_df['Date'] = pd.to_datetime(
                existing_df['Date'], utc=True
            ).dt.strftime('%Y-%m-%d')
            existing_df = existing_df[
                [c for c in ['Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume']
                 if c in existing_df.columns]
            ]

            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=['Date'], keep='last')
            combined_df = combined_df.sort_values('Date').reset_index(drop=True)
            combined_df.to_csv(cache_file, index=False)

            return True

        except Exception as e:
            print(f" [Stooq incremental error: {str(e)[:30]}]", end="")
            return False

    def _download_ticker_alpaca_incremental(self, ticker: str, global_last_date: Optional[str] = None) -> Optional[bool]:
        """
        Download incremental data for a single ticker using Alpaca API

        This method automatically detects each ticker's actual last cached date
        and only downloads the missing data, ensuring accurate per-ticker updates.

        IMPORTANT: Only downloads COMPLETED trading days to avoid incomplete data.
        The end_date is set to yesterday (or last Friday if today is weekend),
        ensuring we never fetch partial intraday data.

        Args:
            ticker (str): Stock ticker symbol
            global_last_date (str, optional): Global last date hint (for optimization).
                If provided and ticker's cache is up-to-date, skip the file read.

        Returns:
            bool: True if successful, False if failed, None if no new data
        """
        cache_file = os.path.join(self.cache_dir, f"{ticker}.csv")

        # Check if cache file exists
        if not os.path.exists(cache_file):
            # No cache, download from Stooq instead
            return self._download_ticker_stooq(ticker)

        # Get this ticker's actual last cached date
        try:
            df = pd.read_csv(cache_file, usecols=['Date', 'Volume'])
            if df.empty:
                return self._download_ticker_stooq(ticker)
            # Use utc=True to handle mixed timezone formats
            df['Date'] = pd.to_datetime(df['Date'], utc=True)
            last_row = df.loc[df['Date'].idxmax()]
            ticker_last_date = last_row['Date'].strftime('%Y-%m-%d')
            last_volume = last_row.get('Volume', 1)
        except Exception as e:
            print(f" [Error reading cache: {str(e)[:20]}]", end="")
            return False

        # If last row has Volume=0 (incomplete data), re-fetch from that date
        # Otherwise, start from the next day
        last_date_obj = datetime.strptime(ticker_last_date, '%Y-%m-%d')
        if pd.isna(last_volume) or last_volume == 0:
            start_date = ticker_last_date  # re-fetch the incomplete day
        else:
            start_date = (last_date_obj + timedelta(days=1)).strftime('%Y-%m-%d')

        # CRITICAL: Only download COMPLETED trading days
        # If current time is after US market close (4:00 PM ET), today's data is available
        # Otherwise, use yesterday to avoid fetching incomplete intraday data
        end_date = self._get_latest_available_date()

        # If already up to date
        if start_date > end_date:
            return None

        try:
            # Alpaca Data API endpoint for historical bars
            url = f"{self.alpaca_data_endpoint}/v2/stocks/{ticker}/bars"

            headers = {
                'APCA-API-KEY-ID': self.alpaca_api_key,
                'APCA-API-SECRET-KEY': self.alpaca_secret_key
            }

            params = {
                'start': start_date,
                'end': end_date,
                'timeframe': '1Day',
                'adjustment': 'all',  # split+dividend adjusted, consistent with Stooq history
                'feed': 'iex',        # IEX feed: available on free Alpaca tier
                'limit': 10000
            }

            response = requests.get(url, headers=headers, params=params, timeout=30)

            if response.status_code != 200:
                # Handle rate limiting (429)
                if response.status_code == 429:
                    print(f" [Rate limit]", end="")
                    # Exponential backoff: wait 10s, 20s, 40s
                    wait_time = 10 * (2 ** 0)  # Start with 10s
                    print(f" [Waiting {wait_time}s]", end="")
                    time.sleep(wait_time)
                    # Retry once after backoff
                    response = requests.get(url, headers=headers, params=params, timeout=30)
                    if response.status_code != 200:
                        print(f" [HTTP {response.status_code}]", end="")
                        return False
                else:
                    print(f" [HTTP {response.status_code}]", end="")
                    return False

            data = response.json()

            if 'bars' not in data or not data['bars']:
                # No new data available
                return None

            # Convert to DataFrame
            bars = data['bars']
            new_df = pd.DataFrame(bars)

            # Rename columns to match our format
            column_mapping = {
                't': 'Date',
                'o': 'Open',
                'h': 'High',
                'l': 'Low',
                'c': 'Close',
                'v': 'Volume'
            }
            new_df = new_df.rename(columns=column_mapping)

            # Convert timestamp to date
            new_df['Date'] = pd.to_datetime(new_df['Date']).dt.strftime('%Y-%m-%d')

            # Add Symbol column
            new_df.insert(1, 'Symbol', ticker)

            # Select only needed columns
            new_df = new_df[['Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume']]

            # Load existing cache and standardize format
            existing_df = pd.read_csv(cache_file)

            # Standardize existing data: select only standard columns if they exist
            standard_cols = ['Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume']
            available_cols = [c for c in standard_cols if c in existing_df.columns]

            # If Symbol column missing, add it
            if 'Symbol' not in existing_df.columns:
                existing_df['Symbol'] = ticker

            existing_df = existing_df[available_cols]

            # Standardize existing dates (handle timezone-aware formats)
            existing_df['Date'] = pd.to_datetime(existing_df['Date'], utc=True).dt.strftime('%Y-%m-%d')

            # Concatenate and remove duplicates
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)

            # Remove duplicates based on Date (keep new data if duplicate)
            combined_df = combined_df.drop_duplicates(subset=['Date'], keep='last')
            combined_df = combined_df.sort_values('Date')

            # Save back to cache with standardized format
            combined_df.to_csv(cache_file, index=False)

            return True

        except Exception as e:
            print(f" [Error: {str(e)[:30]}]", end="")
            return False

    def _get_latest_available_date(self) -> str:
        """
        Get the latest date for which complete market data is available.

        Logic:
        - If current time is after US market close (4:00 PM ET), today's data is available
        - Otherwise, only yesterday's data is complete

        Returns:
            str: Latest available date in YYYY-MM-DD format
        """
        # Get current time in US Eastern timezone
        eastern_tz = ZoneInfo(self.US_EASTERN_TZ)
        now_eastern = datetime.now(eastern_tz)
        today_eastern = now_eastern.date()

        # Check if market has closed (after 4:00 PM ET)
        market_closed_today = now_eastern.hour >= self.MARKET_CLOSE_HOUR

        if market_closed_today:
            # Today's data is complete, can update to today
            return today_eastern.strftime('%Y-%m-%d')
        else:
            # Market still open or hasn't opened, use yesterday
            yesterday = today_eastern - timedelta(days=1)
            return yesterday.strftime('%Y-%m-%d')

    def _get_cache_last_date(self) -> Optional[str]:
        """
        Get the latest date across all cached files

        Returns:
            str: Latest date in format YYYY-MM-DD, or None if no cache
        """
        if not os.path.exists(self.cache_dir):
            return None

        cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.csv')]

        if not cache_files:
            return None

        latest_date = None

        # Sample up to 10 files to find the common last date
        sample_files = cache_files[:min(10, len(cache_files))]

        for filename in sample_files:
            try:
                filepath = os.path.join(self.cache_dir, filename)
                df = pd.read_csv(filepath, usecols=['Date'])

                if df.empty:
                    continue

                # Use utc=True to handle mixed timezone formats
                df['Date'] = pd.to_datetime(df['Date'], utc=True)
                file_last_date = df['Date'].max()

                if latest_date is None or file_last_date > latest_date:
                    latest_date = file_last_date

            except Exception:
                continue

        if latest_date:
            return latest_date.strftime('%Y-%m-%d')

        return None

    def _print_summary(self, elapsed_time: float):
        """
        Print download summary statistics

        Args:
            elapsed_time (float): Elapsed time in seconds
        """
        print("\n" + "=" * 70)
        print("DOWNLOAD SUMMARY")
        print("=" * 70)
        print(f"\n📊 Statistics:")
        print(f"   Total stocks: {self.stats['total']}")
        print(f"   ✓ Downloaded: {self.stats['downloaded']}")
        print(f"   ✗ Failed: {self.stats['failed']}")
        print(f"   ⊘ Skipped: {self.stats['skipped']}")
        print(f"\n⏱️  Time elapsed: {int(elapsed_time // 60)}m {int(elapsed_time % 60)}s")
        print(f"📁 Cache directory: {self.cache_dir}")
        print("\n" + "=" * 70)

    def update_ticker(self, ticker: str) -> bool:
        """
        Update a single ticker's cache data using Alpaca API

        This is a convenience method for updating individual stocks without
        processing the entire S&P 500 list.

        Args:
            ticker (str): Stock ticker symbol (e.g., 'AAPL', 'MSFT')

        Returns:
            bool: True if updated successfully, False if failed, None if already up to date

        Example:
            manager = CacheManager()
            manager.update_ticker('AAPL')  # Update only AAPL
        """
        # Check Alpaca credentials
        if not self.alpaca_api_key or not self.alpaca_secret_key:
            print("❌ Error: Alpaca API credentials not found in .env file")
            return False

        print(f"📥 Updating {ticker}...", end=" ")
        sys.stdout.flush()

        result = self._download_ticker_alpaca_incremental(ticker)

        if result is True:
            print("✓ Updated")
            return True
        elif result is None:
            print("⊘ Already up to date")
            return True
        else:
            print("✗ Failed")
            return False

    def update_tickers(self, tickers: List[str]) -> Dict:
        """
        Update multiple specific tickers' cache data using Alpaca API

        Args:
            tickers (list): List of ticker symbols to update

        Returns:
            dict: Summary of update results
                - 'updated': list of successfully updated tickers
                - 'skipped': list of tickers already up to date
                - 'failed': list of tickers that failed to update

        Example:
            manager = CacheManager()
            result = manager.update_tickers(['AAPL', 'MSFT', 'GOOGL'])
        """
        results = {
            'updated': [],
            'skipped': [],
            'failed': []
        }

        # Check Alpaca credentials
        if not self.alpaca_api_key or not self.alpaca_secret_key:
            print("❌ Error: Alpaca API credentials not found in .env file")
            results['failed'] = tickers
            return results

        print(f"📥 Updating {len(tickers)} tickers...")

        for idx, ticker in enumerate(tickers):
            print(f"[{idx + 1}/{len(tickers)}] {ticker:<6} ", end="")
            sys.stdout.flush()

            result = self._download_ticker_alpaca_incremental(ticker)

            if result is True:
                results['updated'].append(ticker)
                print("✓")
            elif result is None:
                results['skipped'].append(ticker)
                print("⊘")
            else:
                results['failed'].append(ticker)
                print("✗")

            if idx < len(tickers) - 1:
                time.sleep(0.1)

        print(f"\n📊 Summary: {len(results['updated'])} updated, "
              f"{len(results['skipped'])} up to date, {len(results['failed'])} failed")

        return results

    def get_ticker_last_date(self, ticker: str) -> Optional[str]:
        """
        Get the last cached date for a specific ticker

        Args:
            ticker (str): Stock ticker symbol

        Returns:
            str: Last date in YYYY-MM-DD format, or None if not cached

        Example:
            manager = CacheManager()
            last_date = manager.get_ticker_last_date('AAPL')
            print(f"AAPL data up to: {last_date}")
        """
        cache_file = os.path.join(self.cache_dir, f"{ticker}.csv")

        if not os.path.exists(cache_file):
            return None

        try:
            df = pd.read_csv(cache_file, usecols=['Date'])
            if df.empty:
                return None
            # Use utc=True to handle mixed timezone formats
            df['Date'] = pd.to_datetime(df['Date'], utc=True)
            return df['Date'].max().strftime('%Y-%m-%d')
        except Exception:
            return None

    def repair_volume(self, since_date: str = '2026-01-13') -> Dict:
        """
        Repair corrupted volume data caused by Alpaca IEX feed.

        IEX only captures ~2-5% of total market volume. This method
        re-downloads OHLCV data from Stooq for all tickers whose 2026
        volume is suspiciously low (< 10% of 2025 median), replacing
        the bad rows with correct consolidated-market data.

        Args:
            since_date: First date to consider for repair (default '2026-01-13',
                        the day IEX data started appearing in most caches)

        Returns:
            dict with 'repaired', 'skipped', 'failed' lists
        """
        print("=" * 70)
        print(f"VOLUME REPAIR - replacing IEX data since {since_date} with Stooq")
        print("=" * 70)

        results = {'repaired': [], 'skipped': [], 'failed': []}

        if not os.path.exists(self.cache_dir):
            print("❌ Cache directory not found")
            return results

        cache_files = sorted(f for f in os.listdir(self.cache_dir) if f.endswith('.csv'))
        total = len(cache_files)

        for idx, fname in enumerate(cache_files):
            ticker = fname.replace('.csv', '')
            cache_file = os.path.join(self.cache_dir, fname)
            progress = (idx + 1) / total * 100
            print(f"[{idx+1}/{total}] ({progress:.1f}%) {ticker:<6} ", end="")
            sys.stdout.flush()

            try:
                df = pd.read_csv(cache_file)
                df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.strftime('%Y-%m-%d')

                mask_2025 = pd.to_datetime(df['Date']).dt.year == 2025
                mask_2026 = pd.to_datetime(df['Date']).dt.year == 2026

                if mask_2026.sum() == 0:
                    print("⊘ (no 2026 data)")
                    results['skipped'].append(ticker)
                    continue

                v2025 = df.loc[mask_2025, 'Volume'].median() if mask_2025.sum() > 0 else 0
                v2026 = df.loc[mask_2026, 'Volume'].median()

                if v2025 == 0 or (v2026 / v2025) >= 0.10:
                    print("⊘ (volume OK)")
                    results['skipped'].append(ticker)
                    continue

                # Volume is corrupted: re-fetch from Stooq since since_date
                stooq_ticker = self._convert_ticker_to_stooq(ticker)
                end_date = self._get_latest_available_date()
                params = {
                    's':  stooq_ticker,
                    'd1': since_date.replace('-', ''),
                    'd2': end_date.replace('-', ''),
                    'i':  'd',
                }

                response = requests.get(self.STOOQ_BASE_URL, params=params, timeout=15)
                if response.status_code != 200:
                    print(f"✗ [HTTP {response.status_code}]")
                    results['failed'].append(ticker)
                    time.sleep(self.STOOQ_REQUEST_DELAY)
                    continue

                content = response.content.decode('utf-8')
                if 'No data' in content or len(content) < 50:
                    print("✗ [No Stooq data]")
                    results['failed'].append(ticker)
                    time.sleep(self.STOOQ_REQUEST_DELAY)
                    continue

                new_df = pd.read_csv(io.StringIO(content))
                if new_df.empty:
                    print("✗ [Empty response]")
                    results['failed'].append(ticker)
                    time.sleep(self.STOOQ_REQUEST_DELAY)
                    continue

                new_df.columns = [c.capitalize() for c in new_df.columns]
                new_df['Date'] = pd.to_datetime(new_df['Date']).dt.strftime('%Y-%m-%d')
                if 'Symbol' not in new_df.columns:
                    new_df.insert(1, 'Symbol', ticker)

                # Remove old corrupted rows and replace with Stooq data
                df_clean = df[pd.to_datetime(df['Date']) < pd.Timestamp(since_date)].copy()
                combined = pd.concat([df_clean, new_df[
                    [c for c in ['Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume']
                     if c in new_df.columns]
                ]], ignore_index=True)
                combined = combined.drop_duplicates(subset=['Date'], keep='last')
                combined = combined.sort_values('Date').reset_index(drop=True)
                combined.to_csv(cache_file, index=False)

                print("✓")
                results['repaired'].append(ticker)

            except Exception as e:
                print(f"✗ [{str(e)[:25]}]")
                results['failed'].append(ticker)

            time.sleep(self.STOOQ_REQUEST_DELAY)

        print(f"\n{'='*70}")
        print(f"REPAIR SUMMARY: {len(results['repaired'])} repaired | "
              f"{len(results['skipped'])} skipped | {len(results['failed'])} failed")
        print(f"{'='*70}")
        return results

    def get_cache_status(self) -> Dict:
        """
        Get cache metadata and status information

        Returns:
            dict: Cache status including latest date, number of tickers, etc.
        """
        status = {
            'cache_dir': self.cache_dir,
            'exists': os.path.exists(self.cache_dir),
            'latest_date': None,
            'num_tickers': 0,
            'ticker_list': []
        }

        if not status['exists']:
            return status

        try:
            cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.csv')]
            status['num_tickers'] = len(cache_files)
            status['ticker_list'] = [f.replace('.csv', '') for f in cache_files]

            if cache_files:
                status['latest_date'] = self._get_cache_last_date()

        except Exception as e:
            print(f"✗ Error getting cache status: {e}")

        return status


def main():
    """
    Command-line interface for the cache manager
    """
    print("\n" + "=" * 70)
    print("S&P 500 CACHE MANAGER - Stooq + Alpaca")
    print("=" * 70)

    manager = CacheManager()

    # Check for command line arguments
    force_full = '--full' in sys.argv or '-f' in sys.argv

    if force_full:
        print("\n🔄 Force full download mode activated")

    # Run update
    success = manager.update_cache(force_full=force_full)

    if success:
        print("\n✓ Cache update completed successfully")
    else:
        print("\n⚠️  Cache update completed with some failures")
        print("   Check the summary above for details")

    print()


if __name__ == "__main__":
    main()
