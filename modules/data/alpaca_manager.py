"""
Alpaca API Cache Manager for S&P 500 Stock Data

This module downloads historical stock data using the Alpaca Data API v2.
It replaces the Yahoo Finance downloader to provide more reliable access.

Features:
- Authenticated requests using API Key & Secret
- Rate limiting respects Alpaca's 200 req/min limit
- Data normalization to be compatible with existing analysis tools
- Resume capability: skips data that is already downloaded
"""

import os
import sys
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from typing import List, Optional, Dict

class AlpacaCacheManager:
    """
    Manages data downloading from Alpaca API.
    """
    
    # Configuration
    CACHE_DIR = 'cache/constituent_data'
    SP500_LIST_FILE = 'cache/sp500_list.csv'
    START_DATE = '2016-01-01' # Adjusted to a reasonable default, can be 2010 if needed
    
    # Alpaca Rate Limit: 200 requests per minute
    # Safe delay: 60s / 200 = 0.3s. Let's use 0.4s to be safe.
    REQUEST_DELAY = 0.4 
    
    def __init__(self):
        """Initialize and load environment variables"""
        load_dotenv()
        
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.secret_key = os.getenv('ALPACA_SECRET_KEY')
        self.endpoint = os.getenv('ALPACA_ENDPOINT', 'https://paper-api.alpaca.markets')
        
        if not self.api_key or not self.secret_key:
            print("❌ Error: Alpaca credentials not found in .env file")
            sys.exit(1)
            
        self._setup_paths()
        
        self.stats = {
            'downloaded': 0,
            'failed': 0,
            'skipped': 0,
            'total': 0
        }

    def _setup_paths(self):
        """Determine correct paths relative to execution directory"""
        self.base_dir = '.'

        self.cache_dir = self.CACHE_DIR
        self.sp500_list_file = self.SP500_LIST_FILE

        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_headers(self) -> Dict[str, str]:
        return {
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.secret_key,
            'accept': 'application/json'
        }

    def update_cache(self) -> bool:
        """Main execution method"""
        print("=" * 70)
        print("ALPACA DATA DOWNLOADER")
        print("=" * 70)
        print(f"Endpoint: {self.endpoint}")
        
        tickers = self._load_sp500_list()
        self.stats['total'] = len(tickers)
        
        print(f"\n📋 Configuration:")
        print(f"   Stocks to process: {len(tickers)}")
        print(f"   Start Date: {self.START_DATE}")
        print(f"   Rate Limit Delay: {self.REQUEST_DELAY}s")
        
        print(f"\n📥 Starting download...\n")
        
        start_time = time.time()
        
        for idx, ticker in enumerate(tickers):
            progress = (idx + 1) / len(tickers) * 100
            print(f"[{idx + 1}/{len(tickers)}] ({progress:.1f}%) {ticker:<6} ", end="")
            sys.stdout.flush()
            
            # Check if file exists (Resume capability)
            if self._file_exists(ticker):
                print("✓ (skipped)")
                self.stats['skipped'] += 1
                continue
                
            success = self._download_ticker(ticker)
            
            if success:
                print("✓")
                self.stats['downloaded'] += 1
            else:
                print("✗")
                self.stats['failed'] += 1
                
            # Rate limiting
            time.sleep(self.REQUEST_DELAY)
            
        self._print_summary(time.time() - start_time)
        return self.stats['failed'] == 0

    def _file_exists(self, ticker: str) -> bool:
        filepath = os.path.join(self.cache_dir, f"{ticker}.csv")
        return os.path.exists(filepath) and os.path.getsize(filepath) > 0

    def _download_ticker(self, ticker: str) -> bool:
        """Fetch data from Alpaca and save as CSV"""
        url = f"https://data.alpaca.markets/v2/stocks/{ticker}/bars"
        
        params = {
            'timeframe': '1Day',
            'start': self.START_DATE,
            'limit': 10000,
            'adjustment': 'raw',
            'feed': 'iex'  # 'iex' is usually available for free/paper
        }
        
        try:
            response = requests.get(url, headers=self._get_headers(), params=params, timeout=10)
            
            if response.status_code != 200:
                print(f" [Error: {response.status_code} - {response.text}]", end="")
                return False
                
            data = response.json()
            
            if 'bars' not in data or not data['bars']:
                print(" [No data]", end="")
                # Return True because it's not a fetch error, just no data exists (maybe delisted)
                # But creating an empty file might be useful to skip next time? 
                # For now let's return False to indicate 'no data saved'
                return False
                
            # Parse data
            bars = data['bars']
            df = pd.DataFrame(bars)
            
            # Rename columns to match Yahoo Finance format
            # Alpaca: t, o, h, l, c, v
            # Target: Date, Open, High, Low, Close, Volume
            df = df.rename(columns={
                't': 'Date',
                'o': 'Open',
                'h': 'High',
                'l': 'Low',
                'c': 'Close',
                'v': 'Volume'
            })
            
            # Format Date
            df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
            
            # Add Symbol column
            df.insert(1, 'Symbol', ticker)
            
            # Keep only necessary columns
            columns_to_keep = ['Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume']
            df = df[columns_to_keep]
            
            # Save to CSV
            output_file = os.path.join(self.cache_dir, f"{ticker}.csv")
            df.to_csv(output_file, index=False)
            
            return True
            
        except Exception as e:
            print(f" [Exception: {e}]", end="")
            return False

    def _load_sp500_list(self) -> List[str]:
        if not os.path.exists(self.sp500_list_file):
            print(f"❌ Error: List file not found: {self.sp500_list_file}")
            sys.exit(1)
        try:
            df = pd.read_csv(self.sp500_list_file)
            return [t.strip() for t in df['Symbol'].dropna() if t.strip()]
        except Exception as e:
            print(f"❌ Error loading list: {e}")
            sys.exit(1)

    def _print_summary(self, elapsed_time):
        print("\n" + "=" * 70)
        print("SUMMARY")
        print(f"Total: {self.stats['total']}")
        print(f"Downloaded: {self.stats['downloaded']}")
        print(f"Skipped: {self.stats['skipped']}")
        print(f"Failed: {self.stats['failed']}")
        print(f"Time: {int(elapsed_time)}s")
        print("=" * 70)

if __name__ == "__main__":
    manager = AlpacaCacheManager()
    manager.update_cache()
