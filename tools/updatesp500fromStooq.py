
import os
import sys
import time
import shutil
import requests
import pandas as pd
import io
from datetime import datetime

class StooqDownloader:
    def __init__(self):
        # Determine paths relative to this script
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.script_dir, 'data')
        self.list_file = os.path.join(self.script_dir, 'sp500_list.csv')
        
        self.stooq_base_url = "https://stooq.com/q/d/l/"
        
        # Stats
        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0
        }

    def run(self):
        print("=" * 60)
        print("S&P 500 DATA DOWNLOADER (STOOQ)")
        print("=" * 60)

        # 1. Load Tickers
        tickers = self._load_tickers()
        if not tickers:
            print("No tickers found. Exiting.")
            sys.exit(1)
        
        self.stats['total'] = len(tickers)
        print(f"Found {len(tickers)} tickers in {os.path.basename(self.list_file)}")

        # 2. Prepare Data Directory
        self._prepare_data_dir()

        # 3. Download Data
        print(f"\nStarting download... (Delay: 1.0s)")
        print("-" * 60)

        start_time = time.time()
        
        for idx, ticker in enumerate(tickers):
            self._process_ticker(idx, ticker, len(tickers))
            
            # Rate limiting - be polite to Stooq
            if idx < len(tickers) - 1:
                time.sleep(1.0)
        
        # 4. Summary
        elapsed = time.time() - start_time
        self._print_summary(elapsed)

    def _load_tickers(self):
        if not os.path.exists(self.list_file):
            print(f"Error: {self.list_file} not found.")
            return []
        
        try:
            df = pd.read_csv(self.list_file)
            if 'Symbol' not in df.columns:
                print("Error: 'Symbol' column not found in CSV.")
                return []
            
            # Clean tickers
            tickers = df['Symbol'].dropna().apply(lambda x: str(x).strip()).tolist()
            return [t for t in tickers if t]
        except Exception as e:
            print(f"Error reading CSV: {e}")
            return []

    def _prepare_data_dir(self):
        # Remove if exists
        if os.path.exists(self.data_dir):
            try:
                shutil.rmtree(self.data_dir)
                print("Cleaned existing data directory.")
            except Exception as e:
                print(f"Error cleaning data directory: {e}")
                sys.exit(1)
        
        # Create fresh
        os.makedirs(self.data_dir, exist_ok=True)
        print(f"Created fresh data directory: {self.data_dir}")

    def _process_ticker(self, idx, ticker, total):
        # Progress
        progress = (idx + 1) / total * 100
        print(f"[{idx+1}/{total}] {progress:5.1f}% {ticker:<6} ", end="")
        sys.stdout.flush()

        # Convert to Stooq format (BRK-B -> BRK.B.US)
        stooq_ticker = f"{ticker.replace('-', '.')}.US"
        
        params = {
            's': stooq_ticker,
            'i': 'd' # daily
        }

        try:
            response = requests.get(self.stooq_base_url, params=params, timeout=15)
            
            if response.status_code == 200:
                content = response.content.decode('utf-8')
                
                # Basic validation
                if "No data" in content or len(content) < 50:
                    print("✗ (No data)")
                    self.stats['failed'] += 1
                    return

                # Parse to verify CSV structure
                try:
                    df = pd.read_csv(io.StringIO(content))
                    
                    if df.empty:
                        print("✗ (Empty CSV)")
                        self.stats['failed'] += 1
                        return
                        
                    # Save
                    output_path = os.path.join(self.data_dir, f"{ticker}.csv")
                    with open(output_path, 'w') as f:
                        f.write(content)
                        
                    print("✓")
                    self.stats['success'] += 1
                    
                except Exception as e:
                    print(f"✗ (Parse Error)")
                    self.stats['failed'] += 1
            else:
                print(f"✗ (HTTP {response.status_code})")
                self.stats['failed'] += 1

        except Exception as e:
            print(f"✗ (Error: {str(e)[:20]})")
            self.stats['failed'] += 1

    def _print_summary(self, elapsed):
        print("\n" + "=" * 60)
        print("DOWNLOAD SUMMARY")
        print("=" * 60)
        print(f"Total Tickers : {self.stats['total']}")
        print(f"Successful    : {self.stats['success']}")
        print(f"Failed        : {self.stats['failed']}")
        print(f"Time Elapsed  : {int(elapsed // 60)}m {int(elapsed % 60)}s")
        print(f"Data Location : {self.data_dir}")
        print("=" * 60)

if __name__ == "__main__":
    downloader = StooqDownloader()
    downloader.run()
