#!/usr/bin/env python3
"""
Update S&P 500 List and Data

This script provides two functions:
1. Update S&P 500 ticker list from Wikipedia
2. Incrementally update stock price data via Stooq (date-range queries)

Usage:
    python tools/update_sp500.py              # Update both list and data
    python tools/update_sp500.py --list-only  # Only update ticker list
    python tools/update_sp500.py --data-only  # Only update price data
"""

import io
import os
import sys
import argparse

# Add project root to path for imports
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd
import requests


def update_sp500_list():
    """Update S&P 500 ticker list from Wikipedia."""
    try:
        print("=" * 60)
        print("Fetching S&P 500 list from Wikipedia...")
        print("=" * 60)
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {"User-Agent": "Mozilla/5.0 (compatible; sp500-updater/1.0)"}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        tables = pd.read_html(io.StringIO(response.text))

        # Validate: check if tables were found
        if not tables:
            print("✗ Error: No tables found on Wikipedia page")
            return False

        # The first table is usually the constituents list
        df = tables[0]

        # Validate: check required columns exist
        required_cols = ['Symbol', 'Security']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"✗ Error: Missing required columns: {missing_cols}")
            print(f"   Available columns: {list(df.columns)}")
            return False

        # Select relevant columns
        # Wikipedia columns: 'Symbol', 'Security', 'GICS Sector', etc.
        df_clean = df[['Symbol', 'Security']].copy()

        # Validate: check if data is not empty
        if df_clean.empty:
            print("✗ Error: No data found in the S&P 500 table")
            return False

        # Standardize Symbol format: replace '.' with '-' for consistency
        # Wikipedia uses BRK.B, but data sources like Stooq expect BRK-B
        df_clean['Symbol'] = df_clean['Symbol'].str.replace('.', '-', regex=False)

        # Remove any rows with empty symbols
        df_clean = df_clean.dropna(subset=['Symbol'])
        df_clean = df_clean[df_clean['Symbol'].str.strip() != '']

        # Validate: ensure we have a reasonable number of companies
        if len(df_clean) < 400:
            print(f"✗ Warning: Only {len(df_clean)} companies found (expected ~500)")
            print("   The Wikipedia page structure may have changed")
            return False

        # Define output path (absolute, relative to project root)
        output_dir = os.path.join(_PROJECT_ROOT, 'cache')
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'sp500_list.csv')

        # Save to CSV
        df_clean.to_csv(output_file, index=False)

        print(f"✓ Successfully updated S&P 500 list.")
        print(f"   Saved to: {output_file}")
        print(f"   Total companies: {len(df_clean)}")
        print("   Preview:")
        print(df_clean.head())
        return True

    except Exception as e:
        print(f"✗ Error updating list: {e}")
        print("Note: This script requires 'lxml' or 'beautifulsoup4' and 'html5lib'.")
        print("Try: pip install lxml")
        return False


def update_sp500_data():
    """
    Incrementally update S&P 500 stock price data.

    Uses CacheManager.incremental_update() to fetch missing trading days
    from Stooq (date-range queries), appending to existing CSV files.
    No API key required.
    """
    try:
        print("\n" + "=" * 60)
        print("Updating S&P 500 price data (incremental via Stooq)...")
        print("=" * 60)

        from modules.data import CacheManager

        cache_mgr = CacheManager()

        # Show current cache status
        status = cache_mgr.get_cache_status()
        print(f"\nCurrent cache status:")
        print(f"   Latest date:      {status['latest_date']}")
        print(f"   Number of tickers: {status['num_tickers']}")

        # Run incremental update
        print(f"\nRunning incremental update...")
        result = cache_mgr.incremental_update()

        new_status = cache_mgr.get_cache_status()
        print(f"\n✓ Incremental update completed.")
        print(f"   New latest date:  {new_status['latest_date']}")
        print(f"   Tickers in cache: {new_status['num_tickers']}")

        return True

    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("Make sure you're running from the project root directory.")
        return False
    except Exception as e:
        print(f"✗ Error updating data: {e}")
        print("\nPossible causes:")
        print("   1. No internet connection")
        print("   2. Stooq temporarily unavailable")
        print("   3. Cache directory missing or corrupted")
        return False



def main():
    parser = argparse.ArgumentParser(
        description="Update S&P 500 ticker list and price data"
    )
    parser.add_argument(
        '--list-only',
        action='store_true',
        help='Only update the ticker list from Wikipedia'
    )
    parser.add_argument(
        '--data-only',
        action='store_true',
        help='Only update price data via Stooq incremental update'
    )
    args = parser.parse_args()

    print("=" * 60)
    print("S&P 500 Data Update Tool")
    print("=" * 60)

    success = True

    if args.data_only:
        success = update_sp500_data()
    elif args.list_only:
        success = update_sp500_list()
    else:
        # Update both (default)
        list_ok = update_sp500_list()
        data_ok = update_sp500_data()
        success = list_ok and data_ok

    print("\n" + "=" * 60)
    if success:
        print("✓ Update completed successfully!")
    else:
        print("✗ Update completed with errors.")
    print("=" * 60)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
