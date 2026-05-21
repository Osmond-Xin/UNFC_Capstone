#!/usr/bin/env python3
"""
Generate Expiry Day Watchlist - Multi-Pattern Only

Scans S&P 500 stocks for pattern matches before the next expiry date.
Only outputs stocks that match MULTIPLE patterns (2+).

Usage:
    python tools/generate_watchlist.py
    python tools/generate_watchlist.py --target-expiry 2025-12-19
    python tools/generate_watchlist.py --start 2022-01-01 --end 2025-12-31
"""

import sys
import os
import argparse
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from modules.data import CacheManager, DataLoader, ExpiryCalculator
from modules.features import TechnicalIndicators, FeaturePipeline
from modules.evaluation import ExpiryScanner
from modules.analysis.volatility_filter import VolatilityFilter


def parse_args():
    parser = argparse.ArgumentParser(description='Generate multi-pattern watchlist for expiry day')
    parser.add_argument('--target-expiry', type=str, default='auto',
                        help='Target expiry date (YYYY-MM-DD) or "auto" for most recent')
    parser.add_argument('--start', type=str, default='2022-01-01',
                        help='Analysis start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=None,
                        help='Analysis end date (YYYY-MM-DD), defaults to today')
    parser.add_argument('--no-filter', action='store_true',
                        help='Disable historic mover filter (scan all S&P 500)')
    parser.add_argument('--min-patterns', type=int, default=2,
                        help='Minimum number of patterns required (default: 2)')
    return parser.parse_args()


def count_patterns(pattern_str: str) -> int:
    """Count number of patterns in comma-separated string"""
    if pd.isna(pattern_str) or not pattern_str:
        return 0
    return len([p.strip() for p in pattern_str.split(',') if p.strip()])


def main():
    args = parse_args()

    # Set defaults
    if args.end is None:
        args.end = pd.Timestamp.now().strftime('%Y-%m-%d')

    print("=" * 60)
    print("EXPIRY DAY WATCHLIST GENERATOR (Multi-Pattern)")
    print("=" * 60)
    print(f"Target Expiry: {args.target_expiry}")
    print(f"Analysis Range: {args.start} to {args.end}")
    print(f"Min Patterns: {args.min_patterns}")
    print()

    # Determine project root (parent of tools directory)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache_dir = os.path.join(project_root, 'cache', 'constituent_data')

    # Initialize components
    loader = DataLoader(cache_dir=cache_dir)

    # Generate expiry dates
    expiry_dates = ExpiryCalculator.generate_expiry_dates(args.start, args.end)

    # Determine target expiry
    if args.target_expiry == 'auto':
        today = pd.Timestamp.now()
        past_expiries = [d for d in expiry_dates if d <= today]
        target_expiry = past_expiries[-1] if past_expiries else expiry_dates[0]
    else:
        target_expiry = pd.to_datetime(args.target_expiry)

    print(f"Using Target Expiry: {target_expiry.strftime('%Y-%m-%d')}")

    # Load stock data
    sp500_tickers = loader.get_sp500_tickers()
    print(f"Loading {len(sp500_tickers)} stocks...")
    stock_data = loader.load_sp500_batch(
        sp500_tickers,
        start_date=args.start,
        end_date=args.end,
        validate=False
    )
    print(f"Loaded {len(stock_data)} stocks")

    # Calculate features
    pipeline = FeaturePipeline([
        TechnicalIndicators(config={
            'rsi_period': 14,
            'ma_periods': [9, 20, 50],
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'bb_period': 20,
            'volume_ma_period': 20,
            'consecutive_lookback': 5
        })
    ])

    print("Calculating technical indicators...")
    for ticker in stock_data:
        stock_data[ticker] = pipeline.transform(stock_data[ticker])

    # Apply volatility filter (historic movers)
    if not args.no_filter:
        print("Applying historic mover filter...")
        vol_filter = VolatilityFilter(lookback_periods=11)
        target_tickers = vol_filter.get_target_universe(stock_data, target_expiry)
        stock_data = {t: stock_data[t] for t in target_tickers if t in stock_data}
        print(f"Filtered to {len(stock_data)} historic movers")

    # Use target_expiry as the scan date
    scan_expiry = target_expiry
    print(f"\nScanning for patterns on: {scan_expiry.strftime('%Y-%m-%d (%A)')}")

    # Scan for patterns
    scanner = ExpiryScanner()
    watchlist = scanner.scan(stock_data, scan_expiry)

    if watchlist.empty:
        print("\nNo stocks match any patterns.")
        return

    # Filter for multi-pattern matches only
    watchlist['pattern_count'] = watchlist['Pattern'].apply(count_patterns)
    multi_pattern = watchlist[watchlist['pattern_count'] >= args.min_patterns].copy()

    if multi_pattern.empty:
        print(f"\nNo stocks match {args.min_patterns}+ patterns.")
        print(f"(Total with any pattern: {len(watchlist)})")
        return

    # Sort by pattern count descending
    multi_pattern = multi_pattern.sort_values('pattern_count', ascending=False)

    # Output results
    print("\n" + "=" * 60)
    print(f"MULTI-PATTERN WATCHLIST - {scan_expiry.strftime('%Y-%m-%d (%A)')}")
    print(f"Stocks matching {args.min_patterns}+ patterns: {len(multi_pattern)}")
    print("=" * 60)

    # Print ticker list
    tickers = multi_pattern['Ticker'].tolist()
    print(f"\nTickers: {', '.join(tickers)}")

    # Print details
    print("\nDetails:")
    print("-" * 60)
    for _, row in multi_pattern.iterrows():
        print(f"\n{row['Ticker']} ({row['pattern_count']} patterns):")
        print(f"  Patterns: {row['Pattern']}")
        print(f"  RSI: {row['Current_RSI']:.1f}")
        print(f"  MA20 Dist: {row['MA20_Dist']:+.2f}%")
        print(f"  Vol Ratio: {row['Volume_Ratio']:.2f}x")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
