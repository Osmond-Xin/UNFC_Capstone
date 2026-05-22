#!/usr/bin/env python3
"""
Return Calculator Tool

Calculate stock returns based on buy/sell dates using cached data.
- Buy: at the OPEN price of the next trading day after buy_date
- Sell: at the OPEN price of sell_date (or nearest prior trading day)
- Max Gain: highest percentage gain during holding period (based on daily High)
- Max Drawdown: lowest percentage loss during holding period (based on daily Low)

Usage:
    # Command line
    python return_calculator.py AAPL,GOOGL,MSFT 2024-01-02 2024-12-31

    # As module
    from return_calculator import calculate_returns
    results = calculate_returns(['AAPL', 'GOOGL'], '2024-01-02', '2024-12-31')
"""

import os
import sys
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional, Tuple

# Add parent directory to path for module imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_cache_dir() -> str:
    """Get the cache directory path"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(script_dir, '..', 'cache', 'constituent_data')
    return os.path.normpath(cache_dir)


def load_ticker_data(ticker: str, cache_dir: str) -> Optional[pd.DataFrame]:
    """
    Load stock data from cache

    Args:
        ticker: Stock ticker symbol
        cache_dir: Path to cache directory

    Returns:
        DataFrame with stock data or None if not found
    """
    filepath = os.path.join(cache_dir, f"{ticker}.csv")

    if not os.path.exists(filepath):
        return None

    try:
        df = pd.read_csv(filepath)
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        df = df.set_index('Date').sort_index()
        return df
    except Exception as e:
        print(f"Error loading {ticker}: {e}")
        return None


def find_nearest_trading_day(
    df: pd.DataFrame,
    target_date: datetime,
    direction: str = 'forward',
    price_type: str = 'Close'
) -> Tuple[Optional[datetime], Optional[float]]:
    """
    Find the nearest trading day and its price

    Args:
        df: DataFrame with stock data
        target_date: Target date to find
        direction: 'forward' to find next trading day, 'backward' for previous
        price_type: 'Open' or 'Close' price to return

    Returns:
        Tuple of (actual_date, price) or (None, None)
    """
    target = pd.Timestamp(target_date)

    if target in df.index:
        return target, df.loc[target, price_type]

    if direction == 'forward':
        future_dates = df.index[df.index >= target]
        if len(future_dates) > 0:
            actual_date = future_dates[0]
            return actual_date, df.loc[actual_date, price_type]
    else:
        past_dates = df.index[df.index <= target]
        if len(past_dates) > 0:
            actual_date = past_dates[-1]
            return actual_date, df.loc[actual_date, price_type]

    return None, None


def find_next_trading_day(
    df: pd.DataFrame,
    target_date: datetime,
    price_type: str = 'Open'
) -> Tuple[Optional[datetime], Optional[float]]:
    """
    Find the next trading day AFTER the target date (excludes the target date itself)

    Args:
        df: DataFrame with stock data
        target_date: Target date (will find the day after this)
        price_type: 'Open' or 'Close' price to return

    Returns:
        Tuple of (actual_date, price) or (None, None)
    """
    target = pd.Timestamp(target_date)

    # Find dates strictly after target
    future_dates = df.index[df.index > target]
    if len(future_dates) > 0:
        actual_date = future_dates[0]
        return actual_date, df.loc[actual_date, price_type]

    return None, None


def calculate_return(
    ticker: str,
    buy_date: str,
    sell_date: str,
    cache_dir: str
) -> Dict:
    """
    Calculate return for a single ticker

    Args:
        ticker: Stock ticker symbol
        buy_date: Buy date (YYYY-MM-DD)
        sell_date: Sell date (YYYY-MM-DD)
        cache_dir: Path to cache directory

    Returns:
        Dictionary with return calculation results
    """
    result = {
        'ticker': ticker,
        'buy_date': buy_date,
        'sell_date': sell_date,
        'actual_buy_date': None,
        'actual_sell_date': None,
        'buy_price': None,
        'sell_price': None,
        'return_pct': None,
        'max_gain_pct': None,
        'max_drawdown_pct': None,
        'status': 'success',
        'error': None
    }

    df = load_ticker_data(ticker, cache_dir)

    if df is None:
        result['status'] = 'error'
        result['error'] = f"Ticker {ticker} not found in cache"
        return result

    buy_dt = datetime.strptime(buy_date, '%Y-%m-%d')
    sell_dt = datetime.strptime(sell_date, '%Y-%m-%d')

    # Find actual trading days
    # Buy at next trading day's open price (day after buy_date)
    actual_buy_date, buy_price = find_next_trading_day(df, buy_dt, 'Open')
    # Sell at sell_date's open price
    actual_sell_date, sell_price = find_nearest_trading_day(df, sell_dt, 'backward', 'Open')

    if buy_price is None:
        result['status'] = 'error'
        result['error'] = f"No trading data available on or after {buy_date}"
        return result

    if sell_price is None:
        result['status'] = 'error'
        result['error'] = f"No trading data available on or before {sell_date}"
        return result

    if actual_buy_date >= actual_sell_date:
        result['status'] = 'error'
        result['error'] = "Buy date must be before sell date"
        return result

    # Calculate return
    return_pct = ((sell_price - buy_price) / buy_price) * 100

    # Calculate max gain and max drawdown during holding period
    holding_period = df.loc[actual_buy_date:actual_sell_date]
    if not holding_period.empty:
        # Max gain: highest high relative to buy price
        max_high = holding_period['High'].max()
        max_gain_pct = ((max_high - buy_price) / buy_price) * 100

        # Max drawdown: lowest low relative to buy price
        min_low = holding_period['Low'].min()
        max_drawdown_pct = ((min_low - buy_price) / buy_price) * 100
    else:
        max_gain_pct = return_pct
        max_drawdown_pct = return_pct

    result['actual_buy_date'] = actual_buy_date.strftime('%Y-%m-%d')
    result['actual_sell_date'] = actual_sell_date.strftime('%Y-%m-%d')
    result['buy_price'] = round(buy_price, 4)
    result['sell_price'] = round(sell_price, 4)
    result['return_pct'] = round(return_pct, 2)
    result['max_gain_pct'] = round(max_gain_pct, 2)
    result['max_drawdown_pct'] = round(max_drawdown_pct, 2)

    return result


def calculate_returns(
    tickers: List[str],
    buy_date: str,
    sell_date: str
) -> pd.DataFrame:
    """
    Calculate returns for multiple tickers

    Args:
        tickers: List of stock ticker symbols
        buy_date: Buy date (YYYY-MM-DD)
        sell_date: Sell date (YYYY-MM-DD)

    Returns:
        DataFrame with return calculations for all tickers
    """
    cache_dir = get_cache_dir()
    results = []

    for ticker in tickers:
        result = calculate_return(ticker.upper().strip(), buy_date, sell_date, cache_dir)
        results.append(result)

    df = pd.DataFrame(results)
    return df


def print_results(df: pd.DataFrame) -> None:
    """Print results in a formatted table"""
    print("\n" + "=" * 80)
    print("RETURN CALCULATION RESULTS")
    print("=" * 80)

    success_df = df[df['status'] == 'success']
    error_df = df[df['status'] == 'error']

    if not success_df.empty:
        print(f"\nSuccessful calculations: {len(success_df)}")
        print("-" * 110)
        print(f"{'Ticker':<8} {'Buy Date':<12} {'Sell Date':<12} {'Buy Price':>10} {'Sell Price':>10} {'Return %':>10} {'Max Gain':>10} {'Max DD':>10}")
        print("-" * 110)

        for _, row in success_df.iterrows():
            print(f"{row['ticker']:<8} {row['actual_buy_date']:<12} {row['actual_sell_date']:<12} "
                  f"{row['buy_price']:>10.2f} {row['sell_price']:>10.2f} {row['return_pct']:>+10.2f}% "
                  f"{row['max_gain_pct']:>+9.2f}% {row['max_drawdown_pct']:>+9.2f}%")

        print("-" * 110)

        # Summary statistics
        avg_return = success_df['return_pct'].mean()
        avg_max_gain = success_df['max_gain_pct'].mean()
        avg_max_dd = success_df['max_drawdown_pct'].mean()
        winners = (success_df['return_pct'] > 0).sum()
        losers = (success_df['return_pct'] < 0).sum()

        print(f"\nSummary:")
        print(f"  Average Return: {avg_return:+.2f}%")
        print(f"  Average Max Gain: {avg_max_gain:+.2f}%")
        print(f"  Average Max Drawdown: {avg_max_dd:+.2f}%")
        print(f"  Winners/Losers: {winners}/{losers}")
        print(f"  Best:  {success_df.loc[success_df['return_pct'].idxmax(), 'ticker']} ({success_df['return_pct'].max():+.2f}%)")
        print(f"  Worst: {success_df.loc[success_df['return_pct'].idxmin(), 'ticker']} ({success_df['return_pct'].min():+.2f}%)")

    if not error_df.empty:
        print(f"\nFailed calculations: {len(error_df)}")
        print("-" * 80)
        for _, row in error_df.iterrows():
            print(f"  {row['ticker']}: {row['error']}")

    print("\n" + "=" * 80)


def main():
    """Main entry point for command line usage"""
    if len(sys.argv) < 4:
        print("Usage: python return_calculator.py <tickers> <buy_date> <sell_date>")
        print("")
        print("Arguments:")
        print("  tickers   - Comma-separated list of stock symbols (e.g., AAPL,GOOGL,MSFT)")
        print("  buy_date  - Buy date in YYYY-MM-DD format")
        print("  sell_date - Sell date in YYYY-MM-DD format")
        print("")
        print("Example:")
        print("  python return_calculator.py AAPL,GOOGL,MSFT 2024-01-02 2024-12-31")
        sys.exit(1)

    tickers_str = sys.argv[1]
    buy_date = sys.argv[2]
    sell_date = sys.argv[3]

    # Parse tickers
    tickers = [t.strip() for t in tickers_str.split(',')]

    # Validate dates
    try:
        datetime.strptime(buy_date, '%Y-%m-%d')
        datetime.strptime(sell_date, '%Y-%m-%d')
    except ValueError:
        print("Error: Dates must be in YYYY-MM-DD format")
        sys.exit(1)

    print(f"\nCalculating returns for {len(tickers)} ticker(s)")
    print(f"Buy date:  {buy_date}")
    print(f"Sell date: {sell_date}")

    # Calculate returns
    results = calculate_returns(tickers, buy_date, sell_date)

    # Print results
    print_results(results)

    return results


if __name__ == '__main__':
    main()
