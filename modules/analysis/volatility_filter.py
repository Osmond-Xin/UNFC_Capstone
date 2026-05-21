
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from ..data.expiry_calculator import ExpiryCalculator

class VolatilityFilter:
    """
    Filters stocks based on historical volatility on expiry days.
    Replicates logic from V1 Analysis:
    - Looks back at the last N monthly expiry dates
    - Identifies Top 10 Gainers and Bottom 10 Losers for each date
    - Returns the unique set of these 'Target Stocks'
    """
    
    def __init__(self, lookback_periods: int = 11, expiry_calculator=None):
        """
        Initialize the filter
        
        Args:
            lookback_periods (int): Number of past expiry dates to analyze (default: 11)
            expiry_calculator (class): Expiry calculator class to use (default: US ExpiryCalculator)
        """
        self.lookback_periods = lookback_periods
        self.expiry_calculator = expiry_calculator if expiry_calculator else ExpiryCalculator
        
    def get_target_universe(
        self, 
        stock_data: Dict[str, pd.DataFrame], 
        end_date: str
    ) -> List[str]:
        """
        Identify target stocks that have been significant movers on past expiry dates.
        
        Args:
            stock_data (dict): Dictionary of stock DataFrames
            end_date (str): The analysis date (cutoff for looking back)
            
        Returns:
            list: List of unique ticker symbols
        """
        # 1. Generate past expiry dates
        expiry_dates = self._get_past_expiries(end_date)
        
        top_performers = []
        bottom_performers = []
        
        print(f"   Filtering Universe: Analyzing last {len(expiry_dates)} expiry dates...")
        
        # 2. Analyze each date
        for date in expiry_dates:
            daily_performance = []
            
            # Convert date to UTC for index matching
            if date.tz is None:
                query_date = date.tz_localize('UTC')
            else:
                query_date = date.tz_convert('UTC')
            
            for ticker, df in stock_data.items():
                try:
                    # Find data for this specific date
                    if query_date in df.index:
                        row = df.loc[query_date]
                        open_price = row['Open']
                        close_price = row['Close']
                        
                        if open_price > 0:
                            change_pct = ((close_price - open_price) / open_price) * 100
                            daily_performance.append({
                                'Ticker': ticker,
                                'Change_Pct': change_pct
                            })
                except Exception:
                    continue
            
            if not daily_performance:
                continue
                
            # Convert to DataFrame for sorting
            perf_df = pd.DataFrame(daily_performance)
            perf_df = perf_df.sort_values('Change_Pct', ascending=False)
            
            # Get Top 10
            top_10 = perf_df.head(10)['Ticker'].tolist()
            top_performers.extend(top_10)
            
            # Get Bottom 10
            bottom_10 = perf_df.tail(10)['Ticker'].tolist()
            bottom_performers.extend(bottom_10)
            
        # 3. Unique set - Preserving Order (Top Winners First)
        # V1 logic implicitly puts top performers first if not sorted
        seen = set()
        target_universe = []
        
        # Prioritize Top Performers, then Bottom Performers
        # Also prioritize recent dates (already processed in order)
        full_list = top_performers + bottom_performers
        
        for ticker in full_list:
            if ticker not in seen:
                target_universe.append(ticker)
                seen.add(ticker)
        
        return target_universe

    def _get_past_expiries(self, end_date: str) -> List[str]:
        """
        Get the last N monthly expiry dates before end_date
        """
        # Generate enough candidate dates and filter
        # We start from 1.5 years ago to be safe
        start_date = (pd.Timestamp(end_date) - pd.DateOffset(months=self.lookback_periods + 6)).strftime('%Y-%m-%d')
        
        all_expiries = self.expiry_calculator.generate_expiry_dates(start_date, end_date)
        
        # Filter: strictly before end_date
        end_ts = pd.to_datetime(end_date)
        valid_expiries = [d for d in all_expiries if d < end_ts]
        
        # Taking the last N (e.g., 11)
        return valid_expiries[-self.lookback_periods:]
