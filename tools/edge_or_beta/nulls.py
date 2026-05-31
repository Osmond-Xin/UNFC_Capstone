import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional

def run_random_stock_null(
    trades: List[Dict],
    all_data: Dict[str, pd.DataFrame],
    hold_days: int,
    commission: float,
    years: float,
    strategy_cagr: float,
    n_null: int = 1000,
    seed: int = 42
) -> Tuple[float, float, float, List[float]]:
    """
    Random Stock Same-Dates Null.
    Tests stock-selection skill by keeping the actual entry/exit dates
    but replacing tickers with randomly selected S&P 500 constituents.
    
    Returns:
        Tuple: (p_select, mean_cagr, median_cagr, null_distribution)
    """
    rng = np.random.default_rng(seed)
    
    # 1. Precompute eligible tickers for each trade to optimize simulation speed
    trade_eligibility = []
    for trade in trades:
        entry_dt = trade["entry_date"]
        exit_dt = trade["exit_date"]
        
        eligible = []
        for ticker, df in all_data.items():
            if entry_dt in df.index and exit_dt in df.index:
                ep = df.loc[entry_dt, "Open"]
                xp = df.loc[exit_dt, "Close"]
                if not (pd.isna(ep) or pd.isna(xp) or ep <= 0 or xp <= 0):
                    eligible.append(ticker)
        
        trade_eligibility.append(eligible)

    null_cagrs = []
    
    # 2. Run Monte Carlo iterations
    for _ in range(n_null):
        # Generate random trades for this iteration
        sim_trades = []
        for i, trade in enumerate(trades):
            eligible = trade_eligibility[i]
            if not eligible:
                continue
            
            # Select random ticker
            rand_ticker = rng.choice(eligible)
            df = all_data[rand_ticker]
            
            entry_price = float(df.loc[trade["entry_date"], "Open"])
            exit_price = float(df.loc[trade["exit_date"], "Close"])
            
            gross = (exit_price - entry_price) / entry_price
            net = gross - commission
            
            sim_trades.append({
                "entry_date": trade["entry_date"],
                "net_return": net
            })
            
        if not sim_trades:
            null_cagrs.append(0.0)
            continue
            
        # Group by entry date and compute mean return per cycle
        df_sim = pd.DataFrame(sim_trades)
        cycle_returns = df_sim.groupby("entry_date")["net_return"].mean()
        
        # Compound returns
        end_val = np.prod(1.0 + cycle_returns.values)
        cagr = (end_val) ** (1.0 / years) - 1.0 if years > 0 else 0.0
        null_cagrs.append(float(cagr))
        
    null_cagrs = sorted(null_cagrs)
    
    # Calculate p-value: fraction of null CAGRs >= strategy CAGR
    p_select = float(np.mean(np.array(null_cagrs) >= strategy_cagr))
    mean_cagr = float(np.mean(null_cagrs))
    median_cagr = float(np.median(null_cagrs))
    
    return p_select, mean_cagr, median_cagr, null_cagrs


def run_random_etf_timing_null(
    spy_df: pd.DataFrame,
    num_cycles: int,
    hold_days: int,
    commission: float,
    years: float,
    strategy_cagr: float,
    start_date: str,
    end_date: str,
    n_null: int = 1000,
    seed: int = 42
) -> Tuple[float, float, float, List[float]]:
    """
    Random ETF Timing Null.
    Tests market timing skill by selecting random entry dates in the analysis period,
    buying SPY, holding for hold_days, and compounding returns.
    
    Returns:
        Tuple: (p_timing, mean_cagr, median_cagr, null_distribution)
    """
    rng = np.random.default_rng(seed)
    
    # Ensure index is sorted and filtered to range
    spy_df = spy_df.sort_index()
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    spy_range = spy_df[(spy_df.index >= start_ts) & (spy_df.index <= end_ts)]
    
    # Generate list of valid index positions for entries
    # Exclude the last hold_days to avoid running out of bounds
    max_idx = len(spy_range) - hold_days - 1
    if max_idx <= 0:
        return 1.0, 0.0, 0.0, [0.0] * n_null
        
    null_cagrs = []
    
    for _ in range(n_null):
        # Draw random index positions
        # In a real timing simulation, we select num_cycles random entry days
        rand_indices = rng.choice(max_idx, size=num_cycles, replace=True)
        
        cycle_returns = []
        for idx in rand_indices:
            row_entry = spy_range.iloc[idx]
            row_exit = spy_range.iloc[idx + hold_days]
            
            entry_price = float(row_entry["Open"])
            exit_price = float(row_exit["Close"])
            
            gross = (exit_price - entry_price) / entry_price
            net = gross - commission
            cycle_returns.append(net)
            
        end_val = np.prod(1.0 + np.array(cycle_returns))
        cagr = (end_val) ** (1.0 / years) - 1.0 if years > 0 else 0.0
        null_cagrs.append(float(cagr))
        
    null_cagrs = sorted(null_cagrs)
    p_timing = float(np.mean(np.array(null_cagrs) >= strategy_cagr))
    mean_cagr = float(np.mean(null_cagrs))
    median_cagr = float(np.median(null_cagrs))
    
    return p_timing, mean_cagr, median_cagr, null_cagrs
