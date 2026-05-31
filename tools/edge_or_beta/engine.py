import os
import uuid
from datetime import datetime
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from modules.data.data_loader import DataLoader
from modules.data.expiry_calculator import ExpiryCalculator
from modules.features.feature_pipeline import FeaturePipeline
from modules.features.technical_indicators import TechnicalIndicators
from modules.evaluation.metrics import calculate_composite_score

from .rules import evaluate_signal
from .nulls import run_random_stock_null, run_random_etf_timing_null
from .decompose import calculate_capm_decomposition
from .verdict import resolve_verdict
from .schemas import validate_result_bundle

def _find_loc(df: pd.DataFrame, date: pd.Timestamp) -> Optional[int]:
    """Return integer iloc for date, or the iloc of the nearest earlier date."""
    norm = pd.Timestamp(date).tz_localize(None)
    candidates = df.index[df.index <= norm]
    if len(candidates) == 0:
        return None
    return df.index.get_loc(candidates[-1])

def _get_entry(df: pd.DataFrame, expiry_date: pd.Timestamp) -> Tuple[Optional[pd.Timestamp], Optional[float]]:
    """Get entry date and price (Open on expiry date)."""
    loc = _find_loc(df, expiry_date)
    if loc is None:
        return None, None
    row = df.iloc[loc]
    entry_date = df.index[loc]
    open_price = row.get("Open", np.nan)
    if pd.isna(open_price) or open_price <= 0:
        return None, None
    return entry_date, float(open_price)

def _get_exit(df: pd.DataFrame, entry_date: pd.Timestamp, hold_days: int) -> Tuple[Optional[pd.Timestamp], Optional[float]]:
    """Get exit date and price (Close after hold_days trading days)."""
    entry_loc = _find_loc(df, entry_date)
    if entry_loc is None:
        return None, None
    end_loc = min(entry_loc + hold_days, len(df) - 1)
    if end_loc <= entry_loc:
        return None, None
    row = df.iloc[end_loc]
    close_price = row.get("Close", np.nan)
    if pd.isna(close_price) or close_price <= 0:
        return None, None
    return df.index[end_loc], float(close_price)

def evaluate_rule(
    rule_id: str,
    params: dict,
    universe: Optional[List[str]] = None,
    start: str = "2015-01-01",
    end: str = "2026-05-22",
    hold_days: int = 6,
    commission: float = 0.002,
    top_k: int = 3,
    n_null: int = 1000,
    seed: int = 42,
    cache_dir: str = "cache"
) -> Dict:
    """
    Evaluates a technical trading rule preset against three baselines:
    1. Random stock selections on same dates
    2. Random ETF timing
    3. Passive buy-and-hold SPY
    
    Returns:
        Dict: A serializable ResultBundle matching the schema.
    """
    # 1. Load SPY data
    spy_path = os.path.join(cache_dir, "constituent_data", "SPY.csv")
    if not os.path.exists(spy_path):
        raise FileNotFoundError(f"SPY benchmark file not found at {spy_path}")
    
    spy_df = pd.read_csv(spy_path, parse_dates=["Date"])
    spy_df["Date"] = pd.to_datetime(spy_df["Date"]).dt.tz_localize(None)
    spy_df = spy_df.set_index("Date").sort_index()

    # Guard: the buy-and-hold SPY baseline and per-trade CAPM matching require SPY price
    # coverage across the full requested window. Without it the engine would silently
    # substitute 0.0 SPY returns and a truncated equity curve, corrupting the verdict.
    # A 7-day grace absorbs a start/end that lands on a weekend or market holiday.
    req_start = pd.to_datetime(start)
    req_end = pd.to_datetime(end)
    if spy_df.index.min() > req_start + pd.Timedelta(days=7):
        raise ValueError(
            f"SPY benchmark data starts {spy_df.index.min().date()}, after the requested "
            f"start {req_start.date()}. Narrow the date range or extend {spy_path}."
        )
    if spy_df.index.max() < req_end - pd.Timedelta(days=7):
        raise ValueError(
            f"SPY benchmark data ends {spy_df.index.max().date()}, before the requested "
            f"end {req_end.date()}. Narrow the date range or extend {spy_path}."
        )

    # 2. Load Ticker Data
    loader = DataLoader(
        cache_dir=os.path.join(cache_dir, "constituent_data"),
        auto_update=False
    )
    
    all_tickers = universe if universe is not None else loader.get_sp500_tickers()
    # Filter out SPY from stock universe to avoid self-selection
    all_tickers = [t for t in all_tickers if t != "SPY"]
    
    pipeline = FeaturePipeline([TechnicalIndicators()])
    
    all_data: Dict[str, pd.DataFrame] = {}
    for ticker in all_tickers:
        try:
            df = loader.load_ticker(ticker, validate=False)
            if df is None or df.empty:
                continue
            df = pipeline.transform(df)
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df = df.sort_index()
            all_data[ticker] = df
        except Exception:
            pass
            
    # 3. Generate Expiry dates
    expiry_dates = ExpiryCalculator.generate_expiry_dates(start, end)
    
    # Define ranking logic based on rule
    # Default is lower is better (RSI oversold)
    rank_direction = "lower_is_better"
    if rule_id in ["ma_crossover", "momentum_12_1"]:
        rank_direction = "higher_is_better"
        
    # 4. Main Backtest Loop
    trades = []
    open_positions = [] # Elements are {"exit_date": Timestamp}
    
    max_concurrent = 10 # Enforce maximum concurrent positions
    
    for expiry_dt in expiry_dates:
        # Close out finished positions
        open_positions = [p for p in open_positions if p["exit_date"] > expiry_dt]
        slots_available = max_concurrent - len(open_positions)
        
        if slots_available <= 0:
            continue
            
        signals = []
        for ticker, df in all_data.items():
            expiry_loc = _find_loc(df, expiry_dt)
            if expiry_loc is None or expiry_loc < 1:
                continue
                
            signal_loc = expiry_loc - 1
            signal_dt = df.index[signal_loc]
            
            # Check if signal triggers
            if evaluate_signal(rule_id, df, signal_loc, params):
                # Calculate ranking value
                rank_val = 0.0
                if rule_id in ["capstone_expiry_rule", "rsi_oversold"]:
                    rank_val = float(df.iloc[signal_loc].get("RSI", 50.0))
                elif rule_id == "ma_crossover":
                    # Rank by SMA_9 distance above SMA_50
                    sma_9 = df.iloc[signal_loc].get("SMA_9", 0.0)
                    sma_50 = df.iloc[signal_loc].get("SMA_50", 1.0)
                    rank_val = float((sma_9 - sma_50) / (sma_50 if sma_50 != 0 else 1.0))
                elif rule_id == "consecutive_red":
                    rank_val = float(df.iloc[signal_loc].get("Consecutive_Count", 0.0))
                elif rule_id == "momentum_12_1":
                    curr_p = df.iloc[signal_loc].get("Close", 1.0)
                    past_p = df.iloc[max(0, signal_loc - 252)].get("Close", 1.0)
                    rank_val = float(curr_p / past_p - 1.0)
                    
                signals.append({
                    "ticker": ticker,
                    "signal_date": signal_dt,
                    "rank_value": rank_val,
                    "rsi_at_signal": float(df.iloc[signal_loc].get("RSI", np.nan)),
                    "consecutive_at_signal": float(df.iloc[signal_loc].get("Consecutive_Count", np.nan))
                })
                
        if not signals:
            continue
            
        # Rank and select top K
        is_reverse = (rank_direction == "higher_is_better")
        sorted_signals = sorted(signals, key=lambda x: x["rank_value"], reverse=is_reverse)
        top_signals = sorted_signals[:min(top_k, slots_available, len(sorted_signals))]
        
        for sig in top_signals:
            ticker = sig["ticker"]
            df = all_data[ticker]
            
            entry_dt, entry_price = _get_entry(df, expiry_dt)
            if entry_dt is None or entry_price is None:
                continue
                
            exit_dt, exit_price = _get_exit(df, entry_dt, hold_days)
            if exit_dt is None or exit_price is None:
                continue
                
            gross_r = (exit_price - entry_price) / entry_price
            net_r = gross_r - commission
            
            # Matched SPY return
            spy_entry_loc = _find_loc(spy_df, entry_dt)
            spy_exit_loc = _find_loc(spy_df, exit_dt)
            if spy_entry_loc is not None and spy_exit_loc is not None:
                spy_entry_price = spy_df.iloc[spy_entry_loc]["Open"]
                spy_exit_price = spy_df.iloc[spy_exit_loc]["Close"]
                spy_r = float((spy_exit_price - spy_entry_price) / spy_entry_price)
            else:
                spy_r = 0.0
                
            trades.append({
                "ticker": ticker,
                "signal_date": sig["signal_date"],
                "entry_date": entry_dt,
                "exit_date": exit_dt,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "gross_return": float(gross_r),
                "net_return": float(net_r),
                "spy_return": spy_r,
                "rsi_at_signal": sig["rsi_at_signal"],
                "consecutive_at_signal": sig["consecutive_at_signal"]
            })
            open_positions.append({"exit_date": exit_dt})

    # Sort trades chronologically
    trades = sorted(trades, key=lambda x: x["entry_date"])
    
    # 5. Calculate Years and CAGRs
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)
    years = (end_ts - start_ts).days / 365.25
    
    # Compounding strategy returns
    if trades:
        trade_df = pd.DataFrame(trades)
        cycle_returns = trade_df.groupby("entry_date")["net_return"].mean()
        strategy_cumulative = np.prod(1.0 + cycle_returns.values)
        strategy_cagr = float((strategy_cumulative) ** (1.0 / years) - 1.0) if years > 0 else 0.0
        
        # Win Rate
        win_rate = float((trade_df["net_return"] > 0).mean())
        
        # Profit Factor
        pos_sum = trade_df[trade_df["net_return"] > 0]["net_return"].sum()
        neg_sum = abs(trade_df[trade_df["net_return"] < 0]["net_return"].sum())
        profit_factor = float(pos_sum / neg_sum) if neg_sum > 0 else (float("inf") if pos_sum > 0 else 1.0)
        
        # Sharpe (standard scaling over trade cycles)
        excess_returns = cycle_returns.values - 0.0 # assumes 0 risk-free
        if len(excess_returns) > 1 and np.std(excess_returns) > 0:
            sharpe = float((np.mean(excess_returns) / np.std(excess_returns)) * np.sqrt(12)) # assume 12 cycles/year
        else:
            sharpe = 0.0
            
        # Max Drawdown
        cum_equity = np.cumprod(1.0 + cycle_returns.values)
        running_max = np.maximum.accumulate(cum_equity)
        drawdowns = (cum_equity - running_max) / running_max
        max_drawdown = float(abs(drawdowns.min())) if len(drawdowns) > 0 else 0.0
        
        # Create strategy equity curve points
        equity_curve = [{"date": start_ts.strftime("%Y-%m-%d"), "value": 1.0}]
        running_val = 1.0
        for dt, ret in cycle_returns.items():
            running_val *= (1.0 + ret)
            equity_curve.append({
                "date": dt.strftime("%Y-%m-%d"),
                "value": float(running_val)
            })
    else:
        strategy_cagr = 0.0
        win_rate = 0.0
        profit_factor = 1.0
        sharpe = 0.0
        max_drawdown = 0.0
        equity_curve = [{"date": start_ts.strftime("%Y-%m-%d"), "value": 1.0}]
        
    # 6. SPY Buy and Hold Benchmark
    spy_filtered = spy_df[(spy_df.index >= start_ts) & (spy_df.index <= end_ts)]
    if not spy_filtered.empty:
        spy_start_price = spy_filtered.iloc[0]["Close"]
        spy_end_price = spy_filtered.iloc[-1]["Close"]
        spy_cumulative = spy_end_price / spy_start_price
        spy_cagr = float((spy_cumulative) ** (1.0 / years) - 1.0) if years > 0 else 0.0
        
        spy_daily_returns = spy_filtered["Close"].pct_change().dropna().values
        if len(spy_daily_returns) > 0 and np.std(spy_daily_returns) > 0:
            spy_sharpe = float((np.mean(spy_daily_returns) / np.std(spy_daily_returns)) * np.sqrt(252))
        else:
            spy_sharpe = 0.0
            
        spy_cum_equity = (1.0 + spy_daily_returns).cumprod()
        spy_running_max = np.maximum.accumulate(spy_cum_equity)
        spy_drawdowns = (spy_cum_equity - spy_running_max) / spy_running_max
        spy_max_dd = float(abs(spy_drawdowns.min())) if len(spy_drawdowns) > 0 else 0.0
        
        # Compile SPY benchmark equity curve on the same dates as strategy equity curve
        spy_equity_curve = [{"date": start_ts.strftime("%Y-%m-%d"), "value": 1.0}]
        for pt in equity_curve[1:]:
            dt_ts = pd.to_datetime(pt["date"])
            spy_loc = _find_loc(spy_filtered, dt_ts)
            if spy_loc is not None:
                curr_spy_val = float(spy_filtered.iloc[spy_loc]["Close"] / spy_start_price)
                spy_equity_curve.append({
                    "date": pt["date"],
                    "value": curr_spy_val
                })
    else:
        spy_cagr = 0.0
        spy_sharpe = 0.0
        spy_max_dd = 0.0
        spy_equity_curve = [{"date": start_ts.strftime("%Y-%m-%d"), "value": 1.0}]
        
    # 7. Run Monte Carlo Null Tests
    num_cycles = len(trade_df.groupby("entry_date")) if trades else 0
    if trades and num_cycles > 0:
        p_select, mean_select, median_select, dist_select = run_random_stock_null(
            trades=trades,
            all_data=all_data,
            hold_days=hold_days,
            commission=commission,
            years=years,
            strategy_cagr=strategy_cagr,
            n_null=n_null,
            seed=seed
        )
        
        p_timing, mean_timing, median_timing, dist_timing = run_random_etf_timing_null(
            spy_df=spy_df,
            num_cycles=num_cycles,
            hold_days=hold_days,
            commission=commission,
            years=years,
            strategy_cagr=strategy_cagr,
            start_date=start,
            end_date=end,
            n_null=n_null,
            seed=seed
        )
        
        # 8. CAPM Decomposition
        strategy_trade_returns = [t["net_return"] for t in trades]
        spy_trade_returns = [t["spy_return"] for t in trades]
        decomp = calculate_capm_decomposition(
            strategy_returns=strategy_trade_returns,
            spy_returns=spy_trade_returns,
            strategy_cagr=strategy_cagr,
            spy_cagr=spy_cagr,
            num_trades=len(trades)
        )
    else:
        p_select, mean_select, median_select, dist_select = 1.0, 0.0, 0.0, [0.0] * n_null
        p_timing, mean_timing, median_timing, dist_timing = 1.0, 0.0, 0.0, [0.0] * n_null
        decomp = {
            "alpha_annualized": 0.0,
            "alpha_t": 0.0,
            "alpha_p": 1.0,
            "beta": 1.0,
            "beta_share": None,
            "method": "trade_window_capm"
        }
        
    # 9. Compute Composite Score
    composite = calculate_composite_score(profit_factor, sharpe, win_rate, max_drawdown)
    
    # 10. Verdict Resolution
    verdict = resolve_verdict(
        trade_count=len(trades),
        p_select=p_select,
        p_timing=p_timing,
        strategy_cagr=strategy_cagr,
        spy_cagr=spy_cagr,
        alpha_t=decomp["alpha_t"],
        alpha_annualized=decomp["alpha_annualized"],
        beta_share=decomp["beta_share"]
    )
    
    # 11. Compile ResultBundle
    bundle = {
        "schema_version": "1.0",
        "run_id": str(uuid.uuid4()),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "rule": {
            "id": rule_id,
            "label": {
                "capstone_expiry_rule": "RSI oversold + consecutive red candles near expiry",
                "rsi_oversold": "Simple RSI oversold",
                "ma_crossover": "Moving average crossover",
                "consecutive_red": "N consecutive red candles",
                "momentum_12_1": "12-1 momentum"
            }.get(rule_id, rule_id),
            "params": params
        },
        "settings": {
            "universe": "sp500_current_cache",
            "start": start,
            "end": end,
            "hold_days": hold_days,
            "commission": commission,
            "top_k": top_k,
            "n_null": n_null,
            "seed": seed
        },
        "data_quality": {
            "tickers_requested": len(all_tickers),
            "tickers_used": len(all_data),
            "trades": len(trades),
            "warnings": [
                "Current S&P 500 cache is not point-in-time; survivorship bias may remain."
            ]
        },
        "strategy": {
            "cagr": float(strategy_cagr),
            "sharpe": float(sharpe),
            "win_rate": float(win_rate),
            "profit_factor": float(profit_factor),
            "max_drawdown": float(max_drawdown),
            "composite": float(composite),
            "equity_curve": equity_curve
        },
        "benchmarks": {
            "random_stock_same_dates": {
                "metric": "cagr",
                "mean": float(mean_select),
                "median": float(median_select),
                "p_value": float(p_select),
                "percentile": float(100.0 - p_select * 100.0),
                "distribution": [float(val) for val in dist_select]
            },
            "random_etf_timing": {
                "metric": "cagr",
                "mean": float(mean_timing),
                "median": float(median_timing),
                "p_value": float(p_timing),
                "percentile": float(100.0 - p_timing * 100.0),
                "distribution": [float(val) for val in dist_timing]
            },
            "buy_hold_spy": {
                "cagr": float(spy_cagr),
                "sharpe": float(spy_sharpe),
                "max_drawdown": float(spy_max_dd),
                "equity_curve": spy_equity_curve
            }
        },
        "decomposition": decomp,
        "verdict": verdict
    }
    
    # 12. Validate
    validate_result_bundle(bundle)
    
    return bundle
