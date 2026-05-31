import pandas as pd
import numpy as np

def evaluate_signal(rule_id: str, df: pd.DataFrame, idx_pos: int, params: dict) -> bool:
    """
    Evaluates whether the rule is triggered at the given index position.
    
    Args:
        rule_id (str): The ID of the rule preset
        df (pd.DataFrame): DataFrame with technical indicators pre-computed
        idx_pos (int): The integer index position in the DataFrame to check
        params (dict): Parameters for the rule
        
    Returns:
        bool: True if signal is triggered, False otherwise
    """
    if idx_pos < 0 or idx_pos >= len(df):
        return False
        
    if rule_id == "capstone_expiry_rule":
        rsi_threshold = params.get("rsi_threshold", 30)
        min_consecutive = params.get("min_consecutive", 3)
        
        rsi = df.iloc[idx_pos].get("RSI", np.nan)
        consec_count = df.iloc[idx_pos].get("Consecutive_Count", np.nan)
        consec_dir = df.iloc[idx_pos].get("Consecutive_Direction", np.nan)
        
        if pd.isna(rsi) or pd.isna(consec_count) or pd.isna(consec_dir):
            return False
            
        return rsi < rsi_threshold and consec_count >= min_consecutive and consec_dir == -1
        
    elif rule_id == "rsi_oversold":
        rsi_threshold = params.get("rsi_threshold", 30)
        
        rsi = df.iloc[idx_pos].get("RSI", np.nan)
        if pd.isna(rsi):
            return False
            
        return rsi < rsi_threshold
        
    elif rule_id == "ma_crossover":
        short_period = params.get("short_period", 9)
        long_period = params.get("long_period", 50)
        
        short_ma = df.iloc[idx_pos].get(f"SMA_{short_period}", np.nan)
        long_ma = df.iloc[idx_pos].get(f"SMA_{long_period}", np.nan)
        
        if pd.isna(short_ma) or pd.isna(long_ma):
            return False
            
        return short_ma > long_ma
        
    elif rule_id == "consecutive_red":
        min_consecutive = params.get("min_consecutive", 4)
        
        consec_count = df.iloc[idx_pos].get("Consecutive_Count", np.nan)
        consec_dir = df.iloc[idx_pos].get("Consecutive_Direction", np.nan)
        
        if pd.isna(consec_count) or pd.isna(consec_dir):
            return False
            
        return consec_count >= min_consecutive and consec_dir == -1
        
    elif rule_id == "momentum_12_1":
        # Price 12 months ago vs 1 month ago. Since we are dealing with daily data,
        # let's approximate with SMA_9 > SMA_50 or positive 252-day return.
        # Let's check 252-day price return.
        lookback = params.get("lookback", 252)
        if idx_pos < lookback:
            return False
        
        curr_price = df.iloc[idx_pos].get("Close", np.nan)
        past_price = df.iloc[idx_pos - lookback].get("Close", np.nan)
        
        if pd.isna(curr_price) or pd.isna(past_price) or past_price <= 0:
            return False
            
        return curr_price > past_price

    else:
        raise ValueError(f"Unknown rule_id: {rule_id}")
