"""
Metrics and Statistical Analysis

This module provides statistical analysis functions including:
- Correlation analysis
- Statistical significance testing
- Pattern summarization

Usage:
    from modules.evaluation import calculate_correlation_matrix

    corr_matrix = calculate_correlation_matrix(df, features)
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from scipy import stats


def calculate_correlation_matrix(
    df: pd.DataFrame,
    features: List[str] = None,
    method: str = 'pearson'
) -> pd.DataFrame:
    """
    Calculate correlation matrix for features

    Args:
        df (pd.DataFrame): DataFrame with feature columns
        features (list, optional): List of features to include. If None, uses all numeric columns
        method (str): Correlation method ('pearson', 'spearman', 'kendall')

    Returns:
        pd.DataFrame: Correlation matrix
    """
    if features is None:
        # Use all numeric columns
        features = df.select_dtypes(include=[np.number]).columns.tolist()

    # Filter to only existing columns
    features = [f for f in features if f in df.columns]

    if not features:
        return pd.DataFrame()

    # Calculate correlation
    corr_matrix = df[features].corr(method=method)

    return corr_matrix


def statistical_significance_test(
    group1: np.ndarray,
    group2: np.ndarray,
    test_type: str = 't-test'
) -> Dict:
    """
    Perform statistical significance test between two groups

    Args:
        group1 (array): First group of values
        group2 (array): Second group of values
        test_type (str): Type of test ('t-test', 'mannwhitneyu', 'ks')

    Returns:
        dict: Test results including statistic and p-value
    """
    # Remove NaN values
    group1 = group1[~np.isnan(group1)]
    group2 = group2[~np.isnan(group2)]

    if len(group1) == 0 or len(group2) == 0:
        return {
            'test_type': test_type,
            'statistic': np.nan,
            'p_value': np.nan,
            'significant': False,
            'error': 'Empty group(s)'
        }

    try:
        if test_type == 't-test':
            # Independent t-test
            statistic, p_value = stats.ttest_ind(group1, group2)

        elif test_type == 'mannwhitneyu':
            # Mann-Whitney U test (non-parametric)
            statistic, p_value = stats.mannwhitneyu(group1, group2)

        elif test_type == 'ks':
            # Kolmogorov-Smirnov test
            statistic, p_value = stats.ks_2samp(group1, group2)

        else:
            raise ValueError(f"Unknown test_type: {test_type}")

        return {
            'test_type': test_type,
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'group1_mean': np.mean(group1),
            'group2_mean': np.mean(group2),
            'group1_size': len(group1),
            'group2_size': len(group2)
        }

    except Exception as e:
        return {
            'test_type': test_type,
            'statistic': np.nan,
            'p_value': np.nan,
            'significant': False,
            'error': str(e)
        }


def summarize_patterns(
    pattern_results: pd.DataFrame,
    group_by: str = 'pattern_name'
) -> pd.DataFrame:
    """
    Summarize pattern matching results

    Args:
        pattern_results (pd.DataFrame): DataFrame with pattern match results
            Expected columns: pattern_name, ticker, signal, return_pct, etc.
        group_by (str): Column to group by (default: 'pattern_name')

    Returns:
        pd.DataFrame: Summary statistics for each pattern
    """
    if pattern_results.empty:
        return pd.DataFrame()

    # Group by pattern
    summary = pattern_results.groupby(group_by).agg({
        'ticker': 'count',  # Number of matches
        'return_pct': ['mean', 'median', 'std', 'min', 'max']
    })

    # Flatten column names
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]

    # Rename for clarity
    summary = summary.rename(columns={
        'ticker_count': 'num_matches'
    })

    # Calculate win rate if signal and return_pct are available
    if 'signal' in pattern_results.columns and 'return_pct' in pattern_results.columns:
        def calc_win_rate(group):
            # Positive return for long signal (1), negative return for short signal (-1)
            correct = ((group['signal'] == 1) & (group['return_pct'] > 0)) | \
                     ((group['signal'] == -1) & (group['return_pct'] < 0))
            return (correct.sum() / len(group)) * 100 if len(group) > 0 else 0

        summary['win_rate_pct'] = pattern_results.groupby(group_by).apply(calc_win_rate)

    # Reset index to make group_by a column
    summary = summary.reset_index()

    return summary


def calculate_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sharpe ratio

    Args:
        returns (array): Array of returns (in percentage or decimal)
        risk_free_rate (float): Risk-free rate (annualized)
        periods_per_year (int): Number of periods per year (252 for daily)

    Returns:
        float: Sharpe ratio
    """
    # Remove NaN values
    returns = returns[~np.isnan(returns)]

    if len(returns) == 0:
        return np.nan

    # Calculate excess returns
    excess_returns = returns - (risk_free_rate / periods_per_year)

    # Calculate Sharpe ratio
    if excess_returns.std() == 0:
        return np.nan

    sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(periods_per_year)

    return sharpe


def calculate_max_drawdown(
    returns: np.ndarray
) -> Tuple[float, int, int]:
    """
    Calculate maximum drawdown

    Args:
        returns (array): Array of returns

    Returns:
        tuple: (max_drawdown, start_idx, end_idx)
    """
    # Calculate cumulative returns
    cumulative = (1 + returns / 100).cumprod()

    # Calculate running maximum
    running_max = np.maximum.accumulate(cumulative)

    # Calculate drawdowns
    drawdowns = (cumulative - running_max) / running_max

    # Find maximum drawdown
    max_dd_idx = np.argmin(drawdowns)
    max_dd = drawdowns[max_dd_idx]

    # Find start of drawdown (peak before max drawdown)
    peak_idx = np.argmax(cumulative[:max_dd_idx + 1]) if max_dd_idx > 0 else 0

    return max_dd * 100, peak_idx, max_dd_idx


def calculate_composite_score(
    profit_factor: float,
    sharpe: float,
    win_rate: float,
    max_drawdown: float,
) -> float:
    """
    Compute the pre-registered capstone composite score (v4.1 weights).

    Formula:
        min(PF / 2.0, 1.0) × 0.35
      + min(max(Sharpe, 0) / 2.0, 1.0) × 0.30
      + win_rate × 0.20
      + (1 − min(|MaxDD| / 0.30, 1.0)) × 0.15

    Args:
        profit_factor: Gross profit / gross loss (≥ 0).
        sharpe:        Annualised Sharpe ratio (can be negative).
        win_rate:      Fraction of winning trades in [0, 1].
        max_drawdown:  Maximum drawdown as a fraction (e.g. −0.12 or 0.12;
                       the absolute value is used).

    Returns:
        Composite score in [0, 1].

    Example:
        >>> calculate_composite_score(1.5, 1.0, 0.6, 0.1)
        0.6325
    """
    pf_component = min(profit_factor / 2.0, 1.0) * 0.35
    sharpe_component = min(max(sharpe, 0.0) / 2.0, 1.0) * 0.30
    wr_component = win_rate * 0.20
    dd_component = (1.0 - min(abs(max_drawdown) / 0.30, 1.0)) * 0.15
    return pf_component + sharpe_component + wr_component + dd_component


def calculate_information_ratio(
    strategy_returns: np.ndarray,
    benchmark_returns: np.ndarray
) -> float:
    """
    Calculate Information Ratio

    IR = mean(active_return) / std(active_return)

    Args:
        strategy_returns (array): Strategy returns
        benchmark_returns (array): Benchmark returns

    Returns:
        float: Information ratio
    """
    # Calculate active returns
    active_returns = strategy_returns - benchmark_returns

    # Remove NaN
    active_returns = active_returns[~np.isnan(active_returns)]

    if len(active_returns) == 0 or active_returns.std() == 0:
        return np.nan

    ir = active_returns.mean() / active_returns.std()

    return ir
