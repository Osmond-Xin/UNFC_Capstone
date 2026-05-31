import numpy as np
from scipy import stats
from typing import List, Dict, Optional

def calculate_capm_decomposition(
    strategy_returns: List[float],
    spy_returns: List[float],
    strategy_cagr: float,
    spy_cagr: float,
    num_trades: int
) -> Dict:
    """
    Perform a trade-level CAPM regression:
        r_strategy = alpha + beta * r_spy + error
        
    Args:
        strategy_returns: List of strategy net returns per trade (as decimals)
        spy_returns: List of matched SPY returns over the same windows (as decimals)
        strategy_cagr: Annualized CAGR of strategy
        spy_cagr: Annualized CAGR of SPY
        num_trades: Total number of trades
        
    Returns:
        Dict containing CAPM metrics: alpha_annualized, alpha_t, alpha_p, beta, beta_share
    """
    if len(strategy_returns) < 2 or len(spy_returns) < 2:
        return {
            "alpha_annualized": 0.0,
            "alpha_t": 0.0,
            "alpha_p": 1.0,
            "beta": 1.0,
            "beta_share": None,
            "method": "trade_window_capm"
        }
        
    x = np.array(spy_returns)
    y = np.array(strategy_returns)
    
    # Run OLS regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    beta = float(slope)
    alpha_per_trade = float(intercept)
    
    # T-statistic and P-value for alpha (intercept)
    # Standard error of intercept:
    # SE(intercept) = std_err * sqrt(sum(x^2) / (N * sum((x - x_mean)^2)))
    # If std_err is 0 or x is constant, we handle it
    n = len(x)
    x_mean = np.mean(x)
    x_var = np.sum((x - x_mean) ** 2)
    
    if x_var > 0 and std_err > 0:
        se_intercept = std_err * np.sqrt(np.sum(x ** 2) / (n * x_var))
        alpha_t = alpha_per_trade / se_intercept
        # 2-tailed p-value
        alpha_p = 2 * (1 - stats.t.cdf(abs(alpha_t), df=n - 2))
    else:
        alpha_t = 0.0; alpha_p = 1.0
        
    # Portfolio-level annualized alpha: the strategy CAGR not explained by its market
    # exposure. Consistent with the portfolio-math definition of beta_share below.
    alpha_annualized = strategy_cagr - beta * spy_cagr

    # beta_share is a communication helper (not a formal test): the fraction of the
    # strategy's return attributable to ordinary market beta. Only meaningful when both
    # the strategy and SPY have positive CAGR; otherwise None.
    if strategy_cagr > 0 and spy_cagr > 0:
        beta_share = min(max((beta * spy_cagr) / strategy_cagr, 0.0), 1.0)
    else:
        beta_share = None
        
    return {
        "alpha_annualized": float(alpha_annualized),
        "alpha_t": float(alpha_t),
        "alpha_p": float(alpha_p),
        "beta": float(beta),
        "beta_share": beta_share,
        "method": "trade_window_capm"
    }
