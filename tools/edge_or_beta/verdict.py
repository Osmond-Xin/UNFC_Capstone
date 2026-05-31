from typing import Dict, List, Optional

def resolve_verdict(
    trade_count: int,
    p_select: float,
    p_timing: float,
    strategy_cagr: float,
    spy_cagr: float,
    alpha_t: float,
    alpha_annualized: float,
    beta_share: Optional[float]
) -> Dict:
    """
    Resolve the verdict status and generate bilingual headlines and detailed reasons.
    
    Args:
        trade_count: Total number of trades executed by the strategy
        p_select: P-value for random stock selection null
        p_timing: P-value for random ETF timing null
        strategy_cagr: CAGR of the strategy
        spy_cagr: CAGR of SPY
        alpha_t: T-statistic of CAPM alpha
        alpha_annualized: Annualized CAPM alpha
        beta_share: Share of returns explained by market beta (or None)
        
    Returns:
        Dict containing verdict fields: status, headline, headline_zh, reasons
    """
    beat_random_stock = p_select < 0.05
    beat_random_etf = p_timing < 0.05
    beat_spy = strategy_cagr > spy_cagr
    significant_alpha = alpha_t >= 2.0 and alpha_annualized > 0
    
    reasons = []
    
    # 1. Check for insufficient sample size
    if trade_count < 10:
        return {
            "status": "insufficient_sample",
            "headline": "Insufficient sample size to draw a confident conclusion.",
            "headline_zh": "样本数量不足，无法得出置信结论。",
            "reasons": [
                f"The rule triggered only {trade_count} trades (minimum 10 required for evaluation).",
                "Statistical tests are unreliable on very small sample sizes."
            ]
        }
        
    # Determine the status
    if beat_random_stock and beat_random_etf and beat_spy and significant_alpha:
        status = "candidate_edge_needs_validation"
        headline = "This rule shows a candidate edge, but requires out-of-sample validation."
        headline_zh = "该规则显示出潜在优势，但仍需要样本外验证。"
    elif beat_random_stock and beat_random_etf and not beat_spy:
        status = "beats_random_but_not_spy"
        headline = "Beats random benchmarks, but trails buy-and-hold SPY."
        headline_zh = "优于随机对照组，但跑输了买入并持有 SPY。"
    elif beta_share is not None and beta_share >= 0.75:
        status = "mostly_beta"
        headline = "This looks more like market beta than alpha."
        headline_zh = "这更像 beta，不像 alpha。"
    else:
        status = "no_evidence_of_edge"
        headline = "No evidence this rule beats holding a low-cost ETF."
        headline_zh = "没有证据表明这条规则比持有低成本 ETF 更好。"
        
    # Compile detailed reason list
    if beat_random_stock:
        reasons.append(f"It successfully beat random same-date stock selections (p-value: {p_select:.3f}).")
    else:
        reasons.append(f"It did not beat random stock picks on the same dates (p-value: {p_select:.3f} >= 0.05).")
        
    if beat_random_etf:
        reasons.append(f"It successfully beat random ETF timing (p-value: {p_timing:.3f}).")
    else:
        reasons.append(f"It did not beat random ETF timing (p-value: {p_timing:.3f} >= 0.05).")
        
    if beat_spy:
        reasons.append(f"It outperformed passive buy-and-hold SPY ({strategy_cagr*100:.2f}% CAGR vs {spy_cagr*100:.2f}% CAGR).")
    else:
        reasons.append(f"It trailed passive buy-and-hold SPY ({strategy_cagr*100:.2f}% CAGR vs {spy_cagr*100:.2f}% CAGR).")
        
    if significant_alpha:
        reasons.append(f"It produced statistically significant alpha (t-stat: {alpha_t:.2f} >= 2.0).")
    else:
        reasons.append(f"Alpha is not statistically significant (t-stat: {alpha_t:.2f} < 2.0).")
        
    if beta_share is not None:
        reasons.append(f"{beta_share*100:.1f}% of the rule's return is explained by ordinary market beta.")
    else:
        reasons.append("Because the rule's return is negative or near zero, beta share is not meaningful.")
        
    return {
        "status": status,
        "headline": headline,
        "headline_zh": headline_zh,
        "reasons": reasons
    }
