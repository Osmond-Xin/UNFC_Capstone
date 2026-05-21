"""
Evaluation Layer for SPY Analysis

Capstone simulation, scanning, and robustness validation.
"""

from .performance_calculator import PerformanceCalculator
from .pattern_analyzer import PatternAnalyzer
from .predictive_patterns import PredictivePatterns
from .expiry_scanner import ExpiryScanner
from .take_profit_backtester import TakeProfitBacktester
from .metrics import (
    calculate_correlation_matrix,
    statistical_significance_test,
    summarize_patterns,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_information_ratio,
    calculate_composite_score,
)
from .portfolio_simulator import run_simulation
from .robust_validation import MonteCarlo, WalkForward, parameter_sensitivity

__all__ = [
    'calculate_composite_score',
    'run_simulation',
    'MonteCarlo',
    'WalkForward',
    'parameter_sensitivity',
    'PerformanceCalculator',
    'PatternAnalyzer',
    'PredictivePatterns',
    'ExpiryScanner',
    'TakeProfitBacktester',
    'calculate_correlation_matrix',
    'statistical_significance_test',
    'summarize_patterns',
    'calculate_sharpe_ratio',
    'calculate_max_drawdown',
    'calculate_information_ratio',
]
