"""
Exploration Module

This module contains experimental scripts for exploring composite indicators
and testing new pattern combinations on expiry day data.

Each file in this module tests a specific composite indicator hypothesis.

Usage:
    # Run individual explorers
    python -m modules.exploration.explore_extreme_overbought
    python -m modules.exploration.explore_deep_oversold
    python -m modules.exploration.explore_momentum_reversal
    python -m modules.exploration.explore_composite_patterns

    # Or import and use in code
    from modules.exploration import ExtremeOverboughtExplorer
    explorer = ExtremeOverboughtExplorer()
    explorer.load_data().analyze_patterns().print_summary()

    # Composite pattern exploration (combines base patterns)
    from modules.exploration import CompositePatternExplorer
    explorer = CompositePatternExplorer()
    explorer.load_data().analyze_patterns().print_summary()
"""

from .explore_extreme_overbought import ExtremeOverboughtExplorer
from .explore_deep_oversold import DeepOversoldExplorer
from .explore_momentum_reversal import MomentumReversalExplorer
from .explore_composite_patterns import CompositePatternExplorer

__all__ = [
    'ExtremeOverboughtExplorer',
    'DeepOversoldExplorer',
    'MomentumReversalExplorer',
    'CompositePatternExplorer',
]
