"""
Predictive Pattern Discovery

This module identifies high-probability trading patterns based on
historical expiry day performance. It defines pattern rules and
calculates historical win rates and average returns.

Usage:
    from modules.evaluation import PredictivePatterns

    patterns = PredictivePatterns()
    results = patterns.discover_patterns(all_patterns_df)
    summary = patterns.get_pattern_summary(results)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass


@dataclass
class PatternDefinition:
    """Definition of a predictive pattern"""
    name: str
    description: str
    condition: Callable[[pd.Series], bool]
    expected_direction: int  # 1 for bullish, -1 for bearish


class PredictivePatterns:
    """
    Discover and evaluate predictive patterns from historical data

    This class defines pattern rules and evaluates their effectiveness
    based on historical expiry day performance.
    """

    def __init__(self):
        """Initialize with default pattern definitions"""
        self.patterns = self._define_default_patterns()

    def _define_default_patterns(self) -> Dict[str, PatternDefinition]:
        """
        Define the default set of predictive patterns

        These patterns are based on technical analysis theory:
        - Oversold conditions often lead to reversals
        - Strong trends tend to continue
        - Extreme readings may signal exhaustion

        Returns:
            dict: Dictionary of pattern definitions
        """
        patterns = {}

        # Pattern 1: Oversold Reversal
        # RSI < 30 and 3+ consecutive red candles -> expect bounce
        patterns['Oversold_Reversal'] = PatternDefinition(
            name='Oversold_Reversal',
            description='RSI < 30 with 3+ consecutive red candles (expect bounce)',
            condition=lambda row: (
                row.get('RSI', 100) < 30 and
                row.get('Candle_Direction', '') == 'red' and
                row.get('Consecutive_Candles', 0) >= 3
            ),
            expected_direction=1  # Expect bullish bounce
        )

        # Pattern 2: Strong Uptrend
        # Price above all MAs (9, 20, 50) + 3+ green candles -> trend continuation
        patterns['Strong_Uptrend'] = PatternDefinition(
            name='Strong_Uptrend',
            description='Above all MAs (9/20/50) with 3+ green candles (trend continuation)',
            condition=lambda row: (
                row.get('MA9_dist', -999) > 0 and
                row.get('MA20_dist', -999) > 0 and
                row.get('MA50_dist', -999) > 0 and
                row.get('Candle_Direction', '') == 'green' and
                row.get('Consecutive_Candles', 0) >= 3
            ),
            expected_direction=1  # Expect bullish continuation
        )

        # Pattern 3: Deep Pullback
        # Price > 5% below MA20 and RSI < 40 -> expect mean reversion
        patterns['Deep_Pullback'] = PatternDefinition(
            name='Deep_Pullback',
            description='Price >5% below MA20 with RSI < 40 (mean reversion setup)',
            condition=lambda row: (
                row.get('MA20_dist', 0) < -5 and
                row.get('RSI', 100) < 40
            ),
            expected_direction=1  # Expect bullish reversion
        )

        # Pattern 4: Overbought Momentum
        # RSI > 70 with green candles -> momentum may continue or reverse
        patterns['Overbought_Momentum'] = PatternDefinition(
            name='Overbought_Momentum',
            description='RSI > 70 with green candles (momentum continuation or exhaustion)',
            condition=lambda row: (
                row.get('RSI', 0) > 70 and
                row.get('Candle_Direction', '') == 'green'
            ),
            expected_direction=1  # Often continues in strong markets
        )

        # Pattern 5: Bearish Exhaustion (DISABLED for observation)
        # RSI > 80 with 4+ green candles -> potential exhaustion
        # NOTE: Temporarily commented out to observe impact
        # patterns['Bearish_Exhaustion'] = PatternDefinition(
        #     name='Bearish_Exhaustion',
        #     description='RSI > 80 with 4+ green candles (exhaustion warning)',
        #     condition=lambda row: (
        #         row.get('RSI', 0) > 80 and
        #         row.get('Candle_Direction', '') == 'green' and
        #         row.get('Consecutive_Candles', 0) >= 4
        #     ),
        #     expected_direction=-1  # Expect bearish reversal
        # )

        # Pattern 6: Volume Surge Reversal
        # High volume (>2x average) with oversold RSI -> institutional interest
        patterns['Volume_Surge_Reversal'] = PatternDefinition(
            name='Volume_Surge_Reversal',
            description='Volume surge (>2x) with RSI < 35 (institutional buying)',
            condition=lambda row: (
                row.get('Volume_Ratio', 0) > 2.0 and
                row.get('RSI', 100) < 35
            ),
            expected_direction=1  # Expect bullish reversal
        )

        # Pattern 7: Height Score Warning
        # High HEIGHT_SCORE (>0.6) with RSI > 65 -> elevated risk of pullback
        patterns['Height_Score_Warning'] = PatternDefinition(
            name='Height_Score_Warning',
            description='HEIGHT_SCORE > 0.6 with RSI > 65 (overbought warning)',
            condition=lambda row: (
                row.get('HEIGHT_SCORE', 0) > 0.6 and
                row.get('RSI', 0) > 65
            ),
            expected_direction=-1  # Expect bearish pullback
        )

        # Pattern 8: Near Resistance
        # Price within 1% of a resistance level -> potential rejection
        patterns['Near_Resistance'] = PatternDefinition(
            name='Near_Resistance',
            description='Price within 1% of resistance level (potential rejection)',
            condition=lambda row: (
                0 < row.get('RESISTANCE_DIST_PCT', 999) < 1.0
            ),
            expected_direction=-1  # Expect bearish rejection
        )

        return patterns

    def discover_patterns(
        self,
        all_patterns: pd.DataFrame,
        min_occurrences: int = 5
    ) -> pd.DataFrame:
        """
        Discover pattern occurrences in historical data

        Args:
            all_patterns (pd.DataFrame): Historical pattern data from PatternAnalyzer
            min_occurrences (int): Minimum occurrences to report a pattern

        Returns:
            pd.DataFrame: Pattern discovery results with columns:
                - pattern_name: Name of the pattern
                - occurrences: Number of times pattern occurred
                - avg_return: Average expiry day return
                - positive_rate: Percentage of positive returns
                - expected_direction: Expected signal direction
                - actual_win_rate: Win rate based on expected direction
        """
        if all_patterns.empty:
            return pd.DataFrame()

        results = []

        for pattern_name, pattern_def in self.patterns.items():
            # Find all rows matching this pattern
            matches = all_patterns.apply(pattern_def.condition, axis=1)
            matched_rows = all_patterns[matches]

            if len(matched_rows) < min_occurrences:
                continue

            returns = matched_rows['Change_Pct']

            # Calculate win rate based on expected direction
            if pattern_def.expected_direction == 1:
                wins = (returns > 0).sum()
            else:
                wins = (returns < 0).sum()

            win_rate = (wins / len(matched_rows)) * 100 if len(matched_rows) > 0 else 0

            results.append({
                'pattern_name': pattern_name,
                'description': pattern_def.description,
                'occurrences': len(matched_rows),
                'avg_return': returns.mean(),
                'median_return': returns.median(),
                'std_return': returns.std(),
                'positive_rate': (returns > 0).mean() * 100,
                'expected_direction': 'Bullish' if pattern_def.expected_direction == 1 else 'Bearish',
                'actual_win_rate': win_rate,
                'best_return': returns.max(),
                'worst_return': returns.min()
            })

        if not results:
            return pd.DataFrame()

        result_df = pd.DataFrame(results)
        # Sort by win rate descending
        result_df = result_df.sort_values('actual_win_rate', ascending=False)
        result_df = result_df.reset_index(drop=True)

        return result_df

    def get_matched_stocks(
        self,
        all_patterns: pd.DataFrame,
        pattern_name: str
    ) -> pd.DataFrame:
        """
        Get all stocks that matched a specific pattern

        Args:
            all_patterns (pd.DataFrame): Historical pattern data
            pattern_name (str): Name of the pattern to match

        Returns:
            pd.DataFrame: Stocks matching the pattern
        """
        if pattern_name not in self.patterns:
            raise ValueError(f"Unknown pattern: {pattern_name}")

        pattern_def = self.patterns[pattern_name]
        matches = all_patterns.apply(pattern_def.condition, axis=1)

        return all_patterns[matches].copy()

    def add_pattern(self, pattern_def: PatternDefinition):
        """
        Add a custom pattern definition

        Args:
            pattern_def (PatternDefinition): Pattern definition to add
        """
        self.patterns[pattern_def.name] = pattern_def

    def create_custom_pattern(
        self,
        name: str,
        description: str,
        rsi_range: tuple = None,
        ma_distances: Dict[str, tuple] = None,
        consecutive_candles: tuple = None,
        candle_direction: str = None,
        volume_ratio: tuple = None,
        expected_direction: int = 1
    ) -> PatternDefinition:
        """
        Create a custom pattern with configurable conditions

        Args:
            name (str): Pattern name
            description (str): Pattern description
            rsi_range (tuple): (min, max) RSI range, e.g., (0, 30) for oversold
            ma_distances (dict): {'MA9_dist': (min, max), ...}
            consecutive_candles (tuple): (min, max) consecutive candle count
            candle_direction (str): 'green' or 'red'
            volume_ratio (tuple): (min, max) volume ratio
            expected_direction (int): 1 for bullish, -1 for bearish

        Returns:
            PatternDefinition: The created pattern definition
        """
        def condition(row):
            result = True

            if rsi_range is not None:
                rsi = row.get('RSI', 50)
                result = result and (rsi_range[0] <= rsi <= rsi_range[1])

            if ma_distances is not None:
                for ma_col, (min_dist, max_dist) in ma_distances.items():
                    dist = row.get(ma_col, 0)
                    result = result and (min_dist <= dist <= max_dist)

            if consecutive_candles is not None:
                count = row.get('Consecutive_Candles', 0)
                result = result and (consecutive_candles[0] <= count <= consecutive_candles[1])

            if candle_direction is not None:
                direction = row.get('Candle_Direction', '')
                result = result and (direction == candle_direction)

            if volume_ratio is not None:
                vol_ratio = row.get('Volume_Ratio', 1.0)
                result = result and (volume_ratio[0] <= vol_ratio <= volume_ratio[1])

            return result

        pattern = PatternDefinition(
            name=name,
            description=description,
            condition=condition,
            expected_direction=expected_direction
        )

        return pattern

    def print_pattern_report(self, pattern_results: pd.DataFrame):
        """
        Print a formatted report of pattern discovery results

        Args:
            pattern_results (pd.DataFrame): Results from discover_patterns()
        """
        if pattern_results.empty:
            print("No patterns discovered.")
            return

        print("=" * 70)
        print("PREDICTIVE PATTERN DISCOVERY REPORT")
        print("=" * 70)

        for _, row in pattern_results.iterrows():
            print(f"\nPattern: {row['pattern_name']}")
            print(f"  {row['description']}")
            print(f"  Occurrences: {row['occurrences']}")
            print(f"  Avg Expiry Day Return: {row['avg_return']:+.2f}%")
            print(f"  Positive Rate: {row['positive_rate']:.1f}%")
            print(f"  Expected Direction: {row['expected_direction']}")
            print(f"  Win Rate (direction-adjusted): {row['actual_win_rate']:.1f}%")

            # Highlight high win-rate patterns
            if row['actual_win_rate'] >= 60:
                print("  ⭐ HIGH PROBABILITY PATTERN")

        print("\n" + "=" * 70)
