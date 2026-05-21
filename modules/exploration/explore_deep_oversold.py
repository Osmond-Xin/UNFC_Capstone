"""
Explore Deep Oversold Composite Indicator

Hypothesis: Combining multiple oversold signals increases reversal probability.
- Deep_Pullback: Price >5% below MA20 + RSI < 40
- Oversold_Reversal: RSI < 30 + 3+ red candles
- Volume_Surge_Reversal: Volume >2x + RSI < 35

Combined conditions suggest extreme oversold state with potential bullish reversal.

Usage:
    python -m modules.exploration.explore_deep_oversold
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Callable

from modules.data import CacheManager, DataLoader, ExpiryCalculator
from modules.features import TechnicalIndicators, FeaturePipeline
from modules.evaluation import PerformanceCalculator, PatternAnalyzer


@dataclass
class CompositePattern:
    """Definition of a composite pattern"""
    name: str
    description: str
    condition: Callable[[pd.Series], bool]
    expected_direction: int  # 1 for bullish, -1 for bearish


class DeepOversoldExplorer:
    """
    Explorer for Deep Oversold composite indicators

    Tests various combinations of oversold conditions to find
    the most reliable bullish reversal signals.
    """

    def __init__(self):
        self.patterns = self._define_patterns()
        self.results = {}

    def _define_patterns(self) -> Dict[str, CompositePattern]:
        """Define composite patterns to test"""
        patterns = {}

        # Composite 1: Deep Pullback + Oversold RSI + Red Candles
        patterns['Extreme_Oversold_v1'] = CompositePattern(
            name='Extreme_Oversold_v1',
            description='MA20_dist<-5% + RSI<30 + 3+ red candles',
            condition=lambda row: (
                row.get('MA20_dist', 0) < -5 and
                row.get('RSI', 100) < 30 and
                row.get('Candle_Direction', '') == 'red' and
                row.get('Consecutive_Candles', 0) >= 3
            ),
            expected_direction=1  # Expect bullish reversal
        )

        # Composite 2: Deep Pullback + Volume Surge
        patterns['Deep_Pullback_Volume'] = CompositePattern(
            name='Deep_Pullback_Volume',
            description='MA20_dist<-5% + RSI<40 + Volume>1.5x',
            condition=lambda row: (
                row.get('MA20_dist', 0) < -5 and
                row.get('RSI', 100) < 40 and
                row.get('Volume_Ratio', 0) > 1.5
            ),
            expected_direction=1
        )

        # Composite 3: Below all MAs + Oversold
        patterns['Below_All_MAs_Oversold'] = CompositePattern(
            name='Below_All_MAs_Oversold',
            description='Below all MAs (9/20/50) + RSI<35',
            condition=lambda row: (
                row.get('MA9_dist', 999) < 0 and
                row.get('MA20_dist', 999) < 0 and
                row.get('MA50_dist', 999) < 0 and
                row.get('RSI', 100) < 35
            ),
            expected_direction=1
        )

        # Composite 4: Extreme Deep Pullback
        patterns['Extreme_Deep_Pullback'] = CompositePattern(
            name='Extreme_Deep_Pullback',
            description='MA20_dist<-10% + RSI<30',
            condition=lambda row: (
                row.get('MA20_dist', 0) < -10 and
                row.get('RSI', 100) < 30
            ),
            expected_direction=1
        )

        # Composite 5: Deep Pullback + Consecutive Red
        patterns['Deep_Pullback_Red_Streak'] = CompositePattern(
            name='Deep_Pullback_Red_Streak',
            description='MA20_dist<-5% + RSI<40 + 2+ red candles',
            condition=lambda row: (
                row.get('MA20_dist', 0) < -5 and
                row.get('RSI', 100) < 40 and
                row.get('Candle_Direction', '') == 'red' and
                row.get('Consecutive_Candles', 0) >= 2
            ),
            expected_direction=1
        )

        # Composite 6: Volume Spike + Extreme Oversold
        patterns['Volume_Spike_Oversold'] = CompositePattern(
            name='Volume_Spike_Oversold',
            description='Volume>2x + RSI<25 (capitulation signal)',
            condition=lambda row: (
                row.get('Volume_Ratio', 0) > 2.0 and
                row.get('RSI', 100) < 25
            ),
            expected_direction=1
        )

        # Composite 7: Multi-timeframe oversold
        patterns['Multi_MA_Oversold'] = CompositePattern(
            name='Multi_MA_Oversold',
            description='MA9_dist<-3% + MA20_dist<-5% + MA50_dist<-8%',
            condition=lambda row: (
                row.get('MA9_dist', 0) < -3 and
                row.get('MA20_dist', 0) < -5 and
                row.get('MA50_dist', 0) < -8
            ),
            expected_direction=1
        )

        # Composite 8: Relaxed oversold with volume confirmation
        patterns['Oversold_Volume_Confirm'] = CompositePattern(
            name='Oversold_Volume_Confirm',
            description='RSI<35 + Volume>1.8x + red candles',
            condition=lambda row: (
                row.get('RSI', 100) < 35 and
                row.get('Volume_Ratio', 0) > 1.8 and
                row.get('Candle_Direction', '') == 'red'
            ),
            expected_direction=1
        )

        return patterns

    def load_data(self, start_date: str = '2022-01-01', end_date: str = '2025-11-21'):
        """Load stock data and calculate features"""
        print("=" * 70)
        print("DEEP OVERSOLD COMPOSITE INDICATOR EXPLORATION")
        print("=" * 70)

        # Initialize components
        loader = DataLoader()

        # Generate expiry dates
        self.expiry_dates = ExpiryCalculator.generate_expiry_dates(start_date, end_date)
        print(f"\n📅 Analysis Period: {start_date} to {end_date}")
        print(f"   Expiry dates: {len(self.expiry_dates)}")

        # Load stock data
        sp500_tickers = loader.get_sp500_tickers()
        print(f"\n📈 Loading {len(sp500_tickers)} S&P 500 stocks...")

        self.stock_data = loader.load_sp500_batch(
            sp500_tickers,
            start_date=start_date,
            end_date=end_date,
            validate=False
        )
        print(f"   Loaded: {len(self.stock_data)} stocks")

        # Calculate technical indicators
        pipeline = FeaturePipeline([
            TechnicalIndicators(config={
                'rsi_period': 14,
                'ma_periods': [9, 20, 50],
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9,
                'bb_period': 20,
                'volume_ma_period': 20,
                'consecutive_lookback': 5
            })
        ])

        print(f"\n🔧 Calculating technical indicators...")
        for ticker in self.stock_data:
            self.stock_data[ticker] = pipeline.transform(self.stock_data[ticker])
        print(f"   ✓ Features calculated")

        return self

    def analyze_patterns(self):
        """Analyze all composite patterns"""
        # Calculate performance for all expiry dates
        perf_calc = PerformanceCalculator()
        pattern_analyzer = PatternAnalyzer()

        print(f"\n🔄 Calculating multi-expiry performance...")
        performance_by_expiry = perf_calc.calculate_multi_expiry(
            self.stock_data, self.expiry_dates
        )
        print(f"   Found data for {len(performance_by_expiry)} expiry dates")

        # Analyze patterns across all expiries
        print(f"\n📊 Analyzing patterns across expiry dates...")
        multi_expiry_patterns = pattern_analyzer.analyze_multi_expiry(
            self.stock_data, performance_by_expiry, self.expiry_dates,
            top_n=10, bottom_n=10
        )

        if multi_expiry_patterns.empty:
            print("   ⚠️ No pattern data available")
            return self

        print(f"   Total samples: {len(multi_expiry_patterns)}")
        self.pattern_data = multi_expiry_patterns

        # Test each composite pattern
        print(f"\n" + "=" * 70)
        print("COMPOSITE PATTERN RESULTS")
        print("=" * 70)

        for pattern_name, pattern_def in self.patterns.items():
            result = self._evaluate_pattern(pattern_def, multi_expiry_patterns)
            self.results[pattern_name] = result
            self._print_pattern_result(pattern_def, result)

        return self

    def _evaluate_pattern(self, pattern: CompositePattern, data: pd.DataFrame) -> Dict:
        """Evaluate a single composite pattern"""
        # Find matching rows
        matches = data.apply(pattern.condition, axis=1)
        matched_rows = data[matches]

        if len(matched_rows) == 0:
            return {
                'occurrences': 0,
                'avg_return': None,
                'positive_rate': None,
                'win_rate': None,
                'matched_data': pd.DataFrame()
            }

        returns = matched_rows['Change_Pct']

        # Calculate win rate based on expected direction
        if pattern.expected_direction == 1:
            wins = (returns > 0).sum()
        else:
            wins = (returns < 0).sum()

        win_rate = (wins / len(matched_rows)) * 100

        return {
            'occurrences': len(matched_rows),
            'avg_return': returns.mean(),
            'median_return': returns.median(),
            'std_return': returns.std(),
            'positive_rate': (returns > 0).mean() * 100,
            'win_rate': win_rate,
            'best_return': returns.max(),
            'worst_return': returns.min(),
            'matched_data': matched_rows
        }

    def _print_pattern_result(self, pattern: CompositePattern, result: Dict):
        """Print result for a single pattern"""
        print(f"\n📌 {pattern.name}")
        print(f"   {pattern.description}")
        print(f"   Expected Direction: {'Bullish' if pattern.expected_direction == 1 else 'Bearish'}")
        print(f"   " + "-" * 50)

        if result['occurrences'] == 0:
            print(f"   ⚠️ No occurrences found")
            return

        print(f"   Occurrences: {result['occurrences']}")
        print(f"   Avg Return: {result['avg_return']:+.2f}%")
        print(f"   Median Return: {result['median_return']:+.2f}%")
        print(f"   Std Dev: {result['std_return']:.2f}%")
        print(f"   Positive Rate: {result['positive_rate']:.1f}%")
        print(f"   Win Rate (direction-adjusted): {result['win_rate']:.1f}%")
        print(f"   Best: {result['best_return']:+.2f}%, Worst: {result['worst_return']:+.2f}%")

        if result['win_rate'] >= 60:
            print(f"   ⭐ HIGH PROBABILITY PATTERN!")
        elif result['win_rate'] >= 55:
            print(f"   ✓ Moderate probability")
        else:
            print(f"   ✗ Low probability")

    def get_summary(self) -> pd.DataFrame:
        """Get summary DataFrame of all results"""
        rows = []
        for name, result in self.results.items():
            if result['occurrences'] > 0:
                rows.append({
                    'Pattern': name,
                    'Occurrences': result['occurrences'],
                    'Avg_Return': result['avg_return'],
                    'Positive_Rate': result['positive_rate'],
                    'Win_Rate': result['win_rate']
                })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df = df.sort_values('Win_Rate', ascending=False)
        return df

    def print_summary(self):
        """Print summary comparison"""
        print(f"\n" + "=" * 70)
        print("SUMMARY COMPARISON")
        print("=" * 70)

        summary = self.get_summary()
        if summary.empty:
            print("No patterns found with sufficient occurrences")
            return

        print(f"\n{'Pattern':<30} {'N':>6} {'AvgRet':>8} {'PosRate':>8} {'WinRate':>8}")
        print("-" * 70)

        for _, row in summary.iterrows():
            print(f"{row['Pattern']:<30} {row['Occurrences']:>6} "
                  f"{row['Avg_Return']:>+7.2f}% {row['Positive_Rate']:>7.1f}% "
                  f"{row['Win_Rate']:>7.1f}%")

        print("\n" + "=" * 70)


def main():
    """Main entry point"""
    explorer = DeepOversoldExplorer()
    explorer.load_data(start_date='2022-01-01', end_date='2025-11-21')
    explorer.analyze_patterns()
    explorer.print_summary()

    return explorer


if __name__ == '__main__':
    main()
