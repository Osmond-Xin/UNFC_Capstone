"""
Unit tests for TechnicalIndicators

Tests technical indicator calculations.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from modules.features import TechnicalIndicators, FeaturePipeline


class TestTechnicalIndicators(unittest.TestCase):
    """Test cases for TechnicalIndicators"""

    def setUp(self):
        """Create test data"""
        # Create synthetic stock data
        np.random.seed(42)
        dates = pd.date_range('2025-01-01', periods=200, freq='D')

        # Generate realistic price data with trend
        base_price = 100
        trend = np.linspace(0, 20, 200)
        noise = np.random.randn(200) * 2

        close_prices = base_price + trend + noise

        self.test_df = pd.DataFrame({
            'Open': close_prices + np.random.randn(200) * 0.5,
            'High': close_prices + np.abs(np.random.randn(200) * 1.5),
            'Low': close_prices - np.abs(np.random.randn(200) * 1.5),
            'Close': close_prices,
            'Volume': np.random.randint(1000000, 10000000, 200)
        }, index=dates)

        # Ensure OHLC logic
        for i in range(len(self.test_df)):
            self.test_df.iloc[i, self.test_df.columns.get_loc('High')] = max(
                self.test_df.iloc[i]['High'],
                self.test_df.iloc[i]['Close'],
                self.test_df.iloc[i]['Open']
            )
            self.test_df.iloc[i, self.test_df.columns.get_loc('Low')] = min(
                self.test_df.iloc[i]['Low'],
                self.test_df.iloc[i]['Close'],
                self.test_df.iloc[i]['Open']
            )

        self.indicators = TechnicalIndicators()

    def test_initialization(self):
        """Test default configuration"""
        self.assertEqual(self.indicators.config['rsi_period'], 14)
        self.assertEqual(self.indicators.config['ma_periods'], [9, 20, 50])
        self.assertEqual(self.indicators.config['bb_period'], 20)

    def test_calculate_all_indicators(self):
        """Test that calculate() adds all expected columns"""
        result = self.indicators.calculate(self.test_df)

        # Check that all expected columns are added
        expected_columns = [
            'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
            'SMA_9', 'SMA_20', 'SMA_50',
            'MA_Distance_9', 'MA_Distance_20', 'MA_Distance_50',
            'BB_Upper', 'BB_Middle', 'BB_Lower', 'BB_Position',
            'Volume_SMA', 'Volume_Ratio',
            'Consecutive_Count', 'Consecutive_Direction'
        ]

        for col in expected_columns:
            self.assertIn(col, result.columns, f"Missing column: {col}")

    def test_rsi_calculation(self):
        """Test RSI calculation"""
        result = self.indicators.calculate(self.test_df)

        # RSI should be between 0 and 100
        rsi_values = result['RSI'].dropna()
        self.assertTrue(all(rsi_values >= 0), "RSI values should be >= 0")
        self.assertTrue(all(rsi_values <= 100), "RSI values should be <= 100")

        # RSI should have some variation (not all same value)
        self.assertGreater(rsi_values.std(), 0, "RSI should have variation")

    def test_macd_calculation(self):
        """Test MACD calculation"""
        result = self.indicators.calculate(self.test_df)

        # Check that MACD columns exist
        self.assertIn('MACD', result.columns)
        self.assertIn('MACD_Signal', result.columns)
        self.assertIn('MACD_Hist', result.columns)

        # MACD_Hist should equal MACD - MACD_Signal
        macd_diff = result['MACD'] - result['MACD_Signal']
        macd_hist = result['MACD_Hist']

        # Compare non-NaN values
        valid_mask = ~(macd_diff.isna() | macd_hist.isna())
        np.testing.assert_array_almost_equal(
            macd_diff[valid_mask].values,
            macd_hist[valid_mask].values,
            decimal=10
        )

    def test_moving_averages(self):
        """Test moving average calculation"""
        result = self.indicators.calculate(self.test_df)

        # Check that MAs exist
        self.assertIn('SMA_9', result.columns)
        self.assertIn('SMA_20', result.columns)
        self.assertIn('SMA_50', result.columns)

        # MAs should be smoothed (less volatile than price)
        price_std = result['Close'].std()
        ma20_std = result['SMA_20'].dropna().std()

        self.assertLess(ma20_std, price_std, "MA should be smoother than price")

        # Check MA distance calculation
        # MA_Distance should be percentage difference
        close = result['Close'].iloc[-1]
        sma_20 = result['SMA_20'].iloc[-1]
        ma_dist = result['MA_Distance_20'].iloc[-1]

        if not pd.isna(sma_20):
            expected_dist = ((close - sma_20) / sma_20) * 100
            self.assertAlmostEqual(ma_dist, expected_dist, places=5)

    def test_bollinger_bands(self):
        """Test Bollinger Bands calculation"""
        result = self.indicators.calculate(self.test_df)

        # Check that BB columns exist
        self.assertIn('BB_Upper', result.columns)
        self.assertIn('BB_Middle', result.columns)
        self.assertIn('BB_Lower', result.columns)
        self.assertIn('BB_Position', result.columns)

        # Upper should be > Middle > Lower
        valid_rows = result.dropna(subset=['BB_Upper', 'BB_Middle', 'BB_Lower'])

        self.assertTrue(all(valid_rows['BB_Upper'] > valid_rows['BB_Middle']))
        self.assertTrue(all(valid_rows['BB_Middle'] > valid_rows['BB_Lower']))

        # BB Position should be between 0 and 1 (mostly, except for outliers)
        bb_pos = result['BB_Position'].dropna()
        median_pos = bb_pos.median()
        self.assertGreater(median_pos, 0)
        self.assertLess(median_pos, 1)

    def test_volume_features(self):
        """Test volume features calculation"""
        result = self.indicators.calculate(self.test_df)

        # Check columns exist
        self.assertIn('Volume_SMA', result.columns)
        self.assertIn('Volume_Ratio', result.columns)

        # Volume ratio should be Current Volume / Average Volume
        # So it should be around 1.0 on average
        volume_ratio = result['Volume_Ratio'].dropna()
        mean_ratio = volume_ratio.mean()

        self.assertAlmostEqual(mean_ratio, 1.0, places=0)

    def test_consecutive_candles(self):
        """Test consecutive candles calculation"""
        result = self.indicators.calculate(self.test_df)

        # Check columns exist
        self.assertIn('Consecutive_Count', result.columns)
        self.assertIn('Consecutive_Direction', result.columns)

        # Consecutive count should be >= 1
        counts = result['Consecutive_Count']
        self.assertTrue(all(counts >= 1))

        # Direction should be 1 or -1
        directions = result['Consecutive_Direction']
        unique_dirs = directions.unique()
        self.assertTrue(all(d in [1, -1] for d in unique_dirs))

        # Count should not exceed lookback + 1 for most cases
        max_count = result['Consecutive_Count'].max()
        self.assertLessEqual(max_count, 10)  # Reasonable upper bound

    def test_get_feature_names(self):
        """Test getting feature names"""
        feature_names = self.indicators.get_feature_names()

        # Should return a list
        self.assertIsInstance(feature_names, list)

        # Should contain expected features
        expected_features = ['RSI', 'MACD', 'SMA_20', 'BB_Upper', 'Volume_SMA']
        for feature in expected_features:
            self.assertIn(feature, feature_names)

    def test_custom_config(self):
        """Test using custom configuration"""
        custom_indicators = TechnicalIndicators(config={
            'rsi_period': 21,
            'ma_periods': [10, 30]
        })

        result = custom_indicators.calculate(self.test_df)

        # Should have custom MA periods
        self.assertIn('SMA_10', result.columns)
        self.assertIn('SMA_30', result.columns)
        self.assertNotIn('SMA_9', result.columns)  # Default, should not exist

    def test_feature_pipeline(self):
        """Test using TechnicalIndicators in a pipeline"""
        pipeline = FeaturePipeline([
            TechnicalIndicators()
        ])

        result = pipeline.transform(self.test_df)

        # Should have all technical indicators
        self.assertIn('RSI', result.columns)
        self.assertIn('MACD', result.columns)

        # Pipeline should report all feature names
        feature_names = pipeline.get_all_feature_names()
        self.assertGreater(len(feature_names), 10)


if __name__ == '__main__':
    unittest.main()
