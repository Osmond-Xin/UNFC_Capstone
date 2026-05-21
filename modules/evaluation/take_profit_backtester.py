"""
Take-Profit Backtester

This module compares multiple exit strategies for trades entered via
the existing signal pipeline (EMA bounce, oversold reversal, etc.).

Six strategies are implemented:
1. fixed_pct      - Exit when close >= entry * (1 + tp%)
2. trailing_stop  - Track highest high, exit when close drops trail% from peak
3. rsi_exit       - Exit when RSI crosses above threshold then drops back below
4. resistance_exit- Exit when close reaches RESISTANCE_LEVEL column
5. kill_candle_exit- Exit on kill candle detection
6. time_based     - Exit after N trading days

All strategies share a common stop-loss (5%) and max_hold_days (60) backstop.

Usage:
    from modules.evaluation import TakeProfitBacktester

    bt = TakeProfitBacktester()
    result = bt.backtest_exit_strategy(df, entry_date, entry_price, 'trailing_stop', trail=0.05)
    comparison = bt.compare_strategies(stock_data, trades_df)
    summary = bt.summarize_comparison(comparison)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from ..config.ema_params import is_kill_candle


class TakeProfitBacktester:
    """
    Compare exit strategies via historical backtesting.
    """

    DEFAULT_STOP_LOSS = 0.05       # 5% stop loss
    DEFAULT_MAX_HOLD = 60          # 60 trading days max hold

    STRATEGY_DEFAULTS = {
        'fixed_pct':        {'tp': 0.10},
        'trailing_stop':    {'trail': 0.05},
        'rsi_exit':         {'threshold': 70},
        'resistance_exit':  {},
        'kill_candle_exit': {},
        'time_based':       {'hold_days': 20},
    }

    def __init__(self, stop_loss: float = None, max_hold_days: int = None):
        self.stop_loss = stop_loss if stop_loss is not None else self.DEFAULT_STOP_LOSS
        self.max_hold_days = max_hold_days if max_hold_days is not None else self.DEFAULT_MAX_HOLD

    # -----------------------------------------------------------------
    # Core backtest method
    # -----------------------------------------------------------------
    def backtest_exit_strategy(
        self,
        df: pd.DataFrame,
        entry_date,
        entry_price: float,
        strategy: str,
        **kwargs,
    ) -> dict:
        """
        Simulate a single trade with a given exit strategy.

        Args:
            df: DataFrame with OHLCV + indicator columns (DatetimeIndex)
            entry_date: Trade entry date (will find >= in index)
            entry_price: Entry price
            strategy: One of the 6 strategy names
            **kwargs: Strategy-specific parameters

        Returns:
            dict with keys: strategy, entry_date, entry_price, exit_date,
                  exit_price, return_pct, hold_days, exit_reason,
                  max_gain_pct, max_drawdown_pct
        """
        entry_date = pd.Timestamp(entry_date)

        # Align entry_date timezone to match the DataFrame index
        idx_tz = df.index.tz if hasattr(df.index, 'tz') else None
        if idx_tz is not None:
            entry_date = entry_date.tz_localize(idx_tz) if entry_date.tzinfo is None else entry_date.tz_convert(idx_tz)
        elif entry_date.tzinfo is not None:
            entry_date = entry_date.tz_convert(None)

        # Slice data from entry onwards
        df_trade = df[df.index >= entry_date]
        if df_trade.empty:
            return self._empty_result(strategy, entry_date, entry_price)

        # Limit to max_hold_days
        df_trade = df_trade.iloc[:self.max_hold_days]

        # Merge defaults with kwargs
        params = {**self.STRATEGY_DEFAULTS.get(strategy, {}), **kwargs}

        # Dispatch
        dispatch = {
            'fixed_pct':        self._run_fixed_pct,
            'trailing_stop':    self._run_trailing_stop,
            'rsi_exit':         self._run_rsi_exit,
            'resistance_exit':  self._run_resistance_exit,
            'kill_candle_exit': self._run_kill_candle_exit,
            'time_based':       self._run_time_based,
        }

        if strategy not in dispatch:
            raise ValueError(f"Unknown strategy: {strategy}. "
                             f"Choose from: {list(dispatch.keys())}")

        return dispatch[strategy](df_trade, entry_date, entry_price, params)

    # -----------------------------------------------------------------
    # Strategy implementations
    # -----------------------------------------------------------------
    def _run_fixed_pct(self, df, entry_date, entry_price, params):
        tp = params['tp']
        tp_price = entry_price * (1 + tp)
        sl_price = entry_price * (1 - self.stop_loss)

        for i, (date, row) in enumerate(df.iterrows()):
            if row['Close'] >= tp_price:
                return self._build_result('fixed_pct', entry_date, entry_price,
                                          date, row['Close'], i + 1, 'take_profit', df, entry_price)
            if row['Close'] <= sl_price:
                return self._build_result('fixed_pct', entry_date, entry_price,
                                          date, row['Close'], i + 1, 'stop_loss', df, entry_price)

        # Max hold reached
        last_date = df.index[-1]
        return self._build_result('fixed_pct', entry_date, entry_price,
                                  last_date, df.iloc[-1]['Close'], len(df), 'max_hold', df, entry_price)

    def _run_trailing_stop(self, df, entry_date, entry_price, params):
        trail = params['trail']
        sl_price = entry_price * (1 - self.stop_loss)
        peak = entry_price

        for i, (date, row) in enumerate(df.iterrows()):
            high = row['High']
            if high > peak:
                peak = high

            trail_stop = peak * (1 - trail)

            if row['Close'] <= sl_price:
                return self._build_result('trailing_stop', entry_date, entry_price,
                                          date, row['Close'], i + 1, 'stop_loss', df, entry_price)
            if row['Close'] <= trail_stop and i > 0:
                return self._build_result('trailing_stop', entry_date, entry_price,
                                          date, row['Close'], i + 1, 'trailing_stop', df, entry_price)

        last_date = df.index[-1]
        return self._build_result('trailing_stop', entry_date, entry_price,
                                  last_date, df.iloc[-1]['Close'], len(df), 'max_hold', df, entry_price)

    def _run_rsi_exit(self, df, entry_date, entry_price, params):
        threshold = params['threshold']
        sl_price = entry_price * (1 - self.stop_loss)
        was_above = False

        for i, (date, row) in enumerate(df.iterrows()):
            if row['Close'] <= sl_price:
                return self._build_result('rsi_exit', entry_date, entry_price,
                                          date, row['Close'], i + 1, 'stop_loss', df, entry_price)

            rsi = row.get('RSI', np.nan)
            if not np.isnan(rsi):
                if rsi >= threshold:
                    was_above = True
                elif was_above and rsi < threshold:
                    return self._build_result('rsi_exit', entry_date, entry_price,
                                              date, row['Close'], i + 1, 'rsi_cross_down', df, entry_price)

        last_date = df.index[-1]
        return self._build_result('rsi_exit', entry_date, entry_price,
                                  last_date, df.iloc[-1]['Close'], len(df), 'max_hold', df, entry_price)

    def _run_resistance_exit(self, df, entry_date, entry_price, params):
        sl_price = entry_price * (1 - self.stop_loss)

        for i, (date, row) in enumerate(df.iterrows()):
            if row['Close'] <= sl_price:
                return self._build_result('resistance_exit', entry_date, entry_price,
                                          date, row['Close'], i + 1, 'stop_loss', df, entry_price)

            res_level = row.get('RESISTANCE_LEVEL', np.nan)
            if not np.isnan(res_level) and row['Close'] >= res_level:
                return self._build_result('resistance_exit', entry_date, entry_price,
                                          date, row['Close'], i + 1, 'resistance_hit', df, entry_price)

        last_date = df.index[-1]
        return self._build_result('resistance_exit', entry_date, entry_price,
                                  last_date, df.iloc[-1]['Close'], len(df), 'max_hold', df, entry_price)

    def _run_kill_candle_exit(self, df, entry_date, entry_price, params):
        sl_price = entry_price * (1 - self.stop_loss)

        # Pre-calculate average body size (20-bar rolling)
        body_pct = ((df['Close'] - df['Open']) / df['Open'] * 100).abs()
        avg_body = body_pct.rolling(window=20, min_periods=5).mean()

        for i, (date, row) in enumerate(df.iterrows()):
            if row['Close'] <= sl_price:
                return self._build_result('kill_candle_exit', entry_date, entry_price,
                                          date, row['Close'], i + 1, 'stop_loss', df, entry_price)

            candle_body = (row['Close'] - row['Open']) / row['Open'] * 100
            avg_b = avg_body.iloc[i] if i < len(avg_body) else body_pct.mean()
            if is_kill_candle(candle_body, avg_b):
                return self._build_result('kill_candle_exit', entry_date, entry_price,
                                          date, row['Close'], i + 1, 'kill_candle', df, entry_price)

        last_date = df.index[-1]
        return self._build_result('kill_candle_exit', entry_date, entry_price,
                                  last_date, df.iloc[-1]['Close'], len(df), 'max_hold', df, entry_price)

    def _run_time_based(self, df, entry_date, entry_price, params):
        hold_days = params['hold_days']
        sl_price = entry_price * (1 - self.stop_loss)

        for i, (date, row) in enumerate(df.iterrows()):
            if row['Close'] <= sl_price:
                return self._build_result('time_based', entry_date, entry_price,
                                          date, row['Close'], i + 1, 'stop_loss', df, entry_price)
            if i + 1 >= hold_days:
                return self._build_result('time_based', entry_date, entry_price,
                                          date, row['Close'], i + 1, 'time_exit', df, entry_price)

        last_date = df.index[-1]
        return self._build_result('time_based', entry_date, entry_price,
                                  last_date, df.iloc[-1]['Close'], len(df), 'max_hold', df, entry_price)

    # -----------------------------------------------------------------
    # Result helpers
    # -----------------------------------------------------------------
    def _build_result(self, strategy, entry_date, entry_price,
                      exit_date, exit_price, hold_days, exit_reason,
                      df_trade, ref_price):
        # Calculate max gain / drawdown over the holding period
        held = df_trade.iloc[:hold_days]
        if not held.empty and ref_price > 0:
            max_high = held['High'].max()
            min_low = held['Low'].min()
            max_gain_pct = (max_high - ref_price) / ref_price * 100
            max_dd_pct = (min_low - ref_price) / ref_price * 100
        else:
            max_gain_pct = 0.0
            max_dd_pct = 0.0

        return_pct = (exit_price - entry_price) / entry_price * 100 if entry_price > 0 else 0.0

        return {
            'strategy': strategy,
            'entry_date': entry_date,
            'entry_price': round(entry_price, 4),
            'exit_date': exit_date,
            'exit_price': round(exit_price, 4),
            'return_pct': round(return_pct, 2),
            'hold_days': hold_days,
            'exit_reason': exit_reason,
            'max_gain_pct': round(max_gain_pct, 2),
            'max_drawdown_pct': round(max_dd_pct, 2),
        }

    def _empty_result(self, strategy, entry_date, entry_price):
        return {
            'strategy': strategy,
            'entry_date': entry_date,
            'entry_price': round(entry_price, 4),
            'exit_date': None,
            'exit_price': None,
            'return_pct': 0.0,
            'hold_days': 0,
            'exit_reason': 'no_data',
            'max_gain_pct': 0.0,
            'max_drawdown_pct': 0.0,
        }

    # -----------------------------------------------------------------
    # Comparison across strategies
    # -----------------------------------------------------------------
    def compare_strategies(
        self,
        stock_data: Dict[str, pd.DataFrame],
        trades_df: pd.DataFrame,
        strategies: List[str] = None,
        strategy_params: Dict[str, dict] = None,
    ) -> pd.DataFrame:
        """
        Run all strategies on every trade and return a comparison DataFrame.

        Args:
            stock_data: dict mapping ticker -> DataFrame (with indicators)
            trades_df: DataFrame with columns: ticker, entry_date, entry_price
            strategies: list of strategy names (default: all 6)
            strategy_params: optional per-strategy kwargs, e.g.
                {'fixed_pct': {'tp': 0.15}, 'trailing_stop': {'trail': 0.07}}

        Returns:
            DataFrame with one row per (trade, strategy) combination
        """
        if strategies is None:
            strategies = list(self.STRATEGY_DEFAULTS.keys())
        if strategy_params is None:
            strategy_params = {}

        results = []

        for _, trade in trades_df.iterrows():
            ticker = trade['ticker']
            entry_date = trade['entry_date']
            entry_price = trade['entry_price']

            if ticker not in stock_data:
                continue

            df = stock_data[ticker]

            for strat in strategies:
                params = strategy_params.get(strat, {})
                result = self.backtest_exit_strategy(
                    df, entry_date, entry_price, strat, **params
                )
                result['ticker'] = ticker
                results.append(result)

        if not results:
            return pd.DataFrame()

        return pd.DataFrame(results)

    def summarize_comparison(self, comparison_df: pd.DataFrame) -> pd.DataFrame:
        """
        Summarize strategy comparison results.

        Args:
            comparison_df: Output from compare_strategies()

        Returns:
            DataFrame indexed by strategy with columns:
                trades, avg_return, median_return, win_rate,
                avg_hold_days, sharpe, profit_factor,
                avg_max_gain, avg_max_drawdown
        """
        if comparison_df.empty:
            return pd.DataFrame()

        summary = []
        for strat, group in comparison_df.groupby('strategy'):
            returns = group['return_pct']
            wins = returns[returns > 0]
            losses = returns[returns <= 0]

            avg_ret = returns.mean()
            std_ret = returns.std()
            sharpe = (avg_ret / std_ret) if std_ret > 0 else 0.0

            gross_profit = wins.sum() if len(wins) > 0 else 0.0
            gross_loss = abs(losses.sum()) if len(losses) > 0 else 0.0
            profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')

            summary.append({
                'strategy': strat,
                'trades': len(group),
                'avg_return': round(avg_ret, 2),
                'median_return': round(returns.median(), 2),
                'win_rate': round((returns > 0).mean() * 100, 1),
                'avg_hold_days': round(group['hold_days'].mean(), 1),
                'sharpe': round(sharpe, 3),
                'profit_factor': round(profit_factor, 2),
                'avg_max_gain': round(group['max_gain_pct'].mean(), 2),
                'avg_max_drawdown': round(group['max_drawdown_pct'].mean(), 2),
            })

        result = pd.DataFrame(summary)
        result = result.sort_values('sharpe', ascending=False).reset_index(drop=True)
        return result
