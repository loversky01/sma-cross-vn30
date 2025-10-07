"""
optimize_parameters.py
────────────────────────────────────────────────────────────
Tối ưu hóa tham số SMA bằng Grid Search

Tìm cặp SMA tối ưu cho từng mã cổ phiếu
Optimization metrics: Sharpe Ratio, Total Return, Win Rate, etc.

Usage:
    python optimize_parameters.py MSN
    python optimize_parameters.py FPT --metric sharpe_ratio
    python optimize_parameters.py VCB --short-range 5,10,15,20 --long-range 30,50,100
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from itertools import product
from tqdm import tqdm

from utils.config_loader import get_config
from utils.indicators import generate_sma_signals
from utils.logger import get_logger
from backtest_advanced import AdvancedBacktester


class ParameterOptimizer:
    """Parameter optimization using Grid Search"""
    
    def __init__(
        self,
        short_range: List[int] = [5, 10, 15, 20],
        long_range: List[int] = [30, 50, 100, 200],
        optimize_metric: str = "sharpe_ratio",
        initial_capital: float = 100_000_000
    ):
        """
        Initialize optimizer
        
        Args:
            short_range: List of short SMA periods to test
            long_range: List of long SMA periods to test
            optimize_metric: Metric to optimize (sharpe_ratio, total_return, win_rate)
            initial_capital: Initial capital for backtest
        """
        self.short_range = short_range
        self.long_range = long_range
        self.optimize_metric = optimize_metric.lower()
        self.initial_capital = initial_capital
        
        self.logger = get_logger("optimizer")
        
        # Load config for backtest settings
        cfg = get_config()
        self.backtest_config = {
            'initial_capital': initial_capital,
            'commission_pct': cfg.commission_pct,
            'tax_pct': cfg.tax_pct,
            'slippage_pct': cfg.slippage_pct,
            'stop_loss_pct': cfg.stop_loss_pct if cfg.use_stop_loss else None,
            'take_profit_pct': cfg.take_profit_pct if cfg.use_take_profit else None,
            'trailing_stop_pct': None,
            'position_size_pct': 100.0
        }
    
    def backtest_params(self, df: pd.DataFrame, short: int, long: int) -> Dict:
        """
        Backtest with specific SMA parameters
        
        Args:
            df: Price DataFrame
            short: Short SMA period
            long: Long SMA period
            
        Returns:
            Dictionary with backtest metrics
        """
        # Generate signals with these parameters
        df_signals = generate_sma_signals(df, short_window=short, long_window=long)
        
        # Run backtest
        backtester = AdvancedBacktester(
            initial_capital=self.backtest_config['initial_capital'],
            commission_pct=self.backtest_config['commission_pct'],
            tax_pct=self.backtest_config['tax_pct'],
            slippage_pct=self.backtest_config['slippage_pct'],
            stop_loss_pct=self.backtest_config['stop_loss_pct'],
            take_profit_pct=self.backtest_config['take_profit_pct'],
            trailing_stop_pct=self.backtest_config['trailing_stop_pct'],
            position_size_pct=self.backtest_config['position_size_pct']
        )
        
        results = backtester.run(df_signals, f"Test({short},{long})")
        
        if not results or 'metrics' not in results:
            return {
                'short': short,
                'long': long,
                'total_return': 0,
                'sharpe_ratio': 0,
                'win_rate': 0,
                'total_trades': 0,
                'max_drawdown': 0
            }
        
        metrics = results['metrics']
        
        return {
            'short': short,
            'long': long,
            'total_return': metrics.get('Total Return (%)', 0),
            'sharpe_ratio': metrics.get('Sharpe Ratio', 0),
            'sortino_ratio': metrics.get('Sortino Ratio', 0),
            'calmar_ratio': metrics.get('Calmar Ratio', 0),
            'win_rate': metrics.get('Win Rate (%)', 0),
            'total_trades': metrics.get('Total Trades', 0),
            'max_drawdown': metrics.get('Max Drawdown (%)', 0),
            'profit_factor': metrics.get('Profit Factor', 0)
        }
    
    def optimize(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Run grid search optimization
        
        Args:
            df: Price DataFrame with 'time' and 'close' columns
            symbol: Stock symbol
            
        Returns:
            DataFrame with all tested parameter combinations and results
        """
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"🔍 Optimizing parameters for {symbol}")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Short SMA range: {self.short_range}")
        self.logger.info(f"Long SMA range: {self.long_range}")
        self.logger.info(f"Optimization metric: {self.optimize_metric}")
        
        # Generate all parameter combinations
        param_combinations = list(product(self.short_range, self.long_range))
        
        # Filter: short must be less than long
        param_combinations = [(s, l) for s, l in param_combinations if s < l]
        
        total_combinations = len(param_combinations)
        self.logger.info(f"Total combinations to test: {total_combinations}\n")
        
        # Test all combinations
        results = []
        
        for short, long in tqdm(param_combinations, desc="Testing parameters"):
            result = self.backtest_params(df, short, long)
            results.append(result)
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Sort by optimization metric
        metric_col_map = {
            'sharpe_ratio': 'sharpe_ratio',
            'total_return': 'total_return',
            'win_rate': 'win_rate',
            'sortino_ratio': 'sortino_ratio',
            'calmar_ratio': 'calmar_ratio',
            'profit_factor': 'profit_factor'
        }
        
        sort_col = metric_col_map.get(self.optimize_metric, 'sharpe_ratio')
        results_df = results_df.sort_values(sort_col, ascending=False)
        
        # Log top results
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"📊 TOP 10 PARAMETER COMBINATIONS")
        self.logger.info(f"{'='*60}")
        self.logger.info("\n" + results_df.head(10).to_string(index=False))
        
        # Best parameters
        best = results_df.iloc[0]
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"🏆 BEST PARAMETERS FOR {symbol}")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Short SMA: {int(best['short'])}")
        self.logger.info(f"Long SMA: {int(best['long'])}")
        self.logger.info(f"Total Return: {best['total_return']:.2f}%")
        self.logger.info(f"Sharpe Ratio: {best['sharpe_ratio']:.3f}")
        self.logger.info(f"Win Rate: {best['win_rate']:.2f}%")
        self.logger.info(f"Total Trades: {int(best['total_trades'])}")
        self.logger.info(f"Max Drawdown: {best['max_drawdown']:.2f}%")
        self.logger.info(f"{'='*60}\n")
        
        return results_df


def optimize_symbol(
    symbol: str,
    short_range: List[int] = None,
    long_range: List[int] = None,
    metric: str = "sharpe_ratio"
) -> pd.DataFrame:
    """
    Optimize parameters for a single symbol
    
    Args:
        symbol: Stock symbol
        short_range: List of short SMA periods
        long_range: List of long SMA periods
        metric: Optimization metric
        
    Returns:
        DataFrame with optimization results
    """
    logger = get_logger("optimizer")
    cfg = get_config()
    
    # Load data
    price_path = os.path.join(cfg.price_dir, f"{symbol}.csv")
    
    if not os.path.exists(price_path):
        logger.error(f"❌ Price file not found: {price_path}")
        return pd.DataFrame()
    
    df = pd.read_csv(price_path, parse_dates=['time'])
    
    if 'close' not in df.columns:
        logger.error(f"❌ 'close' column not found in data")
        return pd.DataFrame()
    
    # Use default ranges if not provided
    if short_range is None:
        short_range = cfg.get('optimization.sma_short_range', [5, 10, 15, 20])
    
    if long_range is None:
        long_range = cfg.get('optimization.sma_long_range', [30, 50, 100, 200])
    
    # Create optimizer
    optimizer = ParameterOptimizer(
        short_range=short_range,
        long_range=long_range,
        optimize_metric=metric,
        initial_capital=cfg.initial_capital
    )
    
    # Run optimization
    results_df = optimizer.optimize(df, symbol)
    
    # Save results
    if len(results_df) > 0:
        output_file = f"optimization_{symbol}.csv"
        results_df.to_csv(output_file, index=False)
        logger.info(f"✅ Results saved to: {output_file}")
    
    return results_df


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Optimize SMA Parameters")
    parser.add_argument("symbol", type=str, help="Stock symbol (e.g., MSN)")
    parser.add_argument("--short-range", type=str, help="Short SMA range (comma-separated, e.g., 5,10,15,20)")
    parser.add_argument("--long-range", type=str, help="Long SMA range (comma-separated, e.g., 30,50,100,200)")
    parser.add_argument("--metric", type=str, default="sharpe_ratio",
                       choices=['sharpe_ratio', 'total_return', 'win_rate', 'sortino_ratio', 'calmar_ratio', 'profit_factor'],
                       help="Optimization metric")
    
    args = parser.parse_args()
    
    symbol = args.symbol.upper()
    
    # Parse ranges
    short_range = None
    long_range = None
    
    if args.short_range:
        short_range = [int(x) for x in args.short_range.split(',')]
    
    if args.long_range:
        long_range = [int(x) for x in args.long_range.split(',')]
    
    # Run optimization
    results = optimize_symbol(
        symbol=symbol,
        short_range=short_range,
        long_range=long_range,
        metric=args.metric
    )


if __name__ == "__main__":
    main()

