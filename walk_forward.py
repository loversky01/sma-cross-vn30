"""
walk_forward.py
────────────────────────────────────────────────────────────
Walk-Forward Analysis để tránh overfitting

Chia dữ liệu thành:
• Training period: Tối ưu tham số
• Testing period: Kiểm tra out-of-sample

Usage:
    python walk_forward.py MSN
    python walk_forward.py FPT --train-pct 70 --test-pct 30
"""

import os
import argparse
import pandas as pd
import numpy as np
from typing import Tuple, Dict

from utils.config_loader import get_config
from utils.logger import get_logger
from utils.indicators import generate_sma_signals
from backtest_advanced import AdvancedBacktester
from optimize_parameters import ParameterOptimizer


class WalkForwardAnalyzer:
    """Walk-forward analysis"""
    
    def __init__(
        self,
        train_pct: float = 70.0,
        test_pct: float = 30.0,
        short_range: list = None,
        long_range: list = None,
        optimize_metric: str = "sharpe_ratio"
    ):
        """
        Initialize analyzer
        
        Args:
            train_pct: Training period percentage
            test_pct: Testing period percentage
            short_range: Short SMA range for optimization
            long_range: Long SMA range for optimization
            optimize_metric: Metric to optimize
        """
        self.train_pct = train_pct
        self.test_pct = test_pct
        self.short_range = short_range or [5, 10, 15, 20]
        self.long_range = long_range or [30, 50, 100, 200]
        self.optimize_metric = optimize_metric
        
        self.logger = get_logger("walk_forward")
        
        # Load config
        cfg = get_config()
        self.backtest_config = {
            'initial_capital': cfg.initial_capital,
            'commission_pct': cfg.commission_pct,
            'tax_pct': cfg.tax_pct,
            'slippage_pct': cfg.slippage_pct,
            'stop_loss_pct': cfg.stop_loss_pct if cfg.use_stop_loss else None,
            'take_profit_pct': cfg.take_profit_pct if cfg.use_take_profit else None,
            'trailing_stop_pct': None,
            'position_size_pct': 100.0
        }
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test sets
        
        Args:
            df: Full DataFrame
            
        Returns:
            Tuple of (train_df, test_df)
        """
        split_idx = int(len(df) * self.train_pct / 100)
        
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        return train_df, test_df
    
    def find_best_params(self, train_df: pd.DataFrame) -> Tuple[int, int]:
        """
        Find best parameters on training data
        
        Args:
            train_df: Training DataFrame
            
        Returns:
            Tuple of (best_short, best_long)
        """
        self.logger.info(f"🔍 Optimizing parameters on training data ({len(train_df)} rows)...")
        
        optimizer = ParameterOptimizer(
            short_range=self.short_range,
            long_range=self.long_range,
            optimize_metric=self.optimize_metric,
            initial_capital=self.backtest_config['initial_capital']
        )
        
        results_df = optimizer.optimize(train_df, "Training")
        
        if len(results_df) == 0:
            self.logger.error("❌ Optimization failed")
            return 10, 50  # Default
        
        best = results_df.iloc[0]
        best_short = int(best['short'])
        best_long = int(best['long'])
        
        self.logger.info(f"✅ Best parameters: SMA({best_short}, {best_long})")
        self.logger.info(f"   Training {self.optimize_metric}: {best[self.optimize_metric]:.3f}")
        
        return best_short, best_long
    
    def backtest_on_test(self, test_df: pd.DataFrame, short: int, long: int) -> Dict:
        """
        Backtest optimized parameters on test data
        
        Args:
            test_df: Test DataFrame
            short: Short SMA period
            long: Long SMA period
            
        Returns:
            Dictionary of test results
        """
        self.logger.info(f"\n📊 Testing on out-of-sample data ({len(test_df)} rows)...")
        
        # Generate signals
        df_signals = generate_sma_signals(test_df, short_window=short, long_window=long)
        
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
        
        return results
    
    def analyze(self, df: pd.DataFrame, symbol: str) -> Dict:
        """
        Run walk-forward analysis
        
        Args:
            df: Full DataFrame with price data
            symbol: Stock symbol
            
        Returns:
            Dictionary of results
        """
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"🚶 Walk-Forward Analysis: {symbol}")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Total data points: {len(df)}")
        self.logger.info(f"Training: {self.train_pct}%")
        self.logger.info(f"Testing: {self.test_pct}%")
        
        # Split data
        train_df, test_df = self.split_data(df)
        
        self.logger.info(f"Training period: {train_df.iloc[0]['time'].date()} to {train_df.iloc[-1]['time'].date()}")
        self.logger.info(f"Testing period: {test_df.iloc[0]['time'].date()} to {test_df.iloc[-1]['time'].date()}")
        
        # Optimize on training data
        best_short, best_long = self.find_best_params(train_df)
        
        # Test on out-of-sample data
        test_results = self.backtest_on_test(test_df, best_short, best_long)
        
        # Log results
        if test_results and 'metrics' in test_results:
            metrics = test_results['metrics']
            
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"📊 OUT-OF-SAMPLE PERFORMANCE")
            self.logger.info(f"{'='*60}")
            self.logger.info(f"Parameters: SMA({best_short}, {best_long})")
            self.logger.info(f"Total Return: {metrics.get('Total Return (%)', 0):.2f}%")
            self.logger.info(f"Sharpe Ratio: {metrics.get('Sharpe Ratio', 0):.3f}")
            self.logger.info(f"Win Rate: {metrics.get('Win Rate (%)', 0):.2f}%")
            self.logger.info(f"Max Drawdown: {metrics.get('Max Drawdown (%)', 0):.2f}%")
            self.logger.info(f"Total Trades: {metrics.get('Total Trades', 0)}")
            self.logger.info(f"{'='*60}\n")
        
        return {
            'symbol': symbol,
            'best_short': best_short,
            'best_long': best_long,
            'train_period': (train_df.iloc[0]['time'], train_df.iloc[-1]['time']),
            'test_period': (test_df.iloc[0]['time'], test_df.iloc[-1]['time']),
            'test_results': test_results
        }


def walk_forward_symbol(
    symbol: str,
    train_pct: float = 70.0,
    test_pct: float = 30.0
) -> Dict:
    """
    Run walk-forward analysis for a symbol
    
    Args:
        symbol: Stock symbol
        train_pct: Training percentage
        test_pct: Testing percentage
        
    Returns:
        Results dictionary
    """
    logger = get_logger("walk_forward")
    cfg = get_config()
    
    # Load data
    price_path = os.path.join(cfg.price_dir, f"{symbol}.csv")
    
    if not os.path.exists(price_path):
        logger.error(f"❌ Price file not found: {price_path}")
        return {}
    
    df = pd.read_csv(price_path, parse_dates=['time'])
    
    if 'close' not in df.columns:
        logger.error(f"❌ 'close' column not found")
        return {}
    
    # Create analyzer
    analyzer = WalkForwardAnalyzer(
        train_pct=train_pct,
        test_pct=test_pct,
        short_range=cfg.get('optimization.sma_short_range', [5, 10, 15, 20]),
        long_range=cfg.get('optimization.sma_long_range', [30, 50, 100, 200]),
        optimize_metric=cfg.get('optimization.optimize_for', 'sharpe_ratio')
    )
    
    # Run analysis
    results = analyzer.analyze(df, symbol)
    
    # Save results
    if results and 'test_results' in results:
        test_results = results['test_results']
        if 'trades' in test_results:
            output_file = f"walk_forward_{symbol}.csv"
            test_results['trades'].to_csv(output_file, index=False)
            logger.info(f"✅ Results saved to: {output_file}")
    
    return results


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Walk-Forward Analysis")
    parser.add_argument("symbol", type=str, help="Stock symbol (e.g., MSN)")
    parser.add_argument("--train-pct", type=float, default=70.0, help="Training percentage")
    parser.add_argument("--test-pct", type=float, default=30.0, help="Testing percentage")
    
    args = parser.parse_args()
    
    symbol = args.symbol.upper()
    
    # Run walk-forward analysis
    results = walk_forward_symbol(
        symbol=symbol,
        train_pct=args.train_pct,
        test_pct=args.test_pct
    )


if __name__ == "__main__":
    main()

