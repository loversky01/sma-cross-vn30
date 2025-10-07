"""
monte_carlo.py
────────────────────────────────────────────────────────────
Monte Carlo simulation để ước tính rủi ro và khả năng lợi nhuận

Mô phỏng ngẫu nhiên N lần để tính:
• Phân phối lợi nhuận có thể xảy ra
• Xác suất đạt mục tiêu lợi nhuận
• Value at Risk (VaR)
• Conditional Value at Risk (CVaR)

Usage:
    python monte_carlo.py MSN
    python monte_carlo.py FPT --runs 1000 --confidence 95
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
from tqdm import tqdm

from utils.config_loader import get_config
from utils.logger import get_logger


class MonteCarloSimulator:
    """Monte Carlo simulation for strategy analysis"""
    
    def __init__(
        self,
        num_runs: int = 1000,
        confidence_level: float = 95.0
    ):
        """
        Initialize simulator
        
        Args:
            num_runs: Number of simulation runs
            confidence_level: Confidence level for VaR (%)
        """
        self.num_runs = num_runs
        self.confidence_level = confidence_level
        self.logger = get_logger("monte_carlo")
    
    def simulate_from_trades(self, trade_returns: List[float], num_trades: int) -> List[float]:
        """
        Simulate portfolio returns by randomly sampling trades
        
        Args:
            trade_returns: Historical trade returns (%)
            num_trades: Number of trades to simulate
            
        Returns:
            List of simulated total returns
        """
        if len(trade_returns) == 0:
            return []
        
        simulated_returns = []
        
        for _ in range(self.num_runs):
            # Random sample with replacement
            sampled_returns = np.random.choice(trade_returns, size=num_trades, replace=True)
            
            # Calculate cumulative return
            cumulative_return = np.prod(1 + np.array(sampled_returns) / 100) - 1
            total_return_pct = cumulative_return * 100
            
            simulated_returns.append(total_return_pct)
        
        return simulated_returns
    
    def calculate_var(self, returns: List[float]) -> float:
        """
        Calculate Value at Risk (VaR)
        
        Args:
            returns: List of returns
            
        Returns:
            VaR at confidence level
        """
        if len(returns) == 0:
            return 0.0
        
        percentile = 100 - self.confidence_level
        var = np.percentile(returns, percentile)
        
        return var
    
    def calculate_cvar(self, returns: List[float]) -> float:
        """
        Calculate Conditional Value at Risk (CVaR / Expected Shortfall)
        
        Args:
            returns: List of returns
            
        Returns:
            CVaR at confidence level
        """
        if len(returns) == 0:
            return 0.0
        
        var = self.calculate_var(returns)
        
        # Average of returns below VaR
        returns_below_var = [r for r in returns if r <= var]
        
        if len(returns_below_var) == 0:
            return var
        
        cvar = np.mean(returns_below_var)
        
        return cvar
    
    def analyze(self, trade_returns: List[float], num_trades: int = None) -> Dict:
        """
        Run Monte Carlo analysis
        
        Args:
            trade_returns: Historical trade returns (%)
            num_trades: Number of trades to simulate (default: same as historical)
            
        Returns:
            Dictionary of analysis results
        """
        if len(trade_returns) == 0:
            self.logger.error("❌ No trade data provided")
            return {}
        
        if num_trades is None:
            num_trades = len(trade_returns)
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"🎲 Monte Carlo Simulation")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Number of runs: {self.num_runs:,}")
        self.logger.info(f"Historical trades: {len(trade_returns)}")
        self.logger.info(f"Simulated trades per run: {num_trades}")
        self.logger.info(f"Confidence level: {self.confidence_level}%")
        
        # Run simulation
        self.logger.info("\nRunning simulations...")
        simulated_returns = self.simulate_from_trades(trade_returns, num_trades)
        
        # Calculate statistics
        mean_return = np.mean(simulated_returns)
        median_return = np.median(simulated_returns)
        std_return = np.std(simulated_returns)
        min_return = np.min(simulated_returns)
        max_return = np.max(simulated_returns)
        
        # Calculate VaR and CVaR
        var = self.calculate_var(simulated_returns)
        cvar = self.calculate_cvar(simulated_returns)
        
        # Probability of profit
        prob_profit = (np.array(simulated_returns) > 0).mean() * 100
        
        # Percentiles
        p5 = np.percentile(simulated_returns, 5)
        p25 = np.percentile(simulated_returns, 25)
        p75 = np.percentile(simulated_returns, 75)
        p95 = np.percentile(simulated_returns, 95)
        
        results = {
            'Num Runs': self.num_runs,
            'Mean Return (%)': mean_return,
            'Median Return (%)': median_return,
            'Std Dev (%)': std_return,
            'Min Return (%)': min_return,
            'Max Return (%)': max_return,
            f'VaR {self.confidence_level}% (%)': var,
            f'CVaR {self.confidence_level}% (%)': cvar,
            'Probability of Profit (%)': prob_profit,
            'P5 (%)': p5,
            'P25 (%)': p25,
            'P75 (%)': p75,
            'P95 (%)': p95,
            'simulated_returns': simulated_returns
        }
        
        # Log results
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"📊 MONTE CARLO RESULTS")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Mean Return: {mean_return:.2f}%")
        self.logger.info(f"Median Return: {median_return:.2f}%")
        self.logger.info(f"Std Dev: {std_return:.2f}%")
        self.logger.info(f"Min Return: {min_return:.2f}%")
        self.logger.info(f"Max Return: {max_return:.2f}%")
        self.logger.info(f"")
        self.logger.info(f"VaR ({self.confidence_level}%): {var:.2f}%")
        self.logger.info(f"CVaR ({self.confidence_level}%): {cvar:.2f}%")
        self.logger.info(f"")
        self.logger.info(f"Probability of Profit: {prob_profit:.2f}%")
        self.logger.info(f"")
        self.logger.info(f"Percentiles:")
        self.logger.info(f"  5th: {p5:.2f}%")
        self.logger.info(f"  25th: {p25:.2f}%")
        self.logger.info(f"  75th: {p75:.2f}%")
        self.logger.info(f"  95th: {p95:.2f}%")
        self.logger.info(f"{'='*60}\n")
        
        return results
    
    def plot_results(self, results: Dict, symbol: str, save_path: str = None):
        """
        Plot Monte Carlo results
        
        Args:
            results: Results dictionary from analyze()
            symbol: Stock symbol
            save_path: Path to save plot (optional)
        """
        simulated_returns = results.get('simulated_returns', [])
        
        if len(simulated_returns) == 0:
            self.logger.error("No results to plot")
            return
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Histogram
        ax1 = axes[0]
        ax1.hist(simulated_returns, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax1.axvline(results['Mean Return (%)'], color='red', linestyle='--', linewidth=2, label=f"Mean: {results['Mean Return (%)']:.2f}%")
        ax1.axvline(results['Median Return (%)'], color='green', linestyle='--', linewidth=2, label=f"Median: {results['Median Return (%)']:.2f}%")
        ax1.axvline(results[f'VaR {self.confidence_level}% (%)'], color='orange', linestyle='--', linewidth=2, label=f"VaR {self.confidence_level}%: {results[f'VaR {self.confidence_level}% (%)']:.2f}%")
        ax1.set_xlabel('Total Return (%)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title(f'Monte Carlo Simulation - {symbol} ({self.num_runs:,} runs)', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2 = axes[1]
        box = ax2.boxplot(simulated_returns, vert=False, widths=0.5,
                          patch_artist=True,
                          boxprops=dict(facecolor='lightblue', edgecolor='black'),
                          medianprops=dict(color='red', linewidth=2),
                          whiskerprops=dict(color='black', linewidth=1.5),
                          capprops=dict(color='black', linewidth=1.5))
        ax2.set_xlabel('Total Return (%)', fontsize=12)
        ax2.set_title('Distribution of Returns (Box Plot)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.axvline(0, color='black', linestyle='-', linewidth=1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"✅ Plot saved to: {save_path}")
        else:
            plt.show()


def simulate_symbol(symbol: str, num_runs: int = 1000, confidence: float = 95.0) -> Dict:
    """
    Run Monte Carlo simulation for a symbol
    
    Args:
        symbol: Stock symbol
        num_runs: Number of simulation runs
        confidence: Confidence level for VaR
        
    Returns:
        Results dictionary
    """
    logger = get_logger("monte_carlo")
    cfg = get_config()
    
    # Load signal data
    signal_path = os.path.join(cfg.signal_dir, f"{symbol}_signals.csv")
    
    if not os.path.exists(signal_path):
        logger.error(f"❌ Signal file not found: {signal_path}")
        return {}
    
    df = pd.read_csv(signal_path, parse_dates=['time'])
    
    # Extract trade returns (simplified: just take signal points)
    df['returns'] = df['close'].pct_change() * 100
    
    # Get returns at signal points
    buy_signals = df[df['signal'] == 1]
    sell_signals = df[df['signal'] == -1]
    
    if len(buy_signals) == 0 or len(sell_signals) == 0:
        logger.error(f"❌ No signals found for {symbol}")
        return {}
    
    # Calculate trade returns (buy to next sell)
    trade_returns = []
    
    for idx, buy_row in buy_signals.iterrows():
        buy_date = buy_row['time']
        buy_price = buy_row['close']
        
        # Find next sell after this buy
        next_sells = sell_signals[sell_signals['time'] > buy_date]
        
        if len(next_sells) > 0:
            sell_price = next_sells.iloc[0]['close']
            trade_return = ((sell_price - buy_price) / buy_price) * 100
            trade_returns.append(trade_return)
    
    if len(trade_returns) == 0:
        logger.error(f"❌ No complete trades found for {symbol}")
        return {}
    
    logger.info(f"Found {len(trade_returns)} historical trades")
    
    # Run simulation
    simulator = MonteCarloSimulator(num_runs=num_runs, confidence_level=confidence)
    results = simulator.analyze(trade_returns)
    
    # Plot
    if results:
        simulator.plot_results(results, symbol, save_path=f"monte_carlo_{symbol}.png")
    
    return results


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Monte Carlo Simulation")
    parser.add_argument("symbol", type=str, help="Stock symbol (e.g., MSN)")
    parser.add_argument("--runs", type=int, default=1000, help="Number of simulation runs")
    parser.add_argument("--confidence", type=float, default=95.0, help="Confidence level for VaR (%)")
    
    args = parser.parse_args()
    
    symbol = args.symbol.upper()
    
    # Run simulation
    results = simulate_symbol(
        symbol=symbol,
        num_runs=args.runs,
        confidence=args.confidence
    )


if __name__ == "__main__":
    main()

