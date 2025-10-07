"""
backtest_advanced.py
────────────────────────────────────────────────────────────
Advanced backtest với Stop Loss, Take Profit và metrics đầy đủ

Features:
• Stop Loss & Take Profit
• Trailing Stop (optional)
• Phí giao dịch thực tế (commission, tax, slippage)
• Advanced metrics (Sharpe, Sortino, Calmar, Profit Factor)
• Equity curve tracking
• Detailed trade log

Usage:
    python backtest_advanced.py MSN
    python backtest_advanced.py FPT --capital 200000000
    python backtest_advanced.py VCB --stop-loss 3 --take-profit 10
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional

# Import utils
from utils.config_loader import get_config
from utils.metrics import calculate_all_metrics
from utils.logger import get_logger, log_backtest_result


class AdvancedBacktester:
    """Advanced backtester with risk management"""
    
    def __init__(
        self,
        initial_capital: float = 100_000_000,
        commission_pct: float = 0.15,
        tax_pct: float = 0.10,
        slippage_pct: float = 0.05,
        stop_loss_pct: Optional[float] = None,
        take_profit_pct: Optional[float] = None,
        trailing_stop_pct: Optional[float] = None,
        position_size_pct: float = 100.0
    ):
        """
        Initialize backtester
        
        Args:
            initial_capital: Initial capital in VND
            commission_pct: Commission percentage
            tax_pct: Tax percentage (on sell only)
            slippage_pct: Slippage percentage
            stop_loss_pct: Stop loss percentage (optional)
            take_profit_pct: Take profit percentage (optional)
            trailing_stop_pct: Trailing stop percentage (optional)
            position_size_pct: Position size as percentage of capital
        """
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct / 100
        self.tax_pct = tax_pct / 100
        self.slippage_pct = slippage_pct / 100
        self.stop_loss_pct = stop_loss_pct / 100 if stop_loss_pct else None
        self.take_profit_pct = take_profit_pct / 100 if take_profit_pct else None
        self.trailing_stop_pct = trailing_stop_pct / 100 if trailing_stop_pct else None
        self.position_size_pct = position_size_pct / 100
        
        # State variables
        self.capital = initial_capital
        self.in_position = False
        self.entry_price = 0
        self.entry_date = None
        self.shares = 0
        self.highest_price_since_entry = 0
        
        # Tracking
        self.trades: List[Dict] = []
        self.equity_curve: List[float] = [initial_capital]
        self.equity_dates: List[pd.Timestamp] = []
        
        self.logger = get_logger("backtest_advanced")
    
    def calculate_costs(self, price: float, shares: int, is_buy: bool) -> float:
        """Calculate transaction costs"""
        gross_value = price * shares
        
        # Commission on both buy and sell
        commission = gross_value * self.commission_pct
        
        # Tax only on sell
        tax = gross_value * self.tax_pct if not is_buy else 0
        
        # Slippage
        slippage = gross_value * self.slippage_pct
        
        return commission + tax + slippage
    
    def check_stop_loss(self, current_price: float) -> bool:
        """Check if stop loss is hit"""
        if not self.stop_loss_pct or not self.in_position:
            return False
        
        loss_pct = (current_price - self.entry_price) / self.entry_price
        return loss_pct <= -self.stop_loss_pct
    
    def check_take_profit(self, current_price: float) -> bool:
        """Check if take profit is hit"""
        if not self.take_profit_pct or not self.in_position:
            return False
        
        profit_pct = (current_price - self.entry_price) / self.entry_price
        return profit_pct >= self.take_profit_pct
    
    def check_trailing_stop(self, current_price: float) -> bool:
        """Check if trailing stop is hit"""
        if not self.trailing_stop_pct or not self.in_position:
            return False
        
        # Update highest price
        if current_price > self.highest_price_since_entry:
            self.highest_price_since_entry = current_price
        
        # Check if price dropped from highest
        drop_from_high = (current_price - self.highest_price_since_entry) / self.highest_price_since_entry
        return drop_from_high <= -self.trailing_stop_pct
    
    def buy(self, price: float, date: pd.Timestamp) -> bool:
        """Execute buy order"""
        if self.in_position:
            return False
        
        # Apply slippage (buy at slightly higher price)
        actual_price = price * (1 + self.slippage_pct)
        
        # Calculate position size
        position_capital = self.capital * self.position_size_pct
        
        # Calculate shares (round down to whole shares)
        self.shares = int(position_capital / actual_price)
        
        if self.shares == 0:
            return False
        
        # Calculate costs
        costs = self.calculate_costs(actual_price, self.shares, is_buy=True)
        
        # Update capital
        total_cost = (actual_price * self.shares) + costs
        self.capital -= total_cost
        
        # Update state
        self.in_position = True
        self.entry_price = actual_price
        self.entry_date = date
        self.highest_price_since_entry = actual_price
        
        self.logger.debug(f"🟢 BUY {self.shares:,} shares @ {actual_price:,.0f} VND on {date.date()}")
        
        return True
    
    def sell(self, price: float, date: pd.Timestamp, reason: str = "Signal") -> bool:
        """Execute sell order"""
        if not self.in_position:
            return False
        
        # Apply slippage (sell at slightly lower price)
        actual_price = price * (1 - self.slippage_pct)
        
        # Calculate costs
        costs = self.calculate_costs(actual_price, self.shares, is_buy=False)
        
        # Update capital
        proceeds = (actual_price * self.shares) - costs
        self.capital += proceeds
        
        # Calculate profit/loss
        total_invested = self.entry_price * self.shares
        profit = proceeds - total_invested
        profit_pct = (profit / total_invested) * 100
        
        # Record trade
        trade = {
            'Entry Date': self.entry_date,
            'Entry Price': self.entry_price,
            'Exit Date': date,
            'Exit Price': actual_price,
            'Shares': self.shares,
            'Profit (VND)': profit,
            'Profit (%)': profit_pct,
            'Exit Reason': reason,
            'Capital After': self.capital
        }
        self.trades.append(trade)
        
        self.logger.debug(f"🔴 SELL {self.shares:,} shares @ {actual_price:,.0f} VND on {date.date()} - {reason} - P/L: {profit_pct:.2f}%")
        
        # Reset position
        self.in_position = False
        self.shares = 0
        self.entry_price = 0
        self.entry_date = None
        self.highest_price_since_entry = 0
        
        return True
    
    def run(self, df: pd.DataFrame, symbol: str) -> Dict:
        """
        Run backtest on data
        
        Args:
            df: DataFrame with columns [time, close, signal]
            symbol: Stock symbol
            
        Returns:
            Dictionary of backtest results
        """
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"🚀 Starting backtest: {symbol}")
        self.logger.info(f"{'='*60}")
        
        # Reset state
        self.capital = self.initial_capital
        self.in_position = False
        self.trades = []
        self.equity_curve = [self.initial_capital]
        self.equity_dates = []
        
        for idx, row in df.iterrows():
            date = row['time']
            price = row['close']
            signal = row.get('signal', 0)
            
            # Track equity
            if self.in_position:
                # Mark-to-market value
                current_value = self.capital + (self.shares * price)
            else:
                current_value = self.capital
            
            self.equity_curve.append(current_value)
            self.equity_dates.append(date)
            
            # Check stop loss / take profit / trailing stop
            if self.in_position:
                if self.check_stop_loss(price):
                    self.sell(price, date, "Stop Loss")
                    continue
                
                if self.check_take_profit(price):
                    self.sell(price, date, "Take Profit")
                    continue
                
                if self.check_trailing_stop(price):
                    self.sell(price, date, "Trailing Stop")
                    continue
            
            # Process signals
            if signal == 1 and not self.in_position:
                self.buy(price, date)
            
            elif signal == -1 and self.in_position:
                self.sell(price, date, "Signal")
        
        # Close any open position at the end
        if self.in_position:
            last_row = df.iloc[-1]
            self.sell(last_row['close'], last_row['time'], "End of Period")
        
        # Calculate metrics
        return self._calculate_results(symbol, df)
    
    def _calculate_results(self, symbol: str, df: pd.DataFrame) -> Dict:
        """Calculate backtest results"""
        
        if len(self.trades) == 0:
            self.logger.warning(f"⚠️  No trades executed for {symbol}")
            return {}
        
        # Trade returns (as percentages)
        trade_returns = [t['Profit (%)'] for t in self.trades]
        
        # Equity curve as Series
        equity_series = pd.Series(self.equity_curve)
        
        # Calculate all metrics
        metrics = calculate_all_metrics(
            equity_curve=equity_series,
            trade_returns=trade_returns,
            initial_capital=self.initial_capital,
            risk_free_rate=0.03  # 3% risk-free rate
        )
        
        # Add symbol
        metrics['Symbol'] = symbol
        
        # Log results
        log_backtest_result(self.logger, symbol, metrics)
        
        # Create trades DataFrame
        trades_df = pd.DataFrame(self.trades)
        
        return {
            'symbol': symbol,
            'metrics': metrics,
            'trades': trades_df,
            'equity_curve': equity_series,
            'equity_dates': self.equity_dates,
            'initial_capital': self.initial_capital,
            'final_capital': metrics['Final Capital']
        }


def backtest_symbol(symbol: str, config: Optional[Dict] = None) -> Dict:
    """
    Backtest a single symbol
    
    Args:
        symbol: Stock symbol
        config: Optional config override
        
    Returns:
        Backtest results dictionary
    """
    logger = get_logger("backtest_advanced")
    
    # Load config
    if config is None:
        cfg = get_config()
        config = {
            'initial_capital': cfg.initial_capital,
            'commission_pct': cfg.commission_pct,
            'tax_pct': cfg.tax_pct,
            'slippage_pct': cfg.slippage_pct,
            'stop_loss_pct': cfg.stop_loss_pct if cfg.use_stop_loss else None,
            'take_profit_pct': cfg.take_profit_pct if cfg.use_take_profit else None,
            'trailing_stop_pct': cfg.get('risk_management.trailing_stop_pct') if cfg.get('risk_management.use_trailing_stop') else None,
            'position_size_pct': cfg.get('risk_management.position_size_pct', 100.0),
            'signal_dir': cfg.signal_dir
        }
    
    # Load data
    signal_path = os.path.join(config['signal_dir'], f"{symbol}_signals.csv")
    
    if not os.path.exists(signal_path):
        logger.error(f"❌ Signal file not found: {signal_path}")
        return {}
    
    df = pd.read_csv(signal_path, parse_dates=['time'])
    df = df[['time', 'close', 'signal']].dropna()
    
    if len(df) == 0:
        logger.error(f"❌ No data for {symbol}")
        return {}
    
    # Create backtester
    backtester = AdvancedBacktester(
        initial_capital=config['initial_capital'],
        commission_pct=config['commission_pct'],
        tax_pct=config['tax_pct'],
        slippage_pct=config['slippage_pct'],
        stop_loss_pct=config['stop_loss_pct'],
        take_profit_pct=config['take_profit_pct'],
        trailing_stop_pct=config['trailing_stop_pct'],
        position_size_pct=config['position_size_pct']
    )
    
    # Run backtest
    results = backtester.run(df, symbol)
    
    return results


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Advanced Backtest with Risk Management")
    parser.add_argument("symbol", type=str, help="Stock symbol (e.g., MSN)")
    parser.add_argument("--capital", type=float, help="Initial capital (VND)")
    parser.add_argument("--stop-loss", type=float, help="Stop loss percentage")
    parser.add_argument("--take-profit", type=float, help="Take profit percentage")
    parser.add_argument("--trailing-stop", type=float, help="Trailing stop percentage")
    parser.add_argument("--commission", type=float, help="Commission percentage")
    parser.add_argument("--tax", type=float, help="Tax percentage")
    parser.add_argument("--slippage", type=float, help="Slippage percentage")
    parser.add_argument("--position-size", type=float, help="Position size percentage")
    
    args = parser.parse_args()
    
    symbol = args.symbol.upper()
    
    # Load config
    cfg = get_config()
    
    # Override with command line args
    config = {
        'initial_capital': args.capital if args.capital else cfg.initial_capital,
        'commission_pct': args.commission if args.commission else cfg.commission_pct,
        'tax_pct': args.tax if args.tax else cfg.tax_pct,
        'slippage_pct': args.slippage if args.slippage else cfg.slippage_pct,
        'stop_loss_pct': args.stop_loss if args.stop_loss else (cfg.stop_loss_pct if cfg.use_stop_loss else None),
        'take_profit_pct': args.take_profit if args.take_profit else (cfg.take_profit_pct if cfg.use_take_profit else None),
        'trailing_stop_pct': args.trailing_stop if args.trailing_stop else None,
        'position_size_pct': args.position_size if args.position_size else cfg.get('risk_management.position_size_pct', 100.0),
        'signal_dir': cfg.signal_dir
    }
    
    # Run backtest
    results = backtest_symbol(symbol, config)
    
    # Print trades
    if results and 'trades' in results:
        logger = get_logger("backtest_advanced")
        logger.info("\n📋 DETAILED TRADES:")
        logger.info("\n" + results['trades'].to_string(index=False))
        
        # Save to CSV
        output_file = f"backtest_{symbol}_advanced.csv"
        results['trades'].to_csv(output_file, index=False)
        logger.info(f"\n✅ Trades saved to: {output_file}")


if __name__ == "__main__":
    main()

