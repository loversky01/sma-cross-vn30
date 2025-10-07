"""
portfolio_backtest.py
────────────────────────────────────────────────────────────
Multi-stock portfolio backtest với risk management

Features:
• Giao dịch đồng thời nhiều mã cổ phiếu
• Portfolio diversification
• Position sizing
• Correlation management
• Portfolio-level metrics

Usage:
    python portfolio_backtest.py
    python portfolio_backtest.py --max-positions 5 --capital 500000000
"""

import os
import argparse
import pandas as pd
import numpy as np
from typing import List, Dict
from datetime import datetime

from utils.config_loader import get_config
from utils.metrics import calculate_all_metrics
from utils.logger import get_logger


class PortfolioBacktester:
    """Portfolio backtester for multiple stocks"""
    
    def __init__(
        self,
        initial_capital: float = 100_000_000,
        max_positions: int = 5,
        position_size_pct: float = 20.0,  # Per stock
        commission_pct: float = 0.15,
        tax_pct: float = 0.10,
        slippage_pct: float = 0.05,
        stop_loss_pct: float = 5.0,
        take_profit_pct: float = 15.0
    ):
        """
        Initialize portfolio backtester
        
        Args:
            initial_capital: Initial capital
            max_positions: Maximum concurrent positions
            position_size_pct: Position size per stock (%)
            commission_pct: Commission (%)
            tax_pct: Tax (%)
            slippage_pct: Slippage (%)
            stop_loss_pct: Stop loss (%)
            take_profit_pct: Take profit (%)
        """
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.position_size_pct = position_size_pct / 100
        self.commission_pct = commission_pct / 100
        self.tax_pct = tax_pct / 100
        self.slippage_pct = slippage_pct / 100
        self.stop_loss_pct = stop_loss_pct / 100
        self.take_profit_pct = take_profit_pct / 100
        
        # State
        self.cash = initial_capital
        self.positions: Dict[str, Dict] = {}  # symbol -> position info
        self.equity_curve: List[float] = [initial_capital]
        self.equity_dates: List[datetime] = []
        self.all_trades: List[Dict] = []
        
        self.logger = get_logger("portfolio")
    
    def calculate_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value"""
        total_value = self.cash
        
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                total_value += position['shares'] * current_prices[symbol]
        
        return total_value
    
    def can_open_position(self) -> bool:
        """Check if we can open a new position"""
        return len(self.positions) < self.max_positions
    
    def calculate_costs(self, price: float, shares: int, is_buy: bool) -> float:
        """Calculate transaction costs"""
        gross = price * shares
        commission = gross * self.commission_pct
        tax = gross * self.tax_pct if not is_buy else 0
        slippage = gross * self.slippage_pct
        return commission + tax + slippage
    
    def buy(self, symbol: str, price: float, date: datetime) -> bool:
        """Open a position"""
        if not self.can_open_position():
            return False
        
        if symbol in self.positions:
            return False
        
        # Calculate position size
        position_capital = self.cash * self.position_size_pct
        
        # Apply slippage
        actual_price = price * (1 + self.slippage_pct)
        
        # Calculate shares
        shares = int(position_capital / actual_price)
        
        if shares == 0:
            return False
        
        # Calculate costs
        costs = self.calculate_costs(actual_price, shares, is_buy=True)
        total_cost = (actual_price * shares) + costs
        
        if total_cost > self.cash:
            return False
        
        # Update cash
        self.cash -= total_cost
        
        # Open position
        self.positions[symbol] = {
            'shares': shares,
            'entry_price': actual_price,
            'entry_date': date,
            'highest_price': actual_price
        }
        
        self.logger.debug(f"🟢 BUY {symbol}: {shares:,} shares @ {actual_price:,.0f} VND")
        
        return True
    
    def sell(self, symbol: str, price: float, date: datetime, reason: str = "Signal") -> bool:
        """Close a position"""
        if symbol not in self.positions:
            return False
        
        position = self.positions[symbol]
        
        # Apply slippage
        actual_price = price * (1 - self.slippage_pct)
        
        # Calculate costs
        costs = self.calculate_costs(actual_price, position['shares'], is_buy=False)
        
        # Calculate proceeds
        proceeds = (actual_price * position['shares']) - costs
        self.cash += proceeds
        
        # Calculate profit
        invested = position['entry_price'] * position['shares']
        profit = proceeds - invested
        profit_pct = (profit / invested) * 100
        
        # Record trade
        trade = {
            'Symbol': symbol,
            'Entry Date': position['entry_date'],
            'Entry Price': position['entry_price'],
            'Exit Date': date,
            'Exit Price': actual_price,
            'Shares': position['shares'],
            'Profit (VND)': profit,
            'Profit (%)': profit_pct,
            'Exit Reason': reason
        }
        self.all_trades.append(trade)
        
        self.logger.debug(f"🔴 SELL {symbol}: {position['shares']:,} shares @ {actual_price:,.0f} VND - {reason} - P/L: {profit_pct:.2f}%")
        
        # Close position
        del self.positions[symbol]
        
        return True
    
    def check_stop_loss_take_profit(self, symbol: str, price: float, date: datetime):
        """Check stop loss and take profit for a position"""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        
        # Update highest price
        if price > position['highest_price']:
            position['highest_price'] = price
        
        # Check stop loss
        loss_pct = (price - position['entry_price']) / position['entry_price']
        if loss_pct <= -self.stop_loss_pct:
            self.sell(symbol, price, date, "Stop Loss")
            return
        
        # Check take profit
        profit_pct = (price - position['entry_price']) / position['entry_price']
        if profit_pct >= self.take_profit_pct:
            self.sell(symbol, price, date, "Take Profit")
            return
    
    def run(self, data_dict: Dict[str, pd.DataFrame]) -> Dict:
        """
        Run portfolio backtest
        
        Args:
            data_dict: Dictionary of symbol -> DataFrame with [time, close, signal]
            
        Returns:
            Dictionary of results
        """
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"🚀 Starting Portfolio Backtest")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Initial Capital: {self.initial_capital:,.0f} VND")
        self.logger.info(f"Max Positions: {self.max_positions}")
        self.logger.info(f"Position Size: {self.position_size_pct*100:.1f}% per stock")
        self.logger.info(f"Stocks: {len(data_dict)}")
        
        # Get all unique dates
        all_dates = set()
        for df in data_dict.values():
            all_dates.update(df['time'].tolist())
        
        all_dates = sorted(list(all_dates))
        
        # Simulate day by day
        for date in all_dates:
            # Get current prices and signals
            current_prices = {}
            signals = {}
            
            for symbol, df in data_dict.items():
                day_data = df[df['time'] == date]
                if len(day_data) > 0:
                    current_prices[symbol] = day_data.iloc[0]['close']
                    signals[symbol] = day_data.iloc[0].get('signal', 0)
            
            # Check stop loss / take profit for existing positions
            for symbol in list(self.positions.keys()):
                if symbol in current_prices:
                    self.check_stop_loss_take_profit(symbol, current_prices[symbol], date)
            
            # Process sell signals
            for symbol in list(self.positions.keys()):
                if signals.get(symbol) == -1:
                    if symbol in current_prices:
                        self.sell(symbol, current_prices[symbol], date, "Signal")
            
            # Process buy signals
            for symbol, signal in signals.items():
                if signal == 1 and self.can_open_position():
                    if symbol in current_prices:
                        self.buy(symbol, current_prices[symbol], date)
            
            # Track equity
            portfolio_value = self.calculate_portfolio_value(current_prices)
            self.equity_curve.append(portfolio_value)
            self.equity_dates.append(date)
        
        # Close all remaining positions
        last_prices = {}
        for symbol, df in data_dict.items():
            if len(df) > 0:
                last_prices[symbol] = df.iloc[-1]['close']
                last_date = df.iloc[-1]['time']
        
        for symbol in list(self.positions.keys()):
            if symbol in last_prices:
                self.sell(symbol, last_prices[symbol], last_date, "End of Period")
        
        # Calculate results
        return self._calculate_results()
    
    def _calculate_results(self) -> Dict:
        """Calculate portfolio results"""
        
        if len(self.all_trades) == 0:
            self.logger.warning("⚠️  No trades executed")
            return {}
        
        # Trade returns
        trade_returns = [t['Profit (%)'] for t in self.all_trades]
        
        # Equity curve
        equity_series = pd.Series(self.equity_curve)
        
        # Calculate metrics
        metrics = calculate_all_metrics(
            equity_curve=equity_series,
            trade_returns=trade_returns,
            initial_capital=self.initial_capital,
            risk_free_rate=0.03
        )
        
        # Add portfolio-specific metrics
        metrics['Number of Stocks'] = len(set(t['Symbol'] for t in self.all_trades))
        
        # Log results
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"📊 PORTFOLIO RESULTS")
        self.logger.info(f"{'='*60}")
        
        for key, value in metrics.items():
            if isinstance(value, float):
                if 'pct' in key.lower() or 'rate' in key.lower() or '%' in key:
                    self.logger.info(f"  {key}: {value:.2f}%")
                elif 'ratio' in key.lower():
                    self.logger.info(f"  {key}: {value:.3f}")
                else:
                    self.logger.info(f"  {key}: {value:,.2f}")
            else:
                self.logger.info(f"  {key}: {value}")
        
        self.logger.info(f"{'='*60}\n")
        
        # Create trades DataFrame
        trades_df = pd.DataFrame(self.all_trades)
        
        return {
            'metrics': metrics,
            'trades': trades_df,
            'equity_curve': equity_series,
            'equity_dates': self.equity_dates,
            'final_capital': metrics['Final Capital']
        }


def run_portfolio_backtest(
    symbols: List[str] = None,
    max_positions: int = 5,
    initial_capital: float = 100_000_000
) -> Dict:
    """
    Run portfolio backtest for multiple symbols
    
    Args:
        symbols: List of stock symbols (None = all VN30)
        max_positions: Maximum concurrent positions
        initial_capital: Initial capital
        
    Returns:
        Results dictionary
    """
    logger = get_logger("portfolio")
    cfg = get_config()
    
    # Use all VN30 if symbols not provided
    if symbols is None:
        symbols = cfg.vn30_symbols
    
    logger.info(f"Loading data for {len(symbols)} stocks...")
    
    # Load all data
    data_dict = {}
    
    for symbol in symbols:
        signal_path = os.path.join(cfg.signal_dir, f"{symbol}_signals.csv")
        
        if not os.path.exists(signal_path):
            logger.warning(f"⚠️  Signal file not found for {symbol}")
            continue
        
        df = pd.read_csv(signal_path, parse_dates=['time'])
        df = df[['time', 'close', 'signal']].dropna()
        
        if len(df) > 0:
            data_dict[symbol] = df
    
    if len(data_dict) == 0:
        logger.error("❌ No data loaded")
        return {}
    
    logger.info(f"✅ Loaded {len(data_dict)} stocks")
    
    # Create backtester
    backtester = PortfolioBacktester(
        initial_capital=initial_capital,
        max_positions=max_positions,
        position_size_pct=100.0 / max_positions,  # Equal weight
        commission_pct=cfg.commission_pct,
        tax_pct=cfg.tax_pct,
        slippage_pct=cfg.slippage_pct,
        stop_loss_pct=cfg.stop_loss_pct if cfg.use_stop_loss else 5.0,
        take_profit_pct=cfg.take_profit_pct if cfg.use_take_profit else 15.0
    )
    
    # Run backtest
    results = backtester.run(data_dict)
    
    return results


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Portfolio Backtest")
    parser.add_argument("--symbols", type=str, help="Comma-separated symbols (default: all VN30)")
    parser.add_argument("--max-positions", type=int, default=5, help="Maximum concurrent positions")
    parser.add_argument("--capital", type=float, help="Initial capital (VND)")
    
    args = parser.parse_args()
    
    # Parse symbols
    symbols = None
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(',')]
    
    # Get capital
    cfg = get_config()
    capital = args.capital if args.capital else cfg.initial_capital
    
    # Run backtest
    results = run_portfolio_backtest(
        symbols=symbols,
        max_positions=args.max_positions,
        initial_capital=capital
    )
    
    # Save results
    if results and 'trades' in results:
        logger = get_logger("portfolio")
        
        # Save trades
        output_file = "portfolio_backtest_trades.csv"
        results['trades'].to_csv(output_file, index=False)
        logger.info(f"✅ Trades saved to: {output_file}")
        
        # Print summary by symbol
        trades_df = results['trades']
        summary = trades_df.groupby('Symbol').agg({
            'Profit (VND)': 'sum',
            'Profit (%)': 'mean',
            'Symbol': 'count'
        }).rename(columns={'Symbol': 'Total Trades'})
        
        logger.info("\n📊 SUMMARY BY SYMBOL:")
        logger.info("\n" + summary.to_string())


if __name__ == "__main__":
    main()

