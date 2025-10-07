"""
metrics.py
────────────────────────────────────────────────────────────
Advanced backtest metrics calculation
Including Sharpe, Sortino, Calmar, Profit Factor, etc.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional


def calculate_returns(prices: pd.Series) -> pd.Series:
    """Calculate returns from prices"""
    return prices.pct_change().dropna()


def calculate_total_return(equity_curve: pd.Series) -> float:
    """
    Calculate total return (%)
    
    Args:
        equity_curve: Series of equity values
        
    Returns:
        Total return percentage
    """
    if len(equity_curve) < 2:
        return 0.0
    
    return ((equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1) * 100


def calculate_annualized_return(equity_curve: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate annualized return
    
    Args:
        equity_curve: Series of equity values
        periods_per_year: Number of periods per year (252 for daily data)
        
    Returns:
        Annualized return percentage
    """
    if len(equity_curve) < 2:
        return 0.0
    
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0]
    n_periods = len(equity_curve)
    
    annualized = (total_return ** (periods_per_year / n_periods) - 1) * 100
    return annualized


def calculate_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate annualized volatility
    
    Args:
        returns: Series of returns
        periods_per_year: Number of periods per year
        
    Returns:
        Annualized volatility percentage
    """
    if len(returns) < 2:
        return 0.0
    
    return returns.std() * np.sqrt(periods_per_year) * 100


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, 
                          periods_per_year: int = 252) -> float:
    """
    Calculate Sharpe Ratio
    
    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate (default 0%)
        periods_per_year: Number of periods per year
        
    Returns:
        Sharpe ratio
    """
    if len(returns) < 2:
        return 0.0
    
    excess_returns = returns - (risk_free_rate / periods_per_year)
    
    if excess_returns.std() == 0:
        return 0.0
    
    return (excess_returns.mean() / excess_returns.std()) * np.sqrt(periods_per_year)


def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0,
                            periods_per_year: int = 252) -> float:
    """
    Calculate Sortino Ratio (uses downside deviation instead of total volatility)
    
    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year
        
    Returns:
        Sortino ratio
    """
    if len(returns) < 2:
        return 0.0
    
    excess_returns = returns - (risk_free_rate / periods_per_year)
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0.0
    
    downside_std = downside_returns.std() * np.sqrt(periods_per_year)
    
    return (excess_returns.mean() * periods_per_year) / downside_std


def calculate_max_drawdown(equity_curve: pd.Series) -> Tuple[float, int, int]:
    """
    Calculate maximum drawdown
    
    Args:
        equity_curve: Series of equity values
        
    Returns:
        Tuple of (max_drawdown_pct, start_idx, end_idx)
    """
    if len(equity_curve) < 2:
        return 0.0, 0, 0
    
    # Calculate running maximum
    running_max = equity_curve.expanding().max()
    
    # Calculate drawdown
    drawdown = (equity_curve - running_max) / running_max * 100
    
    # Find maximum drawdown
    max_dd = drawdown.min()
    max_dd_idx = drawdown.idxmin()
    
    # Find start of drawdown (last peak before max drawdown)
    if max_dd_idx == equity_curve.index[0] or len(equity_curve[:max_dd_idx]) == 0:
        # No drawdown or drawdown starts at beginning
        start_idx = max_dd_idx
    else:
        start_idx = equity_curve[:max_dd_idx].idxmax()
    
    return max_dd, start_idx, max_dd_idx


def calculate_max_drawdown_duration(equity_curve: pd.Series) -> int:
    """
    Calculate maximum drawdown duration (in periods)
    
    Args:
        equity_curve: Series of equity values
        
    Returns:
        Maximum number of periods underwater
    """
    if len(equity_curve) < 2:
        return 0
    
    running_max = equity_curve.expanding().max()
    is_underwater = equity_curve < running_max
    
    # Find consecutive underwater periods
    underwater_periods = []
    current_period = 0
    
    for underwater in is_underwater:
        if underwater:
            current_period += 1
        else:
            if current_period > 0:
                underwater_periods.append(current_period)
            current_period = 0
    
    if current_period > 0:
        underwater_periods.append(current_period)
    
    return max(underwater_periods) if underwater_periods else 0


def calculate_calmar_ratio(equity_curve: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate Calmar Ratio (Annualized Return / Max Drawdown)
    
    Args:
        equity_curve: Series of equity values
        periods_per_year: Number of periods per year
        
    Returns:
        Calmar ratio
    """
    if len(equity_curve) < 2:
        return 0.0
    
    ann_return = calculate_annualized_return(equity_curve, periods_per_year)
    max_dd, _, _ = calculate_max_drawdown(equity_curve)
    
    if max_dd == 0:
        return 0.0
    
    return ann_return / abs(max_dd)


def calculate_win_rate(trade_returns: List[float]) -> float:
    """
    Calculate win rate (percentage of winning trades)
    
    Args:
        trade_returns: List of trade returns (in percentage or absolute)
        
    Returns:
        Win rate percentage
    """
    if len(trade_returns) == 0:
        return 0.0
    
    wins = sum(1 for r in trade_returns if r > 0)
    return (wins / len(trade_returns)) * 100


def calculate_profit_factor(trade_returns: List[float]) -> float:
    """
    Calculate profit factor (Total Wins / Total Losses)
    
    Args:
        trade_returns: List of trade returns
        
    Returns:
        Profit factor
    """
    if len(trade_returns) == 0:
        return 0.0
    
    wins = sum(r for r in trade_returns if r > 0)
    losses = abs(sum(r for r in trade_returns if r < 0))
    
    if losses == 0:
        return float('inf') if wins > 0 else 0.0
    
    return wins / losses


def calculate_average_win_loss_ratio(trade_returns: List[float]) -> float:
    """
    Calculate average win / average loss ratio
    
    Args:
        trade_returns: List of trade returns
        
    Returns:
        Average win/loss ratio
    """
    if len(trade_returns) == 0:
        return 0.0
    
    wins = [r for r in trade_returns if r > 0]
    losses = [abs(r) for r in trade_returns if r < 0]
    
    if len(losses) == 0:
        return float('inf') if len(wins) > 0 else 0.0
    
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    
    if avg_loss == 0:
        return float('inf') if avg_win > 0 else 0.0
    
    return avg_win / avg_loss


def calculate_consecutive_wins_losses(trade_returns: List[float]) -> Tuple[int, int]:
    """
    Calculate maximum consecutive wins and losses
    
    Args:
        trade_returns: List of trade returns
        
    Returns:
        Tuple of (max_consecutive_wins, max_consecutive_losses)
    """
    if len(trade_returns) == 0:
        return 0, 0
    
    max_wins = 0
    max_losses = 0
    current_wins = 0
    current_losses = 0
    
    for ret in trade_returns:
        if ret > 0:
            current_wins += 1
            current_losses = 0
            max_wins = max(max_wins, current_wins)
        elif ret < 0:
            current_losses += 1
            current_wins = 0
            max_losses = max(max_losses, current_losses)
        else:
            current_wins = 0
            current_losses = 0
    
    return max_wins, max_losses


def calculate_all_metrics(equity_curve: pd.Series, trade_returns: List[float],
                         initial_capital: float, risk_free_rate: float = 0.0,
                         periods_per_year: int = 252) -> Dict[str, float]:
    """
    Calculate all backtest metrics
    
    Args:
        equity_curve: Series of equity values over time
        trade_returns: List of individual trade returns
        initial_capital: Initial capital
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year
        
    Returns:
        Dictionary of all metrics
    """
    # Convert equity curve to returns for some calculations
    returns = calculate_returns(equity_curve)
    
    # Return metrics
    total_return = calculate_total_return(equity_curve)
    ann_return = calculate_annualized_return(equity_curve, periods_per_year)
    
    # Risk metrics
    volatility = calculate_volatility(returns, periods_per_year)
    max_dd, _, _ = calculate_max_drawdown(equity_curve)
    max_dd_duration = calculate_max_drawdown_duration(equity_curve)
    
    # Risk-adjusted metrics
    sharpe = calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year)
    sortino = calculate_sortino_ratio(returns, risk_free_rate, periods_per_year)
    calmar = calculate_calmar_ratio(equity_curve, periods_per_year)
    
    # Trade metrics
    total_trades = len(trade_returns)
    win_rate = calculate_win_rate(trade_returns)
    profit_factor = calculate_profit_factor(trade_returns)
    avg_win_loss = calculate_average_win_loss_ratio(trade_returns)
    max_consec_wins, max_consec_losses = calculate_consecutive_wins_losses(trade_returns)
    
    # Best/worst trades
    best_trade = max(trade_returns) if trade_returns else 0
    worst_trade = min(trade_returns) if trade_returns else 0
    avg_trade = np.mean(trade_returns) if trade_returns else 0
    
    # Final capital
    final_capital = equity_curve.iloc[-1] if len(equity_curve) > 0 else initial_capital
    
    return {
        # Return metrics
        'Total Return (%)': round(total_return, 2),
        'Annualized Return (%)': round(ann_return, 2),
        'Final Capital': round(final_capital, 2),
        
        # Risk metrics
        'Volatility (%)': round(volatility, 2),
        'Max Drawdown (%)': round(max_dd, 2),
        'Max DD Duration (periods)': max_dd_duration,
        
        # Risk-adjusted metrics
        'Sharpe Ratio': round(sharpe, 3),
        'Sortino Ratio': round(sortino, 3),
        'Calmar Ratio': round(calmar, 3),
        
        # Trade metrics
        'Total Trades': total_trades,
        'Win Rate (%)': round(win_rate, 2),
        'Profit Factor': round(profit_factor, 3),
        'Avg Win/Loss Ratio': round(avg_win_loss, 3),
        
        # Consecutive trades
        'Max Consecutive Wins': max_consec_wins,
        'Max Consecutive Losses': max_consec_losses,
        
        # Trade statistics
        'Best Trade (%)': round(best_trade, 2),
        'Worst Trade (%)': round(worst_trade, 2),
        'Average Trade (%)': round(avg_trade, 2),
    }


# ────────────────────────────────────────────────────────────
# Example usage
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Sample data
    np.random.seed(42)
    
    # Generate sample equity curve
    initial = 100_000_000
    returns = np.random.normal(0.001, 0.02, 252)  # 1 year of daily returns
    equity = initial * (1 + returns).cumprod()
    equity_curve = pd.Series(equity)
    
    # Generate sample trades
    trade_returns = [5.2, -2.1, 8.5, -3.2, 10.1, -1.5, 6.3, -4.1, 9.2, -2.8]
    
    # Calculate all metrics
    metrics = calculate_all_metrics(
        equity_curve=equity_curve,
        trade_returns=trade_returns,
        initial_capital=initial,
        risk_free_rate=0.03  # 3% risk-free rate
    )
    
    # Print results
    print("=" * 60)
    print("BACKTEST METRICS")
    print("=" * 60)
    for key, value in metrics.items():
        print(f"{key:30s}: {value}")
    print("=" * 60)

