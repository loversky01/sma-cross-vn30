"""
test_metrics.py
────────────────────────────────────────────────────────────
Unit tests for backtest metrics

Run:
    pytest tests/test_metrics.py
    pytest tests/test_metrics.py -v
"""

import pytest
import pandas as pd
import numpy as np
from utils.metrics import (
    calculate_total_return,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_win_rate,
    calculate_profit_factor,
    calculate_all_metrics
)


@pytest.fixture
def sample_equity_curve():
    """Sample equity curve"""
    # Starting at 100M, growing to 150M
    values = np.linspace(100_000_000, 150_000_000, 252)
    return pd.Series(values)


@pytest.fixture
def sample_trade_returns():
    """Sample trade returns"""
    return [5.2, -2.1, 8.5, -3.2, 10.1, -1.5, 6.3, -4.1, 9.2, -2.8]


class TestReturnMetrics:
    """Test return calculations"""
    
    def test_total_return_positive(self):
        """Test positive return"""
        equity = pd.Series([100, 110, 120, 130])
        ret = calculate_total_return(equity)
        
        assert ret == pytest.approx(30.0, rel=1e-6)  # 30% return
    
    def test_total_return_negative(self):
        """Test negative return"""
        equity = pd.Series([100, 90, 80, 70])
        ret = calculate_total_return(equity)
        
        assert ret == pytest.approx(-30.0, rel=1e-6)  # -30% return
    
    def test_total_return_flat(self):
        """Test flat return"""
        equity = pd.Series([100, 100, 100, 100])
        ret = calculate_total_return(equity)
        
        assert ret == pytest.approx(0.0, rel=1e-6)


class TestSharpeRatio:
    """Test Sharpe Ratio calculations"""
    
    def test_sharpe_positive_returns(self):
        """Test Sharpe with positive returns"""
        returns = pd.Series([0.01, 0.02, 0.015, 0.018, 0.022] * 50)  # 250 days
        sharpe = calculate_sharpe_ratio(returns)
        
        assert sharpe > 0  # Should be positive
    
    def test_sharpe_negative_returns(self):
        """Test Sharpe with negative returns"""
        returns = pd.Series([-0.01, -0.02, -0.015] * 84)  # 252 days
        sharpe = calculate_sharpe_ratio(returns)
        
        assert sharpe < 0  # Should be negative
    
    def test_sharpe_zero_volatility(self):
        """Test Sharpe with zero volatility"""
        returns = pd.Series([0.0] * 252)
        sharpe = calculate_sharpe_ratio(returns)
        
        assert sharpe == 0.0


class TestMaxDrawdown:
    """Test Maximum Drawdown calculations"""
    
    def test_max_drawdown_basic(self):
        """Test basic max drawdown"""
        equity = pd.Series([100, 110, 105, 90, 95, 100])
        max_dd, start_idx, end_idx = calculate_max_drawdown(equity)
        
        assert max_dd < 0  # Should be negative
        assert abs(max_dd) > 0  # Should have some drawdown
    
    def test_max_drawdown_no_loss(self):
        """Test with no losses (only gains)"""
        equity = pd.Series([100, 110, 120, 130, 140])
        max_dd, start_idx, end_idx = calculate_max_drawdown(equity)
        
        assert max_dd == pytest.approx(0.0, abs=1e-6)
    
    def test_max_drawdown_all_loss(self):
        """Test with continuous losses"""
        equity = pd.Series([100, 90, 80, 70, 60])
        max_dd, start_idx, end_idx = calculate_max_drawdown(equity)
        
        assert max_dd == pytest.approx(-40.0, rel=1e-6)  # -40% from 100 to 60


class TestTradeMetrics:
    """Test trade-level metrics"""
    
    def test_win_rate(self, sample_trade_returns):
        """Test win rate calculation"""
        win_rate = calculate_win_rate(sample_trade_returns)
        
        # sample_trade_returns = [5.2, -2.1, 8.5, -3.2, 10.1, -1.5, 6.3, -4.1, 9.2, -2.8]
        # Wins: 5.2, 8.5, 10.1, 6.3, 9.2 = 5 wins
        # 5 wins out of 10 trades = 50%
        assert win_rate == 50.0
    
    def test_win_rate_all_wins(self):
        """Test win rate with all winning trades"""
        trades = [1, 2, 3, 4, 5]
        win_rate = calculate_win_rate(trades)
        
        assert win_rate == 100.0
    
    def test_win_rate_all_losses(self):
        """Test win rate with all losing trades"""
        trades = [-1, -2, -3, -4, -5]
        win_rate = calculate_win_rate(trades)
        
        assert win_rate == 0.0
    
    def test_profit_factor(self, sample_trade_returns):
        """Test profit factor calculation"""
        pf = calculate_profit_factor(sample_trade_returns)
        
        wins = sum(r for r in sample_trade_returns if r > 0)
        losses = abs(sum(r for r in sample_trade_returns if r < 0))
        expected_pf = wins / losses
        
        assert pf == pytest.approx(expected_pf, rel=1e-6)
    
    def test_profit_factor_no_losses(self):
        """Test profit factor with no losses"""
        trades = [1, 2, 3, 4, 5]
        pf = calculate_profit_factor(trades)
        
        assert pf == float('inf')


class TestAllMetrics:
    """Test comprehensive metrics calculation"""
    
    def test_all_metrics_complete(self, sample_equity_curve, sample_trade_returns):
        """Test that all metrics are calculated"""
        metrics = calculate_all_metrics(
            equity_curve=sample_equity_curve,
            trade_returns=sample_trade_returns,
            initial_capital=100_000_000
        )
        
        # Check all expected keys are present
        expected_keys = [
            'Total Return (%)',
            'Annualized Return (%)',
            'Sharpe Ratio',
            'Max Drawdown (%)',
            'Win Rate (%)',
            'Total Trades',
            'Profit Factor'
        ]
        
        for key in expected_keys:
            assert key in metrics
    
    def test_all_metrics_values_reasonable(self, sample_equity_curve, sample_trade_returns):
        """Test that metric values are reasonable"""
        metrics = calculate_all_metrics(
            equity_curve=sample_equity_curve,
            trade_returns=sample_trade_returns,
            initial_capital=100_000_000
        )
        
        # Win rate should be 0-100
        assert 0 <= metrics['Win Rate (%)'] <= 100
        
        # Total trades should match input
        assert metrics['Total Trades'] == len(sample_trade_returns)
        
        # Profit factor should be positive
        assert metrics['Profit Factor'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

