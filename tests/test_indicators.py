"""
test_indicators.py
────────────────────────────────────────────────────────────
Unit tests for technical indicators

Run:
    pytest tests/test_indicators.py
    pytest tests/test_indicators.py -v
"""

import pytest
import pandas as pd
import numpy as np
from utils.indicators import (
    calculate_sma, calculate_ema, calculate_rsi,
    calculate_macd, calculate_bollinger_bands,
    generate_sma_signals
)


@pytest.fixture
def sample_prices():
    """Sample price data for testing"""
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(100) * 2)
    return pd.Series(prices)


@pytest.fixture
def sample_df():
    """Sample DataFrame for testing"""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    close = 100 + np.cumsum(np.random.randn(100) * 2)
    
    return pd.DataFrame({
        'time': dates,
        'close': close
    })


class TestSMA:
    """Test SMA calculations"""
    
    def test_sma_basic(self, sample_prices):
        """Test basic SMA calculation"""
        sma = calculate_sma(sample_prices, window=10)
        
        assert len(sma) == len(sample_prices)
        assert sma.iloc[:9].isna().all()  # First 9 values should be NaN
        assert not sma.iloc[9:].isna().any()  # Rest should be valid
    
    def test_sma_values(self):
        """Test SMA values with known data"""
        data = pd.Series([1, 2, 3, 4, 5])
        sma = calculate_sma(data, window=3)
        
        assert np.isnan(sma.iloc[0])
        assert np.isnan(sma.iloc[1])
        assert sma.iloc[2] == 2.0  # (1+2+3)/3
        assert sma.iloc[3] == 3.0  # (2+3+4)/3
        assert sma.iloc[4] == 4.0  # (3+4+5)/3


class TestEMA:
    """Test EMA calculations"""
    
    def test_ema_basic(self, sample_prices):
        """Test basic EMA calculation"""
        ema = calculate_ema(sample_prices, window=10)
        
        assert len(ema) == len(sample_prices)
        assert not ema.isna().all()


class TestRSI:
    """Test RSI calculations"""
    
    def test_rsi_basic(self, sample_prices):
        """Test basic RSI calculation"""
        rsi = calculate_rsi(sample_prices, window=14)
        
        assert len(rsi) == len(sample_prices)
        assert rsi.iloc[14:].between(0, 100).all()  # RSI should be 0-100
    
    def test_rsi_extreme_values(self):
        """Test RSI with extreme values"""
        # All increasing
        data = pd.Series(range(50))
        rsi = calculate_rsi(data, window=14)
        
        assert rsi.iloc[-1] > 70  # Should be overbought


class TestMACD:
    """Test MACD calculations"""
    
    def test_macd_basic(self, sample_prices):
        """Test basic MACD calculation"""
        macd_line, signal_line, macd_hist = calculate_macd(sample_prices)
        
        assert len(macd_line) == len(sample_prices)
        assert len(signal_line) == len(sample_prices)
        assert len(macd_hist) == len(sample_prices)
    
    def test_macd_relationship(self, sample_prices):
        """Test MACD histogram relationship"""
        macd_line, signal_line, macd_hist = calculate_macd(sample_prices)
        
        # MACD histogram = MACD line - Signal line
        calculated_hist = macd_line - signal_line
        
        # Check they're approximately equal (allowing for floating point errors)
        assert np.allclose(
            macd_hist.dropna(),
            calculated_hist.dropna(),
            rtol=1e-10
        )


class TestBollingerBands:
    """Test Bollinger Bands calculations"""
    
    def test_bb_basic(self, sample_prices):
        """Test basic Bollinger Bands calculation"""
        upper, middle, lower = calculate_bollinger_bands(sample_prices)
        
        assert len(upper) == len(sample_prices)
        assert len(middle) == len(sample_prices)
        assert len(lower) == len(sample_prices)
    
    def test_bb_ordering(self, sample_prices):
        """Test that upper > middle > lower"""
        upper, middle, lower = calculate_bollinger_bands(sample_prices)
        
        # After initial NaN period
        valid_data = ~middle.isna()
        
        assert (upper[valid_data] >= middle[valid_data]).all()
        assert (middle[valid_data] >= lower[valid_data]).all()


class TestSMASignals:
    """Test SMA signal generation"""
    
    def test_signal_generation(self, sample_df):
        """Test signal generation"""
        df_signals = generate_sma_signals(sample_df, short_window=10, long_window=20)
        
        assert 'SMA10' in df_signals.columns
        assert 'SMA20' in df_signals.columns
        assert 'signal' in df_signals.columns
    
    def test_signal_values(self, sample_df):
        """Test signal values are valid"""
        df_signals = generate_sma_signals(sample_df, short_window=5, long_window=10)
        
        # Signals should be -1, 0, or 1
        valid_signals = df_signals['signal'].isin([-1, 0, 1])
        assert valid_signals.all()
    
    def test_crossover_signal(self):
        """Test crossover signals with known data"""
        # Create data where we know a crossover will occur
        dates = pd.date_range('2020-01-01', periods=15, freq='D')
        
        # Price goes from 100 down to 80, then up to 120
        prices = [100, 95, 90, 85, 80, 85, 90, 95, 100, 105, 110, 115, 120, 120, 120]
        
        df = pd.DataFrame({
            'time': dates,
            'close': prices
        })
        
        df_signals = generate_sma_signals(df, short_window=3, long_window=5)
        
        # Should have some buy or sell signals
        assert (df_signals['signal'] != 0).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

