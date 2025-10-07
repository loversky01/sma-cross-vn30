"""
indicators.py
────────────────────────────────────────────────────────────
Technical indicators for trading strategies
Including SMA, RSI, MACD, Bollinger Bands, etc.
"""

import numpy as np
import pandas as pd
from typing import Tuple


def calculate_sma(data: pd.Series, window: int) -> pd.Series:
    """
    Calculate Simple Moving Average
    
    Args:
        data: Price series
        window: Moving average window
        
    Returns:
        SMA series
    """
    return data.rolling(window=window).mean()


def calculate_ema(data: pd.Series, window: int) -> pd.Series:
    """
    Calculate Exponential Moving Average
    
    Args:
        data: Price series
        window: Moving average window
        
    Returns:
        EMA series
    """
    return data.ewm(span=window, adjust=False).mean()


def calculate_rsi(data: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI)
    
    Args:
        data: Price series
        window: RSI period (default 14)
        
    Returns:
        RSI series (0-100)
    """
    # Calculate price changes
    delta = data.diff()
    
    # Separate gains and losses
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)
    
    # Calculate average gains and losses
    avg_gains = gains.rolling(window=window).mean()
    avg_losses = losses.rolling(window=window).mean()
    
    # Calculate RS and RSI
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_macd(data: pd.Series, fast: int = 12, slow: int = 26, 
                   signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate MACD (Moving Average Convergence Divergence)
    
    Args:
        data: Price series
        fast: Fast EMA period (default 12)
        slow: Slow EMA period (default 26)
        signal: Signal line period (default 9)
        
    Returns:
        Tuple of (MACD line, Signal line, MACD histogram)
    """
    # Calculate EMAs
    ema_fast = calculate_ema(data, fast)
    ema_slow = calculate_ema(data, slow)
    
    # MACD line
    macd_line = ema_fast - ema_slow
    
    # Signal line
    signal_line = calculate_ema(macd_line, signal)
    
    # MACD histogram
    macd_hist = macd_line - signal_line
    
    return macd_line, signal_line, macd_hist


def calculate_bollinger_bands(data: pd.Series, window: int = 20, 
                              num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands
    
    Args:
        data: Price series
        window: Moving average window (default 20)
        num_std: Number of standard deviations (default 2.0)
        
    Returns:
        Tuple of (Upper band, Middle band, Lower band)
    """
    # Middle band (SMA)
    middle_band = calculate_sma(data, window)
    
    # Standard deviation
    std = data.rolling(window=window).std()
    
    # Upper and lower bands
    upper_band = middle_band + (std * num_std)
    lower_band = middle_band - (std * num_std)
    
    return upper_band, middle_band, lower_band


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, 
                  window: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR)
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        window: ATR period (default 14)
        
    Returns:
        ATR series
    """
    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # ATR (average of True Range)
    atr = tr.rolling(window=window).mean()
    
    return atr


def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                        k_window: int = 14, d_window: int = 3) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate Stochastic Oscillator
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        k_window: %K period (default 14)
        d_window: %D period (default 3)
        
    Returns:
        Tuple of (%K, %D)
    """
    # Lowest low and highest high
    lowest_low = low.rolling(window=k_window).min()
    highest_high = high.rolling(window=k_window).max()
    
    # %K
    k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    
    # %D (SMA of %K)
    d = k.rolling(window=d_window).mean()
    
    return k, d


def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Calculate On-Balance Volume (OBV)
    
    Args:
        close: Close price series
        volume: Volume series
        
    Returns:
        OBV series
    """
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    return obv


def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series,
                 window: int = 14) -> pd.Series:
    """
    Calculate Average Directional Index (ADX)
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        window: ADX period (default 14)
        
    Returns:
        ADX series
    """
    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional movement
    up_move = high - high.shift()
    down_move = low.shift() - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    plus_dm = pd.Series(plus_dm, index=close.index)
    minus_dm = pd.Series(minus_dm, index=close.index)
    
    # Smooth TR and DM
    atr = tr.rolling(window=window).mean()
    plus_di = 100 * (plus_dm.rolling(window=window).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=window).mean() / atr)
    
    # Calculate DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=window).mean()
    
    return adx


def generate_sma_signals(df: pd.DataFrame, short_window: int = 10, 
                        long_window: int = 50) -> pd.DataFrame:
    """
    Generate SMA crossover signals
    
    Args:
        df: DataFrame with 'close' column
        short_window: Short SMA window
        long_window: Long SMA window
        
    Returns:
        DataFrame with SMA columns and signal column
    """
    result = df.copy()
    
    # Calculate SMAs
    result[f'SMA{short_window}'] = calculate_sma(result['close'], short_window)
    result[f'SMA{long_window}'] = calculate_sma(result['close'], long_window)
    
    # Generate signals
    result['signal'] = 0
    
    # Buy signal: short SMA crosses above long SMA
    cross_up = (result[f'SMA{short_window}'] > result[f'SMA{long_window}']) & \
               (result[f'SMA{short_window}'].shift(1) <= result[f'SMA{long_window}'].shift(1))
    
    # Sell signal: short SMA crosses below long SMA
    cross_down = (result[f'SMA{short_window}'] < result[f'SMA{long_window}']) & \
                 (result[f'SMA{short_window}'].shift(1) >= result[f'SMA{long_window}'].shift(1))
    
    result.loc[cross_up, 'signal'] = 1    # Buy
    result.loc[cross_down, 'signal'] = -1  # Sell
    
    return result


def add_all_indicators(df: pd.DataFrame, short_sma: int = 10, long_sma: int = 50,
                       rsi_period: int = 14, macd_params: Tuple[int, int, int] = (12, 26, 9),
                       bb_window: int = 20) -> pd.DataFrame:
    """
    Add all technical indicators to DataFrame
    
    Args:
        df: DataFrame with OHLCV data
        short_sma: Short SMA window
        long_sma: Long SMA window
        rsi_period: RSI period
        macd_params: MACD parameters (fast, slow, signal)
        bb_window: Bollinger Bands window
        
    Returns:
        DataFrame with all indicators
    """
    result = df.copy()
    
    # SMAs
    result[f'SMA{short_sma}'] = calculate_sma(result['close'], short_sma)
    result[f'SMA{long_sma}'] = calculate_sma(result['close'], long_sma)
    
    # RSI
    result['RSI'] = calculate_rsi(result['close'], rsi_period)
    
    # MACD
    macd_line, signal_line, macd_hist = calculate_macd(
        result['close'], *macd_params
    )
    result['MACD'] = macd_line
    result['MACD_signal'] = signal_line
    result['MACD_hist'] = macd_hist
    
    # Bollinger Bands
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(
        result['close'], bb_window
    )
    result['BB_upper'] = bb_upper
    result['BB_middle'] = bb_middle
    result['BB_lower'] = bb_lower
    
    # Add volume-based indicators if volume column exists
    if 'volume' in result.columns:
        result['OBV'] = calculate_obv(result['close'], result['volume'])
    
    # Add ATR if OHLC data available
    if all(col in result.columns for col in ['high', 'low', 'close']):
        result['ATR'] = calculate_atr(result['high'], result['low'], result['close'])
        result['ADX'] = calculate_adx(result['high'], result['low'], result['close'])
        
        stoch_k, stoch_d = calculate_stochastic(
            result['high'], result['low'], result['close']
        )
        result['Stoch_K'] = stoch_k
        result['Stoch_D'] = stoch_d
    
    return result


def filter_signals_with_rsi(signals: pd.Series, rsi: pd.Series,
                            oversold: float = 30, overbought: float = 70) -> pd.Series:
    """
    Filter trading signals using RSI
    
    Args:
        signals: Original signals (1=buy, -1=sell, 0=hold)
        rsi: RSI series
        oversold: RSI oversold level (default 30)
        overbought: RSI overbought level (default 70)
        
    Returns:
        Filtered signals
    """
    filtered = signals.copy()
    
    # Don't buy if RSI is overbought
    filtered[(signals == 1) & (rsi > overbought)] = 0
    
    # Don't sell if RSI is oversold
    filtered[(signals == -1) & (rsi < oversold)] = 0
    
    return filtered


def filter_signals_with_macd(signals: pd.Series, macd_hist: pd.Series) -> pd.Series:
    """
    Filter trading signals using MACD histogram
    
    Args:
        signals: Original signals (1=buy, -1=sell, 0=hold)
        macd_hist: MACD histogram series
        
    Returns:
        Filtered signals
    """
    filtered = signals.copy()
    
    # Only buy if MACD histogram is positive (uptrend)
    filtered[(signals == 1) & (macd_hist <= 0)] = 0
    
    # Only sell if MACD histogram is negative (downtrend)
    filtered[(signals == -1) & (macd_hist >= 0)] = 0
    
    return filtered


# ────────────────────────────────────────────────────────────
# Example usage
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=252, freq='D')
    
    # Generate random OHLCV data
    close = 100 + np.cumsum(np.random.randn(252) * 2)
    high = close + np.random.rand(252) * 2
    low = close - np.random.rand(252) * 2
    volume = np.random.randint(1000000, 5000000, 252)
    
    df = pd.DataFrame({
        'time': dates,
        'close': close,
        'high': high,
        'low': low,
        'volume': volume
    })
    
    print("Original DataFrame:")
    print(df.head())
    print()
    
    # Add all indicators
    df_with_indicators = add_all_indicators(df)
    
    print("DataFrame with indicators:")
    print(df_with_indicators.tail())
    print()
    
    # Generate SMA signals
    df_signals = generate_sma_signals(df, short_window=10, long_window=50)
    
    print("Buy signals:")
    print(df_signals[df_signals['signal'] == 1][['time', 'close', 'SMA10', 'SMA50']])
    print()
    
    print("Sell signals:")
    print(df_signals[df_signals['signal'] == -1][['time', 'close', 'SMA10', 'SMA50']])

