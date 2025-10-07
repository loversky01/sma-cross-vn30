"""
test_config.py
────────────────────────────────────────────────────────────
Unit tests for configuration loader

Run:
    pytest tests/test_config.py
"""

import pytest
import os
from utils.config_loader import Config


class TestConfig:
    """Test configuration loader"""
    
    def test_config_loads(self):
        """Test that config file loads"""
        config = Config('config.yaml')
        
        assert config is not None
        assert config.config is not None
    
    def test_get_simple_key(self):
        """Test getting simple key"""
        config = Config('config.yaml')
        
        initial_capital = config.get('backtest.initial_capital')
        assert initial_capital is not None
        assert isinstance(initial_capital, (int, float))
    
    def test_get_nested_key(self):
        """Test getting nested key"""
        config = Config('config.yaml')
        
        sma_short = config.get('strategy.sma_short')
        assert sma_short is not None
    
    def test_get_with_default(self):
        """Test getting with default value"""
        config = Config('config.yaml')
        
        value = config.get('nonexistent.key', default=42)
        assert value == 42
    
    def test_properties(self):
        """Test config properties"""
        config = Config('config.yaml')
        
        assert config.sma_short is not None
        assert config.sma_long is not None
        assert config.initial_capital is not None
        assert isinstance(config.vn30_symbols, list)
        assert len(config.vn30_symbols) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

