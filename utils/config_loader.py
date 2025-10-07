"""
config_loader.py
────────────────────────────────────────────────────────────
Configuration loader for SMA Cross VN30 system
Load and validate YAML configuration
"""

import os
import yaml
from typing import Any, Dict
from pathlib import Path


class Config:
    """Configuration management class"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize configuration
        
        Args:
            config_path: Path to YAML config file
        """
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key (supports nested keys with dot notation)
        
        Args:
            key: Configuration key (e.g., 'strategy.sma_short')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def reload(self):
        """Reload configuration from file"""
        self.config = self._load_config()
    
    # ────────────────────────────────────────────────────────
    # Convenience properties for common settings
    # ────────────────────────────────────────────────────────
    
    @property
    def data_settings(self) -> Dict[str, Any]:
        """Get data settings"""
        return self.config.get('data', {})
    
    @property
    def strategy_settings(self) -> Dict[str, Any]:
        """Get strategy settings"""
        return self.config.get('strategy', {})
    
    @property
    def risk_settings(self) -> Dict[str, Any]:
        """Get risk management settings"""
        return self.config.get('risk_management', {})
    
    @property
    def backtest_settings(self) -> Dict[str, Any]:
        """Get backtest settings"""
        return self.config.get('backtest', {})
    
    @property
    def optimization_settings(self) -> Dict[str, Any]:
        """Get optimization settings"""
        return self.config.get('optimization', {})
    
    @property
    def dashboard_settings(self) -> Dict[str, Any]:
        """Get dashboard settings"""
        return self.config.get('dashboard', {})
    
    @property
    def logging_settings(self) -> Dict[str, Any]:
        """Get logging settings"""
        return self.config.get('logging', {})
    
    @property
    def database_settings(self) -> Dict[str, Any]:
        """Get database settings"""
        return self.config.get('database', {})
    
    @property
    def performance_settings(self) -> Dict[str, Any]:
        """Get performance settings"""
        return self.config.get('performance', {})
    
    # ────────────────────────────────────────────────────────
    # Specific getters for frequently used values
    # ────────────────────────────────────────────────────────
    
    @property
    def vn30_symbols(self) -> list:
        """Get VN30 symbols list"""
        return self.get('data.vn30_symbols', [])
    
    @property
    def start_date(self) -> str:
        """Get start date"""
        return self.get('data.start_date', '2020-01-01')
    
    @property
    def end_date(self) -> str:
        """Get end date"""
        return self.get('data.end_date', '2025-06-16')
    
    @property
    def sma_short(self) -> int:
        """Get short SMA period"""
        return self.get('strategy.sma_short', 10)
    
    @property
    def sma_long(self) -> int:
        """Get long SMA period"""
        return self.get('strategy.sma_long', 50)
    
    @property
    def stop_loss_pct(self) -> float:
        """Get stop loss percentage"""
        return self.get('risk_management.stop_loss_pct', 5.0)
    
    @property
    def take_profit_pct(self) -> float:
        """Get take profit percentage"""
        return self.get('risk_management.take_profit_pct', 15.0)
    
    @property
    def use_stop_loss(self) -> bool:
        """Check if stop loss is enabled"""
        return self.get('risk_management.use_stop_loss', False)
    
    @property
    def use_take_profit(self) -> bool:
        """Check if take profit is enabled"""
        return self.get('risk_management.use_take_profit', False)
    
    @property
    def initial_capital(self) -> int:
        """Get initial capital"""
        return self.get('backtest.initial_capital', 100_000_000)
    
    @property
    def commission_pct(self) -> float:
        """Get commission percentage"""
        return self.get('backtest.commission_pct', 0.15)
    
    @property
    def tax_pct(self) -> float:
        """Get tax percentage"""
        return self.get('backtest.tax_pct', 0.10)
    
    @property
    def slippage_pct(self) -> float:
        """Get slippage percentage"""
        return self.get('backtest.slippage_pct', 0.05)
    
    @property
    def price_dir(self) -> str:
        """Get price data directory"""
        return self.get('data.price_dir', 'data/price')
    
    @property
    def signal_dir(self) -> str:
        """Get signal data directory"""
        return self.get('data.signal_dir', 'data/signals')
    
    def __repr__(self) -> str:
        return f"Config(config_path='{self.config_path}')"


# Global config instance
_config_instance = None


def get_config(config_path: str = "config.yaml") -> Config:
    """
    Get global configuration instance (Singleton pattern)
    
    Args:
        config_path: Path to config file
        
    Returns:
        Config instance
    """
    global _config_instance
    
    if _config_instance is None:
        _config_instance = Config(config_path)
    
    return _config_instance


def reload_config():
    """Reload global configuration"""
    global _config_instance
    
    if _config_instance is not None:
        _config_instance.reload()


# ────────────────────────────────────────────────────────────
# Example usage
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Load config
    config = get_config()
    
    print(f"SMA Short: {config.sma_short}")
    print(f"SMA Long: {config.sma_long}")
    print(f"Initial Capital: {config.initial_capital:,} VND")
    print(f"Stop Loss: {config.stop_loss_pct}%")
    print(f"Commission: {config.commission_pct}%")
    print(f"VN30 Symbols: {len(config.vn30_symbols)} stocks")
    print(f"\nAll strategy settings:")
    print(config.strategy_settings)

