"""
logger.py
────────────────────────────────────────────────────────────
Logging system for SMA Cross VN30
Replace print statements with proper logging
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output"""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        # Add color to levelname
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
        
        return super().format(record)


def setup_logger(
    name: str = "sma_cross",
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_to_console: bool = True,
    log_format: Optional[str] = None
) -> logging.Logger:
    """
    Setup logger with file and console handlers
    
    Args:
        name: Logger name
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        log_to_console: Whether to log to console
        log_format: Custom log format
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Default format
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Console handler with colors
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        
        console_formatter = ColoredFormatter(
            log_format,
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File handler (no colors)
    if log_file:
        # Create logs directory if needed
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(getattr(logging, log_level.upper()))
        
        file_formatter = logging.Formatter(
            log_format,
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "sma_cross") -> logging.Logger:
    """
    Get existing logger or create new one with default settings
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    # If logger has no handlers, set it up with defaults
    if not logger.handlers:
        # Try to load from config
        try:
            from utils.config_loader import get_config
            config = get_config()
            
            log_settings = config.logging_settings
            log_level = log_settings.get('level', 'INFO')
            log_file = log_settings.get('log_file', 'logs/sma_cross.log')
            log_to_console = log_settings.get('log_to_console', True)
            log_format = log_settings.get('log_format')
            
            # Only create log file if log_to_file is True
            if not log_settings.get('log_to_file', True):
                log_file = None
            
            logger = setup_logger(
                name=name,
                log_level=log_level,
                log_file=log_file,
                log_to_console=log_to_console,
                log_format=log_format
            )
        except Exception:
            # Fallback to default settings
            logger = setup_logger(
                name=name,
                log_level="INFO",
                log_file="logs/sma_cross.log",
                log_to_console=True
            )
    
    return logger


# ────────────────────────────────────────────────────────────
# Convenience functions
# ────────────────────────────────────────────────────────────

def log_trade(logger: logging.Logger, symbol: str, action: str, price: float, 
              date: str, reason: str = ""):
    """Log a trade action"""
    emoji = "🟢" if action.upper() == "BUY" else "🔴"
    logger.info(f"{emoji} {action.upper()} {symbol} @ {price:,.0f} VND on {date} {reason}")


def log_backtest_result(logger: logging.Logger, symbol: str, metrics: dict):
    """Log backtest results"""
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 BACKTEST RESULT: {symbol}")
    logger.info(f"{'='*60}")
    
    for key, value in metrics.items():
        if isinstance(value, float):
            if 'pct' in key.lower() or 'rate' in key.lower() or 'ratio' in key.lower():
                logger.info(f"  {key}: {value:.2f}%")
            else:
                logger.info(f"  {key}: {value:.4f}")
        elif isinstance(value, int):
            logger.info(f"  {key}: {value:,}")
        else:
            logger.info(f"  {key}: {value}")
    
    logger.info(f"{'='*60}\n")


def log_progress(logger: logging.Logger, current: int, total: int, 
                prefix: str = "Progress"):
    """Log progress"""
    pct = (current / total) * 100 if total > 0 else 0
    logger.info(f"{prefix}: {current}/{total} ({pct:.1f}%)")


# ────────────────────────────────────────────────────────────
# Example usage
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Setup logger
    logger = setup_logger(
        name="test_logger",
        log_level="DEBUG",
        log_file="logs/test.log",
        log_to_console=True
    )
    
    # Test different log levels
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
    
    # Test convenience functions
    log_trade(logger, "MSN", "BUY", 85000, "2024-01-15", "(SMA Cross Up)")
    log_trade(logger, "MSN", "SELL", 95000, "2024-02-20", "(Take Profit)")
    
    metrics = {
        "Total Return": 123.45,
        "Win Rate": 65.5,
        "Sharpe Ratio": 1.85,
        "Max Drawdown": -15.2,
        "Total Trades": 25
    }
    log_backtest_result(logger, "VCB", metrics)
    
    # Test progress
    for i in range(1, 11):
        log_progress(logger, i, 10, "Backtesting")

