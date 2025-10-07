"""
database.py
────────────────────────────────────────────────────────────
Database integration (SQLite/PostgreSQL) for SMA Cross system

Store:
• Price data
• Signals
• Backtest results
• Trades

Usage:
    from utils.database import Database
    
    db = Database()
    db.save_prices(df, "MSN")
    df = db.load_prices("MSN")
"""

import os
import sqlite3
import pandas as pd
from typing import Optional, List, Dict
from datetime import datetime
import sqlalchemy as sa
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from utils.config_loader import get_config
from utils.logger import get_logger

Base = declarative_base()


# ────────────────────────────────────────────────────────────
# Database Models
# ────────────────────────────────────────────────────────────

class PriceData(Base):
    """Price data table"""
    __tablename__ = 'prices'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False, index=True)
    time = Column(DateTime, nullable=False, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float, nullable=False)
    volume = Column(Float)


class SignalData(Base):
    """Signal data table"""
    __tablename__ = 'signals'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False, index=True)
    time = Column(DateTime, nullable=False, index=True)
    close = Column(Float, nullable=False)
    sma_short = Column(Float)
    sma_long = Column(Float)
    signal = Column(Integer)  # 1=buy, -1=sell, 0=hold


class BacktestResult(Base):
    """Backtest results table"""
    __tablename__ = 'backtest_results'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False, index=True)
    run_date = Column(DateTime, nullable=False, default=datetime.now)
    strategy_name = Column(String(50))
    initial_capital = Column(Float)
    final_capital = Column(Float)
    total_return_pct = Column(Float)
    sharpe_ratio = Column(Float)
    max_drawdown_pct = Column(Float)
    win_rate_pct = Column(Float)
    total_trades = Column(Integer)
    config_json = Column(Text)  # Store config as JSON


class Trade(Base):
    """Individual trades table"""
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    backtest_id = Column(Integer, index=True)  # Foreign key to backtest_results
    symbol = Column(String(10), nullable=False, index=True)
    entry_date = Column(DateTime, nullable=False)
    entry_price = Column(Float, nullable=False)
    exit_date = Column(DateTime, nullable=False)
    exit_price = Column(Float, nullable=False)
    shares = Column(Integer)
    profit_vnd = Column(Float)
    profit_pct = Column(Float)
    exit_reason = Column(String(50))


# ────────────────────────────────────────────────────────────
# Database Class
# ────────────────────────────────────────────────────────────

class Database:
    """Database manager"""
    
    def __init__(self, db_path: Optional[str] = None, db_type: str = "sqlite"):
        """
        Initialize database connection
        
        Args:
            db_path: Database path (for SQLite) or connection string
            db_type: Database type ('sqlite' or 'postgresql')
        """
        self.logger = get_logger("database")
        
        # Load config
        cfg = get_config()
        db_settings = cfg.database_settings
        
        self.db_type = db_type or db_settings.get('db_type', 'sqlite')
        
        # Create connection string
        if self.db_type == 'sqlite':
            self.db_path = db_path or db_settings.get('db_path', 'data/sma_cross.db')
            
            # Create directory if needed
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            self.connection_string = f'sqlite:///{self.db_path}'
        
        elif self.db_type == 'postgresql':
            host = db_settings.get('pg_host', 'localhost')
            port = db_settings.get('pg_port', 5432)
            user = db_settings.get('pg_user', 'postgres')
            password = db_settings.get('pg_password', '')
            database = db_settings.get('pg_database', 'sma_cross')
            
            self.connection_string = f'postgresql://{user}:{password}@{host}:{port}/{database}'
        
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")
        
        # Create engine and session
        self.engine = create_engine(self.connection_string, echo=False)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        
        # Create tables
        self._create_tables()
        
        self.logger.info(f"✅ Database initialized: {self.db_type}")
    
    def _create_tables(self):
        """Create all tables if they don't exist"""
        Base.metadata.create_all(self.engine)
        self.logger.debug("Database tables created")
    
    # ────────────────────────────────────────────────────────
    # Price Data Methods
    # ────────────────────────────────────────────────────────
    
    def save_prices(self, df: pd.DataFrame, symbol: str):
        """
        Save price data to database
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Stock symbol
        """
        # Delete existing data for this symbol
        self.session.query(PriceData).filter_by(symbol=symbol).delete()
        
        # Insert new data
        for _, row in df.iterrows():
            price = PriceData(
                symbol=symbol,
                time=row['time'],
                open=row.get('open'),
                high=row.get('high'),
                low=row.get('low'),
                close=row['close'],
                volume=row.get('volume')
            )
            self.session.add(price)
        
        self.session.commit()
        self.logger.info(f"✅ Saved {len(df)} price records for {symbol}")
    
    def load_prices(self, symbol: str, start_date: Optional[datetime] = None,
                   end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Load price data from database
        
        Args:
            symbol: Stock symbol
            start_date: Start date filter (optional)
            end_date: End date filter (optional)
            
        Returns:
            DataFrame with price data
        """
        query = self.session.query(PriceData).filter_by(symbol=symbol)
        
        if start_date:
            query = query.filter(PriceData.time >= start_date)
        
        if end_date:
            query = query.filter(PriceData.time <= end_date)
        
        query = query.order_by(PriceData.time)
        
        results = query.all()
        
        if len(results) == 0:
            return pd.DataFrame()
        
        data = [{
            'time': r.time,
            'open': r.open,
            'high': r.high,
            'low': r.low,
            'close': r.close,
            'volume': r.volume
        } for r in results]
        
        return pd.DataFrame(data)
    
    # ────────────────────────────────────────────────────────
    # Signal Data Methods
    # ────────────────────────────────────────────────────────
    
    def save_signals(self, df: pd.DataFrame, symbol: str):
        """Save signal data"""
        self.session.query(SignalData).filter_by(symbol=symbol).delete()
        
        for _, row in df.iterrows():
            signal = SignalData(
                symbol=symbol,
                time=row['time'],
                close=row['close'],
                sma_short=row.get('SMA10'),
                sma_long=row.get('SMA50'),
                signal=row.get('signal', 0)
            )
            self.session.add(signal)
        
        self.session.commit()
        self.logger.info(f"✅ Saved {len(df)} signal records for {symbol}")
    
    def load_signals(self, symbol: str) -> pd.DataFrame:
        """Load signal data"""
        query = self.session.query(SignalData).filter_by(symbol=symbol).order_by(SignalData.time)
        results = query.all()
        
        if len(results) == 0:
            return pd.DataFrame()
        
        data = [{
            'time': r.time,
            'close': r.close,
            'SMA10': r.sma_short,
            'SMA50': r.sma_long,
            'signal': r.signal
        } for r in results]
        
        return pd.DataFrame(data)
    
    # ────────────────────────────────────────────────────────
    # Backtest Results Methods
    # ────────────────────────────────────────────────────────
    
    def save_backtest_result(self, symbol: str, metrics: Dict, config: Dict = None) -> int:
        """
        Save backtest result
        
        Returns:
            Backtest ID
        """
        import json
        
        result = BacktestResult(
            symbol=symbol,
            run_date=datetime.now(),
            strategy_name="SMA Cross",
            initial_capital=config.get('initial_capital') if config else None,
            final_capital=metrics.get('Final Capital'),
            total_return_pct=metrics.get('Total Return (%)'),
            sharpe_ratio=metrics.get('Sharpe Ratio'),
            max_drawdown_pct=metrics.get('Max Drawdown (%)'),
            win_rate_pct=metrics.get('Win Rate (%)'),
            total_trades=metrics.get('Total Trades'),
            config_json=json.dumps(config) if config else None
        )
        
        self.session.add(result)
        self.session.commit()
        
        self.logger.info(f"✅ Saved backtest result for {symbol} (ID: {result.id})")
        
        return result.id
    
    def save_trades(self, backtest_id: int, trades_df: pd.DataFrame):
        """Save trades for a backtest"""
        for _, row in trades_df.iterrows():
            trade = Trade(
                backtest_id=backtest_id,
                symbol=row.get('Symbol', ''),
                entry_date=row['Entry Date'],
                entry_price=row['Entry Price'],
                exit_date=row['Exit Date'],
                exit_price=row['Exit Price'],
                shares=row.get('Shares'),
                profit_vnd=row['Profit (VND)'],
                profit_pct=row['Profit (%)'],
                exit_reason=row.get('Exit Reason')
            )
            self.session.add(trade)
        
        self.session.commit()
        self.logger.info(f"✅ Saved {len(trades_df)} trades for backtest {backtest_id}")
    
    def load_backtest_results(self, symbol: Optional[str] = None) -> pd.DataFrame:
        """Load backtest results"""
        query = self.session.query(BacktestResult)
        
        if symbol:
            query = query.filter_by(symbol=symbol)
        
        query = query.order_by(BacktestResult.run_date.desc())
        
        results = query.all()
        
        if len(results) == 0:
            return pd.DataFrame()
        
        data = [{
            'ID': r.id,
            'Symbol': r.symbol,
            'Run Date': r.run_date,
            'Total Return (%)': r.total_return_pct,
            'Sharpe Ratio': r.sharpe_ratio,
            'Max DD (%)': r.max_drawdown_pct,
            'Win Rate (%)': r.win_rate_pct,
            'Total Trades': r.total_trades
        } for r in results]
        
        return pd.DataFrame(data)
    
    def close(self):
        """Close database connection"""
        self.session.close()
        self.engine.dispose()
        self.logger.info("Database connection closed")


# ────────────────────────────────────────────────────────────
# Example usage
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Initialize database
    db = Database()
    
    # Example: Load prices from CSV and save to database
    import os
    cfg = get_config()
    
    symbol = "MSN"
    csv_path = os.path.join(cfg.price_dir, f"{symbol}.csv")
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, parse_dates=['time'])
        db.save_prices(df, symbol)
        
        # Load back
        loaded_df = db.load_prices(symbol)
        print(f"Loaded {len(loaded_df)} rows for {symbol}")
        print(loaded_df.head())
    
    # Close
    db.close()

