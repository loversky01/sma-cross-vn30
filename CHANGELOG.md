# Changelog

All notable changes to the SMA Cross VN30 project.

## [1.0.0] - 2024-12-XX

### 🎉 Initial Release

#### ✨ Core Features

- **Data Collection**
  - Automated data collection from vnstock API
  - Support for all VN30 stocks
  - Historical price data storage

- **Trading Strategy**
  - SMA Cross strategy (SMA10 x SMA50)
  - Configurable SMA parameters
  - Buy/Sell signal generation

#### 🛡️ Risk Management

- Stop Loss (configurable %)
- Take Profit (configurable %)
- Trailing Stop Loss
- Position sizing control

#### 📊 Backtesting

- Simple backtest (backtest_sma.py)
- Advanced backtest with risk management (backtest_advanced.py)
- Portfolio backtest for multiple stocks
- Walk-forward analysis
- Monte Carlo simulation

#### 📈 Advanced Metrics

- **Return Metrics**
  - Total Return
  - Annualized Return
  - Volatility

- **Risk-Adjusted Metrics**
  - Sharpe Ratio
  - Sortino Ratio
  - Calmar Ratio
  - Information Ratio

- **Trade Statistics**
  - Win Rate
  - Profit Factor
  - Average Win/Loss Ratio
  - Consecutive Wins/Losses
  - Best/Worst Trade

- **Risk Metrics**
  - Maximum Drawdown
  - Max Drawdown Duration

#### 🎨 Visualization

- Interactive Streamlit dashboard
- Price charts with SMA overlays
- Equity curve
- Drawdown chart
- Trade distribution histogram
- Monthly returns heatmap

#### 🔧 Technical Indicators

- Simple Moving Average (SMA)
- Exponential Moving Average (EMA)
- Relative Strength Index (RSI)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Average True Range (ATR)
- Average Directional Index (ADX)
- Stochastic Oscillator
- On-Balance Volume (OBV)

#### ⚙️ Configuration

- YAML configuration file
- Centralized parameter management
- Easy customization

#### 🔍 Optimization

- Grid Search for parameter optimization
- Multi-metric optimization
- Per-stock optimization

#### 💼 Portfolio Management

- Multi-stock portfolio backtest
- Position limits
- Equal/Custom position sizing
- Risk diversification

#### 🗄️ Data Management

- SQLite/PostgreSQL integration
- Price data storage
- Signal storage
- Backtest results storage
- Trade history storage

#### 🖥️ CLI Interface

- Command-line interface
- Easy-to-use commands
- Batch processing support

#### 📚 Documentation

- Comprehensive README
- API documentation
- Jupyter notebooks for tutorials
- Quick start guide

#### 🧪 Testing

- Unit tests with pytest
- Test coverage for core modules
- Automated testing setup

#### 📝 Logging

- Structured logging system
- Console and file logging
- Configurable log levels

### 🔧 Technical Stack

- Python 3.9+
- pandas, numpy for data processing
- plotly for interactive charts
- streamlit for web dashboard
- pytest for testing
- SQLAlchemy for database
- vnstock for data collection

### 📦 Installation

```bash
pip install -r requirements.txt
```

### 🚀 Usage

```bash
# Update data
python sma_cross_vn30.py

# Backtest single stock
python backtest_advanced.py MSN

# Optimize parameters
python optimize_parameters.py VCB

# Portfolio backtest
python portfolio_backtest.py

# Launch dashboard
streamlit run app_dashboard.py

# CLI
python cli.py backtest MSN
python cli.py optimize FPT
```

### ⚠️ Known Issues

None

### 🔮 Future Plans

- Machine learning integration
- More trading strategies
- Real-time trading support
- Telegram/Email alerts
- Web API
- Mobile app

---

**Made with ❤️ for Vietnamese stock traders**

