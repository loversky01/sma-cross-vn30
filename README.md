# 📊 SMA Cross VN30 - Hệ Thống Giao Dịch Định Lượng

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Một hệ thống giao dịch định lượng (quantitative trading system) hoàn chỉnh sử dụng chiến lược **SMA Cross** (Simple Moving Average Crossover) để phân tích, backtest và tối ưu hóa giao dịch trên các cổ phiếu **VN30** của thị trường chứng khoán Việt Nam.

---

## 🎯 Tính Năng Chính

### 📈 **Trading Strategy**
- ✅ Chiến lược SMA Cross (SMA10 x SMA50)
- ✅ Stop Loss & Take Profit tự động
- ✅ Trailing Stop Loss
- ✅ Position Sizing linh hoạt
- ✅ Bộ lọc tín hiệu (Volume, RSI, MACD)

### 📊 **Backtesting**
- ✅ Backtest đơn lẻ hoặc toàn bộ VN30
- ✅ Phí giao dịch thực tế (Commission, Tax, Slippage)
- ✅ Metrics nâng cao (Sharpe Ratio, Sortino, Calmar, Profit Factor)
- ✅ Walk-Forward Analysis
- ✅ Monte Carlo Simulation

### 🔧 **Optimization**
- ✅ Grid Search tối ưu tham số SMA
- ✅ Genetic Algorithm (tùy chọn)
- ✅ Tối ưu riêng cho từng mã cổ phiếu

### 📉 **Technical Indicators**
- ✅ SMA (Simple Moving Average)
- ✅ RSI (Relative Strength Index)
- ✅ MACD (Moving Average Convergence Divergence)
- ✅ Bollinger Bands
- ✅ Volume Analysis

### 💼 **Portfolio Management**
- ✅ Multi-stock portfolio
- ✅ Risk management nâng cao
- ✅ Correlation analysis
- ✅ Sector exposure control

### 📱 **Dashboard & Visualization**
- ✅ Interactive dashboard (Streamlit)
- ✅ Equity curve & drawdown chart
- ✅ Monthly returns heatmap
- ✅ Distribution of returns
- ✅ Trade analysis table

### 🗄️ **Data Management**
- ✅ SQLite/PostgreSQL integration
- ✅ Caching system
- ✅ Real-time data update

---

## 🚀 Quick Start

### **1. Cài Đặt**

```bash
# Clone repository
git clone https://github.com/yourusername/sma-cross-vn30.git
cd sma-cross-vn30

# Tạo virtual environment (khuyến nghị)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate  # Windows

# Cài đặt dependencies
pip install -r requirements.txt
```

### **2. Configuration**

Chỉnh sửa file `config.yaml` để tùy chỉnh tham số:

```yaml
strategy:
  sma_short: 10
  sma_long: 50

risk_management:
  stop_loss_pct: 5.0
  take_profit_pct: 15.0

backtest:
  initial_capital: 100000000
  commission_pct: 0.15
```

### **3. Thu Thập Dữ Liệu**

```bash
# Tải dữ liệu giá và tạo tín hiệu cho toàn bộ VN30
python sma_cross_vn30.py
```

### **4. Backtest**

```bash
# Backtest một mã cụ thể
python backtest_sma.py MSN

# Backtest toàn bộ VN30
python backtest_all_vn30.py

# Advanced backtest với stop loss & metrics đầy đủ
python backtest_advanced.py FPT

# Portfolio backtest
python portfolio_backtest.py
```

### **5. Dashboard**

```bash
# Khởi động dashboard
streamlit run app_dashboard.py
```

Mở trình duyệt: `http://localhost:8501`

### **6. Tối Ưu Hóa**

```bash
# Tối ưu tham số SMA cho một mã
python optimize_parameters.py VCB

# Walk-forward analysis
python walk_forward.py HPG

# Monte Carlo simulation
python monte_carlo.py DGC
```

### **7. Vẽ Biểu Đồ**

```bash
# Vẽ biểu đồ SMA cho một mã
python plot_sma_chart.py FPT

# Vẽ tổng hợp kết quả backtest
python plot_backtest_summary.py
```

---

## 📁 Cấu Trúc Thư Mục

```
sma-cross-vn30/
├── config.yaml                 # Configuration file
├── requirements.txt            # Python dependencies
├── README.md                   # Documentation
│
├── sma_cross_vn30.py          # Data collection & signal generation
├── backtest_sma.py            # Simple backtest
├── backtest_advanced.py       # Advanced backtest (stop loss, metrics)
├── backtest_all_vn30.py       # Backtest all VN30 stocks
├── portfolio_backtest.py      # Multi-stock portfolio backtest
│
├── optimize_parameters.py     # Parameter optimization
├── walk_forward.py            # Walk-forward analysis
├── monte_carlo.py             # Monte Carlo simulation
│
├── app_dashboard.py           # Streamlit dashboard
├── plot_sma_chart.py          # Plot SMA chart
├── plot_backtest_summary.py   # Plot summary charts
│
├── utils/
│   ├── config_loader.py       # Config loader
│   ├── metrics.py             # Backtest metrics
│   ├── indicators.py          # Technical indicators
│   ├── database.py            # Database integration
│   └── logger.py              # Logging system
│
├── data/
│   ├── price/                 # Price data (CSV)
│   ├── signals/               # Signal data (CSV)
│   └── sma_cross.db          # SQLite database (optional)
│
├── logs/                      # Log files
├── cache/                     # Cache directory
├── notebooks/                 # Jupyter notebooks
└── tests/                     # Unit tests
```

---

## 📊 Kết Quả Backtest Mẫu

### **Top 5 Mã Tốt Nhất**

| Mã   | Tổng Lợi Nhuận | Tỷ Suất | Win Rate | Sharpe Ratio |
|------|---------------|---------|----------|--------------|
| DGC  | +422.56%      | 422.56% | 46.67%   | 1.85         |
| LPB  | +306.48%      | 306.48% | 58.33%   | 2.12         |
| VIB  | +258.32%      | 258.32% | 52.94%   | 1.93         |
| FPT  | +208.79%      | 208.79% | 53.85%   | 1.76         |
| SSI  | +181.42%      | 181.42% | 43.75%   | 1.54         |

*Kết quả dựa trên backtest từ 2020-01-01 đến 2025-06-16 với vốn ban đầu 100 triệu VND*

---

## 🧮 Chiến Lược SMA Cross

### **Nguyên Lý**

**SMA Cross** là một trong những chiến lược giao dịch phổ biến nhất:

1. **Tín hiệu MUA** 🟢: Khi đường SMA ngắn (10 ngày) cắt lên trên đường SMA dài (50 ngày)
   - Cho thấy xu hướng tăng đang hình thành
   
2. **Tín hiệu BÁN** 🔴: Khi đường SMA ngắn cắt xuống dưới đường SMA dài
   - Cho thấy xu hướng giảm đang hình thành

### **Risk Management**

- **Stop Loss**: Tự động cắt lỗ khi giá giảm 5% so với giá mua
- **Take Profit**: Tự động chốt lời khi giá tăng 15%
- **Trailing Stop**: Di chuyển stop loss theo giá để bảo vệ lợi nhuận

### **Cải Tiến**

- ✅ Lọc tín hiệu bằng Volume
- ✅ Kết hợp RSI để tránh vùng quá mua/quá bán
- ✅ Xác nhận xu hướng bằng MACD
- ✅ Phí giao dịch thực tế (0.15% commission + 0.10% tax)

---

## 📈 Các Metrics Được Tính Toán

### **Return Metrics**
- Total Return (%)
- Annualized Return (%)
- Cumulative Return

### **Risk Metrics**
- Max Drawdown (%)
- Max Drawdown Duration (days)
- Volatility (Standard Deviation)
- Downside Deviation

### **Risk-Adjusted Returns**
- **Sharpe Ratio**: (Return - Risk-free rate) / Volatility
- **Sortino Ratio**: Return / Downside Deviation
- **Calmar Ratio**: Return / Max Drawdown
- **Information Ratio**

### **Trade Statistics**
- Total Trades
- Win Rate (%)
- Average Win / Average Loss
- Profit Factor (Total Win / Total Loss)
- Consecutive Wins / Losses
- Best Trade / Worst Trade

---

## 🔧 Advanced Features

### **1. Parameter Optimization**

```bash
# Grid search
python optimize_parameters.py MSN --method grid_search

# Genetic algorithm
python optimize_parameters.py MSN --method genetic --generations 100
```

### **2. Walk-Forward Analysis**

Chia dữ liệu thành training và testing để tránh overfitting:

```bash
python walk_forward.py VCB --train-pct 70 --test-pct 30
```

### **3. Monte Carlo Simulation**

Mô phỏng 1000 kịch bản để ước tính rủi ro:

```bash
python monte_carlo.py FPT --runs 1000 --confidence 95
```

### **4. Portfolio Backtest**

Giao dịch đồng thời nhiều mã:

```bash
python portfolio_backtest.py --max-positions 5 --capital 500000000
```

---

## 🎨 Dashboard Features

### **Biểu Đồ**
1. 📈 **Price Chart**: Giá + SMA10 + SMA50 + Buy/Sell signals
2. 💰 **Equity Curve**: NAV theo thời gian
3. 📉 **Drawdown Chart**: Underwater plot
4. 📊 **Distribution**: Histogram của returns
5. 🗓️ **Monthly Heatmap**: Returns theo tháng/năm

### **Metrics Dashboard**
- So sánh SMA Cross vs Buy & Hold
- Sharpe Ratio, Max DD, Win Rate
- Trade statistics table
- Risk metrics

---

## 🧪 Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=. --cov-report=html

# Specific test file
pytest tests/test_backtest.py
```

---

## 📚 Jupyter Notebooks

Xem các tutorial notebooks trong thư mục `notebooks/`:

1. `01_data_collection.ipynb` - Thu thập và xử lý dữ liệu
2. `02_strategy_development.ipynb` - Phát triển chiến lược
3. `03_backtesting.ipynb` - Backtest và đánh giá
4. `04_optimization.ipynb` - Tối ưu hóa tham số
5. `05_portfolio.ipynb` - Quản lý portfolio

---

## 🛠️ Command-Line Interface

```bash
# Show version
python cli.py --version

# Backtest
python cli.py backtest MSN --start 2020-01-01 --capital 100000000

# Optimize
python cli.py optimize VCB --param-range 5-20,30-100

# Dashboard
python cli.py dashboard --port 8501

# Update data
python cli.py update --symbols MSN,FPT,VCB
```

---

## ⚙️ Configuration

### **Strategy Parameters**
```yaml
strategy:
  sma_short: 10        # SMA ngắn (ngày)
  sma_long: 50         # SMA dài (ngày)
  use_rsi: true        # Sử dụng RSI filter
  use_macd: false      # Sử dụng MACD confirmation
```

### **Risk Management**
```yaml
risk_management:
  use_stop_loss: true
  stop_loss_pct: 5.0
  use_take_profit: true
  take_profit_pct: 15.0
  position_size_pct: 100.0
```

### **Backtest Settings**
```yaml
backtest:
  initial_capital: 100000000
  commission_pct: 0.15
  tax_pct: 0.10
  slippage_pct: 0.05
```

---

## 📖 Documentation

### **API Documentation**
Chi tiết về các functions và classes: [docs/api.md](docs/api.md)

### **Strategy Guide**
Hướng dẫn phát triển chiến lược: [docs/strategy_guide.md](docs/strategy_guide.md)

### **Backtest Guide**
Hướng dẫn backtest: [docs/backtest_guide.md](docs/backtest_guide.md)

---

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## ⚠️ Disclaimer

**QUAN TRỌNG**: Dự án này chỉ mục đích **NGHIÊN CỨU VÀ HỌC TẬP**.

- ⚠️ Đây **KHÔNG PHẢI** lời khuyên đầu tư
- ⚠️ Kết quả backtest trong quá khứ **KHÔNG ĐẢM BẢO** lợi nhuận tương lai
- ⚠️ Giao dịch chứng khoán có **RỦI RO** mất vốn
- ⚠️ Hãy tự nghiên cứu và đánh giá trước khi đầu tư
- ⚠️ Tác giả **KHÔNG CHỊU TRÁCH NHIỆM** cho bất kỳ tổn thất nào

**Luôn giao dịch có trách nhiệm!**

---

## 📧 Contact

- Author: **Your Name**
- Email: your.email@example.com
- GitHub: [@yourusername](https://github.com/yourusername)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [vnstock](https://github.com/thinh-vu/vnstock) - Vietnam stock data API
- [Streamlit](https://streamlit.io/) - Dashboard framework
- [Plotly](https://plotly.com/) - Interactive charts
- [pandas-ta](https://github.com/twopirllc/pandas-ta) - Technical analysis library

---

## 📊 Project Stats

![GitHub stars](https://img.shields.io/github/stars/yourusername/sma-cross-vn30?style=social)
![GitHub forks](https://img.shields.io/github/forks/yourusername/sma-cross-vn30?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/yourusername/sma-cross-vn30?style=social)

---

**Made with ❤️ for Vietnamese stock traders**

*Happy Trading! 📈*
