# app_dashboard.py
# ─────────────────────────────────────────────────────────────
"""
Dashboard Streamlit SMA Cross cho VN30:
• Vẽ biểu đồ tương tác
• So sánh hiệu suất SMA Cross vs Buy & Hold
• Equity curve và drawdown chart
• Advanced metrics (Sharpe, Sortino, Calmar)
• Chạy bằng:  streamlit run app_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys

from utils.config_loader import get_config
from utils.metrics import calculate_all_metrics
from backtest_advanced import AdvancedBacktester

SIGNAL_DIR = "data/signals"

def load_data(symbol):
    path = os.path.join(SIGNAL_DIR, f"{symbol}_signals.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, parse_dates=["time"])

@st.cache_data
def run_advanced_backtest(symbol, initial_capital, use_stop_loss, stop_loss_pct, 
                         use_take_profit, take_profit_pct):
    """Run advanced backtest with caching"""
    signal_path = os.path.join(SIGNAL_DIR, f"{symbol}_signals.csv")
    
    if not os.path.exists(signal_path):
        return None
    
    df = pd.read_csv(signal_path, parse_dates=['time'])
    df = df[['time', 'close', 'signal']].dropna()
    
    # Create backtester
    backtester = AdvancedBacktester(
        initial_capital=initial_capital,
        commission_pct=0.15,
        tax_pct=0.10,
        slippage_pct=0.05,
        stop_loss_pct=stop_loss_pct if use_stop_loss else None,
        take_profit_pct=take_profit_pct if use_take_profit else None,
        position_size_pct=100.0
    )
    
    results = backtester.run(df, symbol)
    return results

def plot_chart(df, symbol):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["time"], y=df["close"], name="Giá đóng cửa", line=dict(color="black")))
    fig.add_trace(go.Scatter(x=df["time"], y=df["SMA10"], name="SMA10", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=df["time"], y=df["SMA50"], name="SMA50", line=dict(color="orange")))

    buy = df[df["signal"] == 1]
    sell = df[df["signal"] == -1]

    fig.add_trace(go.Scatter(x=buy["time"], y=buy["close"], name="Mua", mode="markers",
                             marker=dict(symbol="triangle-up", color="green", size=10)))
    fig.add_trace(go.Scatter(x=sell["time"], y=sell["close"], name="Bán", mode="markers",
                             marker=dict(symbol="triangle-down", color="red", size=10)))

    fig.update_layout(title=f"SMA Cross chiến lược – {symbol}",
                      xaxis_title="Ngày", yaxis_title="Giá (VND)",
                      hovermode="x unified")
    return fig

def plot_drawdown(equity_curve):
    """Plot drawdown chart"""
    equity_series = pd.Series(equity_curve)
    running_max = equity_series.expanding().max()
    drawdown = (equity_series - running_max) / running_max * 100
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=drawdown,
        fill='tozeroy',
        fillcolor='rgba(255,0,0,0.2)',
        line=dict(color='red'),
        name='Drawdown'
    ))
    
    fig.update_layout(
        title="Drawdown Chart",
        xaxis_title="Time",
        yaxis_title="Drawdown (%)",
        height=300
    )
    
    return fig


def main():
    st.set_page_config(page_title="SMA Dashboard VN30", layout="wide")
    
    # Header
    st.title("📊 Dashboard chiến lược SMA Cross VN30")
    st.markdown("*Advanced backtesting với stop loss, take profit, và metrics đầy đủ*")
    st.divider()
    
    # Sidebar
    st.sidebar.header("⚙️ Cài đặt")
    
    symbols = [f.replace("_signals.csv", "") for f in os.listdir(SIGNAL_DIR) if f.endswith(".csv")]
    symbol = st.sidebar.selectbox("Chọn mã cổ phiếu", sorted(symbols))
    
    st.sidebar.subheader("💰 Vốn")
    initial_capital = st.sidebar.number_input(
        "Vốn ban đầu (VND)", 
        value=100_000_000, 
        step=10_000_000,
        format="%d"
    )
    
    st.sidebar.subheader("🛡️ Risk Management")
    use_stop_loss = st.sidebar.checkbox("Sử dụng Stop Loss", value=True)
    stop_loss_pct = st.sidebar.slider("Stop Loss (%)", 1.0, 10.0, 5.0, 0.5) if use_stop_loss else None
    
    use_take_profit = st.sidebar.checkbox("Sử dụng Take Profit", value=True)
    take_profit_pct = st.sidebar.slider("Take Profit (%)", 5.0, 30.0, 15.0, 1.0) if use_take_profit else None
    
    # Load data
    df = load_data(symbol)
    if df is None:
        st.error("❌ Không tìm thấy dữ liệu.")
        return
    
    # Run backtest
    with st.spinner("⏳ Đang chạy backtest..."):
        results = run_advanced_backtest(
            symbol, initial_capital, 
            use_stop_loss, stop_loss_pct or 5.0,
            use_take_profit, take_profit_pct or 15.0
        )
    
    if not results or 'metrics' not in results:
        st.error("❌ Backtest thất bại")
        return
    
    metrics = results['metrics']
    trades_df = results['trades']
    equity_curve = results['equity_curve']
    
    # Calculate Buy & Hold
    last_price = df["close"].iloc[-1]
    first_price = df["close"].iloc[0]
    buy_hold_return = ((last_price / first_price) - 1) * 100
    buy_hold_final = initial_capital * (1 + buy_hold_return/100)
    
    # === METRICS SECTION ===
    st.header("📈 Kết quả tổng quan")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "💰 Vốn cuối kỳ",
            f"{metrics['Final Capital']:,.0f} VND",
            f"{metrics['Total Return (%)']:.2f}%"
        )
    
    with col2:
        st.metric(
            "📊 Sharpe Ratio",
            f"{metrics['Sharpe Ratio']:.3f}",
            "Cao" if metrics['Sharpe Ratio'] > 1 else "Thấp"
        )
    
    with col3:
        st.metric(
            "✅ Win Rate",
            f"{metrics['Win Rate (%)']:.1f}%",
            f"{metrics['Total Trades']} lệnh"
        )
    
    with col4:
        st.metric(
            "📉 Max Drawdown",
            f"{metrics['Max Drawdown (%)']:.2f}%",
            f"{metrics['Max DD Duration (periods)']} ngày"
        )
    
    # Comparison with Buy & Hold
    st.subheader("⚖️ So sánh với Buy & Hold")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"""
        **🏦 SMA Cross Strategy**
        - Vốn cuối: {metrics['Final Capital']:,.0f} VND
        - Lợi nhuận: {metrics['Total Return (%)']:.2f}%
        - Sharpe Ratio: {metrics['Sharpe Ratio']:.3f}
        """)
    
    with col2:
        st.info(f"""
        **📈 Buy & Hold**
        - Vốn cuối: {buy_hold_final:,.0f} VND
        - Lợi nhuận: {buy_hold_return:.2f}%
        - Sharpe Ratio: N/A
        """)
    
    # Winner
    if metrics['Total Return (%)'] > buy_hold_return:
        st.success(f"🏆 SMA Cross vượt trội hơn Buy & Hold {metrics['Total Return (%)'] - buy_hold_return:.2f}%")
    else:
        st.warning(f"⚠️ Buy & Hold tốt hơn SMA Cross {buy_hold_return - metrics['Total Return (%)']:.2f}%")
    
    st.divider()
    
    # === CHARTS SECTION ===
    st.header("📊 Biểu đồ")
    
    # Price + Signals Chart
    st.subheader("💹 Giá và Tín hiệu")
    st.plotly_chart(plot_chart(df, symbol), use_container_width=True)
    
    # Equity Curve
    st.subheader("💰 Equity Curve")
    fig_equity = go.Figure()
    
    fig_equity.add_trace(go.Scatter(
        y=equity_curve,
        mode='lines',
        name='Portfolio Value',
        line=dict(color='blue', width=2),
        fill='tozeroy',
        fillcolor='rgba(0,100,255,0.1)'
    ))
    
    fig_equity.add_hline(
        y=initial_capital, 
        line_dash="dash", 
        line_color="gray",
        annotation_text="Initial Capital"
    )
    
    fig_equity.update_layout(
        xaxis_title="Time",
        yaxis_title="Portfolio Value (VND)",
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_equity, use_container_width=True)
    
    # Drawdown Chart
    st.subheader("📉 Drawdown")
    st.plotly_chart(plot_drawdown(equity_curve), use_container_width=True)
    
    st.divider()
    
    # === DETAILED METRICS ===
    st.header("📋 Metrics chi tiết")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📊 Return Metrics**")
        st.write(f"Total Return: {metrics['Total Return (%)']:.2f}%")
        st.write(f"Annualized Return: {metrics['Annualized Return (%)']:.2f}%")
        st.write(f"Volatility: {metrics['Volatility (%)']:.2f}%")
    
    with col2:
        st.markdown("**🎯 Risk-Adjusted Metrics**")
        st.write(f"Sharpe Ratio: {metrics['Sharpe Ratio']:.3f}")
        st.write(f"Sortino Ratio: {metrics['Sortino Ratio']:.3f}")
        st.write(f"Calmar Ratio: {metrics['Calmar Ratio']:.3f}")
    
    with col3:
        st.markdown("**📈 Trade Metrics**")
        st.write(f"Total Trades: {metrics['Total Trades']}")
        st.write(f"Win Rate: {metrics['Win Rate (%)']:.2f}%")
        st.write(f"Profit Factor: {metrics['Profit Factor']:.3f}")
        st.write(f"Avg Win/Loss: {metrics['Avg Win/Loss Ratio']:.3f}")
    
    st.divider()
    
    # === TRADES TABLE ===
    st.header("📋 Lịch sử giao dịch")
    
    if len(trades_df) > 0:
        # Summary stats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Best Trade", f"{metrics['Best Trade (%)']:.2f}%")
        with col2:
            st.metric("Worst Trade", f"{metrics['Worst Trade (%)']:.2f}%")
        with col3:
            st.metric("Average Trade", f"{metrics['Average Trade (%)']:.2f}%")
        with col4:
            consec_wins = metrics.get('Max Consecutive Wins', 0)
            st.metric("Max Consecutive Wins", consec_wins)
        
        # Trades table
        st.dataframe(
            trades_df.style.format({
                'Entry Price': '{:,.0f}',
                'Exit Price': '{:,.0f}',
                'Shares': '{:,}',
                'Profit (VND)': '{:,.0f}',
                'Profit (%)': '{:.2f}',
            }),
            use_container_width=True,
            height=400
        )
        
        # Distribution histogram
        st.subheader("📊 Phân phối lợi nhuận")
        
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=trades_df['Profit (%)'],
            nbinsx=20,
            marker_color='lightblue',
            marker_line_color='black',
            marker_line_width=1
        ))
        
        mean_profit = trades_df['Profit (%)'].mean()
        fig_hist.add_vline(
            x=mean_profit,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {mean_profit:.2f}%"
        )
        
        fig_hist.update_layout(
            xaxis_title="Profit (%)",
            yaxis_title="Frequency",
            height=400
        )
        
        st.plotly_chart(fig_hist, use_container_width=True)
    
    else:
        st.info("Không có giao dịch nào trong giai đoạn.")
    
    # Footer
    st.divider()
    st.caption("📊 SMA Cross VN30 Dashboard | Made with ❤️ using Streamlit")

if __name__ == "__main__":
    main()
