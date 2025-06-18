# app_dashboard.py
# ─────────────────────────────────────────────────────────────
"""
Dashboard Streamlit SMA Cross cho VN30:
• Vẽ biểu đồ tương tác
• So sánh hiệu suất SMA Cross vs Buy & Hold
• Chạy bằng:  streamlit run app_dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os

SIGNAL_DIR = "data/signals"

def load_data(symbol):
    path = os.path.join(SIGNAL_DIR, f"{symbol}_signals.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, parse_dates=["time"])

def backtest(df, vốn=100_000_000):
    giá_mua = 0
    đang_mua = False
    nav = [vốn]
    lệnh = []
    for _, row in df.iterrows():
        giá = row["close"]
        tín = row["signal"]

        if tín == 1 and not đang_mua:
            giá_mua = giá
            đang_mua = True
        elif tín == -1 and đang_mua:
            lãi = (giá - giá_mua) / giá_mua
            vốn *= (1 + lãi)
            lệnh.append(lãi)
            nav.append(vốn)
            đang_mua = False
    return vốn, nav, lệnh

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

def main():
    st.set_page_config(page_title="SMA Dashboard VN30", layout="wide")
    st.title("📊 Dashboard chiến lược SMA Cross VN30")

    # Sidebar
    symbols = [f.replace("_signals.csv", "") for f in os.listdir(SIGNAL_DIR) if f.endswith(".csv")]
    symbol = st.sidebar.selectbox("Chọn mã cổ phiếu", sorted(symbols))
    vốn = st.sidebar.number_input("Vốn đầu tư ban đầu (VND)", value=100_000_000, step=1_000_000)

    # Load data
    df = load_data(symbol)
    if df is None:
        st.error("Không tìm thấy dữ liệu.")
        return

    # So sánh chiến lược
    cuối_kỳ = df["close"].iloc[-1]
    đầu_kỳ = df["close"].iloc[0]
    vốn_hold = vốn * cuối_kỳ / đầu_kỳ
    vốn_sma, nav, lệnh = backtest(df.copy(), vốn)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("🏦 SMA Cross NAV", f"{vốn_sma:,.0f} VND", f"{(vốn_sma/vốn - 1)*100:.2f}%")
    with col2:
        st.metric("📈 Buy & Hold NAV", f"{vốn_hold:,.0f} VND", f"{(vốn_hold/vốn - 1)*100:.2f}%")

    # Đồ thị
    st.plotly_chart(plot_chart(df, symbol), use_container_width=True)

    # Bảng giao dịch
    if lệnh:
        df_nav = pd.DataFrame({"Lần": list(range(1, len(nav))), "NAV": nav[1:]})
        st.subheader("🔁 Giao dịch và NAV")
        st.line_chart(df_nav.set_index("Lần"))
    else:
        st.info("Không có giao dịch nào trong giai đoạn.")

if __name__ == "__main__":
    main()
