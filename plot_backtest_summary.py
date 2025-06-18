# plot_backtest_summary.py
# ─────────────────────────────────────────────────────────────
"""
Biểu đồ hóa kết quả backtest chiến lược SMA toàn bộ VN30

• Đọc file backtest_summary.csv
• Vẽ:
    1. Biểu đồ thanh: Top 10 mã lợi nhuận cao nhất
    2. Biểu đồ scatter: Tỷ suất sinh lời (%) vs Win rate (%)

Cách dùng:
    python plot_backtest_summary.py

Yêu cầu:
    pip install plotly pandas
"""

import pandas as pd
import plotly.express as px
import os

def vẽ_top_lợi_nhuận(df):
    top = df.sort_values("Tổng lợi nhuận (VND)", ascending=False).head(10)
    fig = px.bar(
        top,
        x="Mã", y="Tổng lợi nhuận (VND)",
        title="Top 10 mã có lợi nhuận cao nhất từ chiến lược SMA Cross",
        text="Tổng lợi nhuận (VND)",
        color="Tổng lợi nhuận (VND)",
        color_continuous_scale="Tealgrn"
    )
    fig.update_layout(
        yaxis_title="Lợi nhuận (VND)",
        xaxis_title="Mã cổ phiếu",
        hovermode="x unified"
    )
    fig.update_traces(texttemplate="%{text:,}", textposition="outside")
    fig.show()

def vẽ_scatter(df):
    fig = px.scatter(
        df,
        x="Tỷ suất (%)",
        y="Win rate (%)",
        color="Mã",
        size="Số lệnh",
        text="Mã",
        title="Tỷ suất sinh lời (%) vs Tỷ lệ thắng (%) toàn bộ VN30",
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    fig.update_traces(textposition="top center")
    fig.update_layout(
        xaxis_title="Tỷ suất sinh lời (%)",
        yaxis_title="Tỷ lệ thắng (%)",
        hovermode="closest"
    )
    fig.show()

def main():
    path = "backtest_summary.csv"
    if not os.path.exists(path):
        print("❌ Không tìm thấy file backtest_summary.csv. Hãy chạy backtest_all_vn30.py trước.")
        return

    df = pd.read_csv(path)
    print("✅ Đã tải dữ liệu tổng hợp.")

    vẽ_top_lợi_nhuận(df)
    vẽ_scatter(df)

if __name__ == "__main__":
    main()
