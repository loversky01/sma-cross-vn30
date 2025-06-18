# plot_sma_chart.py
# ──────────────────────────────────────────────────────
"""
TƯƠNG THÍCH PYTHON + R (RStudio) — VẼ ĐỒ THỊ SMA CROSS TIẾNG VIỆT

────────────────────────────────────────────────────────
► PYTHON (VS Code, Jupyter, v.v.)
    python plot_sma_chart.py             # vẽ đồ thị cho MSN
    python plot_sma_chart.py VCB         # truyền mã cổ phiếu

► R / RStudio (dùng reticulate)
    library(reticulate)
    source_python("plot_sma_chart.py")
    plot_sma_vi("FPT")                   # vẽ mã FPT

Lưu ý:
    • Yêu cầu file tín hiệu nằm trong data/signals/<Mã>_signals.csv
      (được tạo bởi script sma_cross_vn30.py).
    • Yêu cầu plotly ≥ 5.0 ; pandas ≥ 1.5
"""

import sys, os
import pandas as pd

# ──────────────────────────────────────────────────────
# 1. Kiểm thử & import plotly
try:
    import plotly.graph_objects as go
except ModuleNotFoundError:
    print("❌ Chưa cài plotly. Vui lòng gõ:  pip install plotly")
    sys.exit(1)

SIGNAL_DIR = "data/signals"      # thư mục chứa *.csv tín hiệu

# ──────────────────────────────────────────────────────
def plot_sma_vi(symbol: str = "MSN") -> None:
    """
    Đọc file <SIGNAL_DIR>/<symbol>_signals.csv và hiển thị đồ thị plotly
    """
    path = os.path.join(SIGNAL_DIR, f"{symbol}_signals.csv")
    if not os.path.exists(path):
        print(f"❌ Không tìm thấy {path}. Hãy chạy sma_cross_vn30.py trước.")
        return

    df = pd.read_csv(path, parse_dates=["time"])
    df.rename(columns={
        "time":  "Ngày",
        "close": "Đóng cửa"
    }, inplace=True)

    # Tạo figure với giá + SMA + điểm Mua/Bán
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["Ngày"], y=df["Đóng cửa"],
        mode="lines", name="Giá đóng cửa", line=dict(color="black")
    ))
    fig.add_trace(go.Scatter(
        x=df["Ngày"], y=df["SMA10"],
        mode="lines", name="SMA10", line=dict(color="blue")
    ))
    fig.add_trace(go.Scatter(
        x=df["Ngày"], y=df["SMA50"],
        mode="lines", name="SMA50", line=dict(color="orange")
    ))

    # Mũi tên Mua / Bán
    buys  = df[df["signal"] ==  1]
    sells = df[df["signal"] == -1]

    fig.add_trace(go.Scatter(
        x=buys["Ngày"], y=buys["Đóng cửa"],
        mode="markers", name="Mua",
        marker=dict(symbol="triangle-up", color="green", size=10)
    ))
    fig.add_trace(go.Scatter(
        x=sells["Ngày"], y=sells["Đóng cửa"],
        mode="markers", name="Bán",
        marker=dict(symbol="triangle-down", color="red", size=10)
    ))

    fig.update_layout(
        title=f"Chiến lược SMA Cross – {symbol}",
        xaxis_title="Ngày",
        yaxis_title="Giá (VND)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom",
                    y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(tickformat=",")   # phân tách hàng nghìn
    fig.show()

# ──────────────────────────────────────────────────────
# 2. Cho phép chạy từ command-line
if __name__ == "__main__":
    # Nếu truyền thêm tham số, dùng làm mã cổ phiếu
    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "MSN"
    plot_sma_vi(ticker)
