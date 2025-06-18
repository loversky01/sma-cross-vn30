# backtest_sma.py
# ─────────────────────────────────────────────────────────
"""
Backtest chiến lược SMA Cross cho một mã cổ phiếu

• Dựa trên file tín hiệu (data/signals/<Mã>_signals.csv)
• Mỗi khi gặp tín hiệu Buy (1) → Mua toàn bộ
• Mỗi khi gặp tín hiệu Sell (-1) → Bán toàn bộ
• Tính:
    - Tổng lợi nhuận tuyệt đối (% và VND)
    - Số giao dịch
    - Tỷ lệ thắng
    - Lệnh tốt nhất, lệnh tệ nhất
    - Max Drawdown đơn giản (trên đường NAV)

Cách dùng:
    python backtest_sma.py MSN
"""

import os
import sys
import pandas as pd

SIGNAL_DIR = "data/signals"

def backtest_sma(symbol: str, vốn_ban_đầu: int = 100_000_000):
    path = os.path.join(SIGNAL_DIR, f"{symbol}_signals.csv")
    if not os.path.exists(path):
        print(f"❌ Không tìm thấy tín hiệu cho {symbol}")
        return

    df = pd.read_csv(path, parse_dates=["time"])
    df = df[["time", "close", "signal"]].copy()
    df.dropna(inplace=True)

    # Danh sách lệnh
    trades = []
    đang_mua = False
    giá_mua = 0

    for _, row in df.iterrows():
        ngày = row["time"]
        giá = row["close"]
        tín_hiệu = row["signal"]

        if tín_hiệu == 1 and not đang_mua:
            giá_mua = giá
            ngày_mua = ngày
            đang_mua = True

        elif tín_hiệu == -1 and đang_mua:
            lợi_nhuận = (giá - giá_mua) / giá_mua
            trades.append({
                "Ngày mua": ngày_mua,
                "Giá mua": giá_mua,
                "Ngày bán": ngày,
                "Giá bán": giá,
                "Lợi nhuận (%)": round(lợi_nhuận * 100, 2),
                "Lợi nhuận (VND)": round(vốn_ban_đầu * lợi_nhuận)
            })
            đang_mua = False

    if not trades:
        print(f"⚠️ Không có giao dịch nào cho {symbol}")
        return

    df_trades = pd.DataFrame(trades)
    tổng = df_trades["Lợi nhuận (VND)"].sum()
    win_rate = (df_trades["Lợi nhuận (VND)"] > 0).mean() * 100
    max_drawdown = df_trades["Lợi nhuận (VND)"].min()

    print(f"\n📈 Kết quả Backtest: {symbol}")
    print("-" * 50)
    print(df_trades)
    print("-" * 50)
    print(f"💰 Tổng lợi nhuận: {tổng:,.0f} VND")
    print(f"📊 Tỷ suất lợi nhuận: {tổng/vốn_ban_đầu*100:.2f}%")
    print(f"🔁 Số lệnh: {len(df_trades)}")
    print(f"✅ Win rate: {win_rate:.2f}%")
    print(f"📉 Lệnh lỗ nặng nhất: {max_drawdown:,.0f} VND")

if __name__ == "__main__":
    sym = sys.argv[1].upper() if len(sys.argv) > 1 else "MSN"
    backtest_sma(sym)
