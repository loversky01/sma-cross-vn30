# backtest_all_vn30.py
# ─────────────────────────────────────────────────────────────
"""
Backtest toàn bộ chiến lược SMA Cross cho 30 mã VN30
─────────────────────────────────────────────────────────────
• Đọc file: data/signals/<Mã>_signals.csv
• Mỗi mã:
    - Mô phỏng mua–bán theo tín hiệu
    - Tính lợi nhuận, win rate, drawdown
• Xuất:
    - Bảng tổng hợp ra terminal
    - Lưu file CSV: backtest_summary.csv
"""

import os
import pandas as pd

SIGNAL_DIR = "data/signals"
VN30 = [
    "VCB", "VHM", "VNM", "VIC", "CTG", "BID", "TCB", "VPB", "FPT", "GAS",
    "MWG", "HPG", "SAB", "MSN", "VRE", "PLX", "MBB", "TPB", "STB", "POW",
    "SSI", "SHB", "HDB", "VIB", "PNJ", "BVH", "GVR", "KDH", "SSB", "PDR"
]

def backtest_one(symbol: str, vốn: int = 100_000_000):
    path = os.path.join(SIGNAL_DIR, f"{symbol}_signals.csv")
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path, parse_dates=["time"])
    df = df[["time", "close", "signal"]].dropna()
    trades = []
    đang_mua = False
    giá_mua = 0

    for _, row in df.iterrows():
        ngày, giá, tín_hiệu = row["time"], row["close"], row["signal"]
        if tín_hiệu == 1 and not đang_mua:
            giá_mua = giá
            đang_mua = True
        elif tín_hiệu == -1 and đang_mua:
            lợi_nhuận = (giá - giá_mua) / giá_mua
            trades.append(vốn * lợi_nhuận)
            đang_mua = False

    if not trades:
        return None

    series = pd.Series(trades)
    tổng = series.sum()
    win = (series > 0).sum()
    lose = (series <= 0).sum()
    win_rate = win / (win + lose) * 100 if (win + lose) > 0 else 0
    lợi_suất = tổng / vốn * 100
    max_dd = series.min()

    return {
        "Mã": symbol,
        "Số lệnh": win + lose,
        "Tổng lợi nhuận (VND)": round(tổng),
        "Tỷ suất (%)": round(lợi_suất, 2),
        "Win rate (%)": round(win_rate, 2),
        "Lệnh lỗ nặng nhất (VND)": round(max_dd)
    }

def main():
    kết_quả = []
    for mã in VN30:
        kq = backtest_one(mã)
        if kq:
            kết_quả.append(kq)
        else:
            print(f"⚠️ Không có dữ liệu hoặc không có giao dịch: {mã}")

    if not kết_quả:
        print("❌ Không có kết quả backtest nào.")
        return

    df = pd.DataFrame(kết_quả)
    df.sort_values("Tổng lợi nhuận (VND)", ascending=False, inplace=True)

    # Xuất CSV
    df.to_csv("backtest_summary.csv", index=False)
    print("\n📊 BẢNG TỔNG HỢP BACKTEST TOÀN BỘ VN30")
    print(df.to_string(index=False))

    print("\n✅ Đã lưu file: backtest_summary.csv")

if __name__ == "__main__":
    main()
