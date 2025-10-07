"""
sma_cross_vn30.py
────────────────────────────────────────────────────────
✓ Tải dữ liệu Daily cho toàn bộ VN30 (2020-01-01 → 2024-12-31)
✓ Tính SMA10 – SMA50
✓ Sinh tín hiệu Buy / Sell (1 / -1)
✓ Lưu file CSV: data/price/ & data/signals/
Yêu cầu:  pip install vnstock==0.2.9.0 pandas numpy
"""

import os
import time
import pandas as pd
from vnstock import stock_historical_data


# ───────────────────────────────────────────────────────
VN30 = [
    "ACB", "BCM", "BID", "CTG", "DGC", "FPT", "GAS", "GVR", "HDB", "HPG",
    "LPB", "MBB", "MSN", "MWG", "PLX", "SAB", "SHB", "SSB", "SSI", "STB",
    "TCB", "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VRE"
]

START_DATE = "2020-01-01"
END_DATE   = "2025-10-08"
SHORT_WIN  = 10      # SMA ngắn
LONG_WIN   = 50      # SMA dài

PRICE_DIR  = "data/price"
SIGNAL_DIR = "data/signals"
os.makedirs(PRICE_DIR,  exist_ok=True)
os.makedirs(SIGNAL_DIR, exist_ok=True)

# ───────────────────────────────────────────────────────
def download_price(symbol: str) -> pd.DataFrame | None:
    """Tải lịch sử giá 1 mã, trả về DataFrame (hoặc None nếu lỗi)"""
    try:
        df = stock_historical_data(symbol, START_DATE, END_DATE, resolution="1D")
        if df.empty:
            print(f"⚠️  {symbol}: không có dữ liệu")
            return None
        df.to_csv(f"{PRICE_DIR}/{symbol}.csv", index=False)
        print(f"✅  {symbol}: lưu {len(df)} dòng giá")
        return df
    except Exception as e:
        print(f"❌  {symbol}: lỗi {e}")
        return None

def add_sma_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Thêm SMA10, SMA50 và tín hiệu Buy/Sell vào DataFrame"""
    out = df.copy()
    out["SMA10"] = out["close"].rolling(SHORT_WIN).mean()
    out["SMA50"] = out["close"].rolling(LONG_WIN).mean()

    out["signal"] = 0
    cross_up   = (out["SMA10"] > out["SMA50"]) & (out["SMA10"].shift(1) <= out["SMA50"].shift(1))
    cross_down = (out["SMA10"] < out["SMA50"]) & (out["SMA10"].shift(1) >= out["SMA50"].shift(1))
    out.loc[cross_up,   "signal"] =  1   # Buy
    out.loc[cross_down, "signal"] = -1   # Sell
    return out

def main():
    for sym in VN30:
        df_price = download_price(sym)
        if df_price is None:
            continue

        df_sig = add_sma_signals(df_price)
        df_sig.to_csv(f"{SIGNAL_DIR}/{sym}_signals.csv", index=False)

        # In 3 phiên gần nhất để kiểm tra
        print(df_sig.tail(3)[["time", "close", "SMA10", "SMA50", "signal"]])
        print("-" * 60)

        time.sleep(1.5)  # tránh spam server vnstock

if __name__ == "__main__":
    main()
