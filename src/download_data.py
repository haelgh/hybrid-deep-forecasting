# -*- coding: utf-8 -*-
"""
download_data.py
----------------
Downloads Cisco (CSCO) daily OHLC data from Yahoo Finance
and saves it to data/raw/csco.csv
"""

import yfinance as yf
import pandas as pd
from pathlib import Path

# ⬇⬇⬇  ДОДАЙ ОЦЕ ⬇⬇⬇
def get_data(symbol: str = "CSCO",
             start: str = "2006-01-01",
             end: str   = "2018-12-31") -> pd.DataFrame:
    """
    Завантажує котирування з Yahoo Finance і повертає DataFrame
    з однією колонкою 'Close' (саме так очікує run_experiments.py).
    """
    df = yf.download(symbol, start=start, end=end,
                     group_by="column", auto_adjust=False,
                     progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df[["Close"]].dropna()
# ⬆⬆⬆  ДОДАЙ ОЦЕ  ⬆⬆⬆

def main():                     # ← твоя стара функція
    symbol = "CSCO"
    start  = "2006-01-01"
    end    = "2018-12-31"

    data = get_data(symbol, start, end)   # ← тепер використовуємо
    if data.empty:
        print("No data downloaded.")
        return

    out_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "csco.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(out_path, index_label="Date")
    print(f"Saved {len(data)} rows to {out_path}")

if __name__ == "__main__":
    main()
