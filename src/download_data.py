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

def main():
    symbol = "CSCO"
    start = "2006-01-01"
    end   = "2018-12-31"

    data = yf.download(
        tickers=symbol,
        start=start,
        end=end,
        group_by="column",
        auto_adjust=False,
        progress=False
    )

    if data.empty:
        print("No data downloaded.")
        return

    # Flatten MultiIndex: keep only first level (fields Open, High, Low, Close, etc.)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    out_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "csco.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(out_path, index_label="Date")
    print(f"Saved {len(data)} rows to {out_path}")

if __name__ == "__main__":
    main()
