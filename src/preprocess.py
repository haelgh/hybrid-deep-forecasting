# -*- coding: utf-8 -*-
"""
preprocess.py
-------------
1. Read data/raw/csco.csv
2. Build lag features (t-1 ... t-N)
3. Split into train / test (70 / 30 chronological)
4. Min-Max scale features (X) and target (y) separately
5. Save:
   - data/processed/train.csv
   - data/processed/test.csv
   - models/scaler_X.pkl, scaler_y.pkl
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
from pathlib import Path

# ---------- parameters ----------
LAGS = 5
TEST_RATIO = 0.30
# --------------------------------

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "csco.csv"
PROC_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
PROC_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def build_lags(series: pd.Series, n_lags: int) -> pd.DataFrame:
    df = pd.concat([series.shift(i) for i in range(1, n_lags + 1)], axis=1)
    df.columns = [f"lag_{i}" for i in range(1, n_lags + 1)]
    df["target"] = series.values
    df.dropna(inplace=True)
    return df

def main() -> None:
    df_raw = pd.read_csv(RAW, index_col="Date", parse_dates=True)
    closes = df_raw["Close"]
    df_lagged = build_lags(closes, LAGS)

    split = int(len(df_lagged) * (1 - TEST_RATIO))
    train_df = df_lagged.iloc[:split]
    test_df = df_lagged.iloc[split:]

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_train = scaler_X.fit_transform(train_df.drop("target", axis=1))
    y_train = scaler_y.fit_transform(train_df[["target"]])
    X_test = scaler_X.transform(test_df.drop("target", axis=1))
    y_test = scaler_y.transform(test_df[["target"]])

    pd.DataFrame(
        np.hstack([X_train, y_train]),
        columns=[*train_df.columns[:-1], "target"]
    ).to_csv(PROC_DIR / "train.csv", index=False)

    pd.DataFrame(
        np.hstack([X_test, y_test]),
        columns=[*test_df.columns[:-1], "target"]
    ).to_csv(PROC_DIR / "test.csv", index=False)

    joblib.dump(scaler_X, MODEL_DIR / "scaler_X.pkl")
    joblib.dump(scaler_y, MODEL_DIR / "scaler_y.pkl")

    print(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")

if __name__ == "__main__":
    main()
