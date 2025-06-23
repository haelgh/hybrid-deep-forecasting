# src/run_experiments.py
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf

from .download_data import get_data         
from models.mlp_baseline import build_mlp
from models.hybrid_gmdh_nf import HybridGMDHNeoFuzzy

WINDOW, TEST_SPLIT, EPOCHS = 20, 0.3, 200

def make_supervised(series):
    X, y = [], []
    for i in range(len(series) - WINDOW):
        X.append(series[i:i + WINDOW])
        y.append(series[i + WINDOW])
    return np.array(X), np.array(y)

def evaluate(y_true, y_pred):
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE":  mean_absolute_error(y_true, y_pred),
        "R2":   r2_score(y_true, y_pred),
        "MAPE": np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    }

def main():
    # ────────── дані ──────────
    df = get_data()                            # DataFrame з колонкою 'Close'
    scaler = MinMaxScaler()
    series = scaler.fit_transform(df["Close"].values.reshape(-1, 1)).flatten()

    X, y = make_supervised(series)
    split = int(len(X) * (1 - TEST_SPLIT))
    X_train, X_test = X[:split],  X[split:]
    y_train, y_test = y[:split],  y[split:]

    # ────────── MLP baseline ──────────
    mlp = build_mlp(WINDOW)
    hist = mlp.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=EPOCHS,
        verbose=0,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)]
    )
    y_pred_mlp = mlp.predict(X_test).flatten()

    # ────────── Hybrid GMDH+Neo-Fuzzy ──────────
    hybrid = HybridGMDHNeoFuzzy()
    hybrid.fit(X_train, y_train)
    y_pred_h = hybrid.predict(X_test).reshape(-1, 1)

    # ────────── денормалізуємо та рахуємо метрики ──────────
    y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_mlp  = scaler.inverse_transform(y_pred_mlp.reshape(-1, 1)).flatten()
    y_pred_h    = scaler.inverse_transform(y_pred_h).flatten()

    metrics_mlp = evaluate(y_test_real, y_pred_mlp)
    metrics_h   = evaluate(y_test_real, y_pred_h)

    pd.DataFrame([metrics_h, metrics_mlp],
                 index=["Hybrid", "MLP"]).to_csv("data/metrics.csv")

    # ────────── графіки ──────────
    plt.figure()
    plt.plot(hist.history["loss"], label="train")
    plt.plot(hist.history["val_loss"], label="val")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    plt.savefig("figures/loss_curve_mlp.pdf")

    plt.figure(figsize=(10, 4))
    plt.plot(y_test_real[:200],       label="Real Close Price", color="black")
    plt.plot(y_pred_mlp[:200],        label="MLP Predicted",   linestyle="--")
    plt.plot(y_pred_h[:200],          label="Hybrid Predicted",linestyle=":")
    plt.title("Real vs MLP vs Hybrid (first 200 points)")
    plt.xlabel("Test Index"); plt.ylabel("Close Price")
    plt.legend()
    plt.savefig("figures/compare_first200.pdf")

if __name__ == "__main__":
    main()
