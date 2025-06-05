# -*- coding: utf-8 -*-
"""
plot_compare.py
---------------
Plots real (denormalized) Close Price vs MLP and Hybrid predictions
on the test dataset.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import matplotlib.pyplot as plt

# Add the 'src' directory to the import path so we can import train_hybrid
root = Path(__file__).resolve().parents[1]
sys.path.append(str(root / "src"))

def load_test_data():
    """
    Reads the test dataset (scaled).
    Returns:
      X_test_scaled (np.ndarray),
      y_test_scaled (np.ndarray),
      dates_test (np.ndarray) — integer indices for plotting.
    """
    proc = root / "data" / "processed"
    test_df = pd.read_csv(proc / "test.csv")
    X_test = test_df.drop("target", axis=1).values
    y_test = test_df["target"].values.reshape(-1, 1)
    dates = np.arange(len(y_test))
    return X_test, y_test, dates

def denormalize(y_scaled, scaler_y):
    """
    Transforms y_scaled (n×1) back to real price values
    using the MinMaxScaler saved to disk.
    """
    return scaler_y.inverse_transform(y_scaled)

def main():
    model_dir = root / "models"

    # 1) Load test data (scaled)
    X_test, y_test_scaled, dates = load_test_data()

    # 2) Load the scaler for the target values (to denormalize)
    scaler_y = joblib.load(model_dir / "scaler_y.pkl")

    # 3) Make MLP prediction
    mlp = load_model(model_dir / "mlp_baseline.h5", compile=False)
    y_pred_mlp_scaled = mlp.predict(X_test)
    y_pred_mlp = denormalize(y_pred_mlp_scaled, scaler_y)

    # 4) Make Hybrid prediction
    Theta = np.load(model_dir / "hybrid_theta.npy", allow_pickle=True)

    #    4.1. Load gmdh_layers.pkl to reconstruct X_last_test
    gmdh_layers = joblib.load(model_dir / "gmdh_layers.pkl")
    last_layer_info = gmdh_layers[-1]

    #    4.2. Build X_last_test based on last GMDH layer information
    n_test = X_test.shape[0]
    d = len(last_layer_info)
    X_last_test = np.zeros((n_test, d))
    for idx, (i, j, w_list) in enumerate(last_layer_info):
        Xi = X_test[:, i].reshape(-1, 1)
        Xj = X_test[:, j].reshape(-1, 1)
        P = np.hstack([
            np.ones((n_test, 1)),
            Xi,
            Xj,
            Xi * Xi,
            Xi * Xj,
            Xj * Xj
        ])
        w = np.array(w_list).reshape(-1, 1)
        X_last_test[:, idx] = P.dot(w).ravel()

    #    4.3. Build fuzzy membership matrix Z_test
    from train_hybrid import compute_fuzzy_membership
    FUZZY_MF = 3  # Must match FUZZY_MF in train_hybrid.py
    Z_test, _ = compute_fuzzy_membership(X_last_test, FUZZY_MF)

    y_pred_hybrid_scaled = Z_test.dot(Theta)
    y_pred_hybrid = denormalize(y_pred_hybrid_scaled, scaler_y)

    # 5) Denormalize true y_test
    y_test = denormalize(y_test_scaled, scaler_y)

    # 6) Plot first 200 points to compare actual, MLP, and Hybrid predictions
    n_plot = min(200, len(dates))
    plt.figure(figsize=(10, 5))
    plt.plot(dates[:n_plot], y_test[:n_plot],
             label="Real Close Price",
             color="black", linewidth=1.2)
    plt.plot(dates[:n_plot], y_pred_mlp[:n_plot],
             label="MLP Predicted",
             linestyle="--", linewidth=1)
    plt.plot(dates[:n_plot], y_pred_hybrid[:n_plot],
             label="Hybrid Predicted",
             linestyle=":", linewidth=1)
    plt.xlabel("Test Index")
    plt.ylabel("Close Price")
    plt.title(f"Real vs MLP vs Hybrid (first {n_plot} points)")
    plt.legend()
    plt.tight_layout()

    # save the figure:
    fig_dir = root / "figures"
    fig_dir.mkdir(exist_ok=True)
    out_path = fig_dir / "compare_plot.png"
    plt.savefig(out_path)
    print(f"Saved comparison plot to {out_path}")


if __name__ == "__main__":
    main()
