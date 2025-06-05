# -*- coding: utf-8 -*-
"""
train_mlp.py
------------
Trains a simple MLP baseline on the processed CSCO dataset.
Saves model and training history.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import tensorflow as tf
from tf.keras import layers, models, callbacks
import json

def main():

    ROOT = Path(__file__).resolve().parents[1]
    PROC = ROOT / "data" / "processed"
    MODEL_DIR = ROOT / "models"
    MODEL_DIR.mkdir(exist_ok=True)

    train_df = pd.read_csv(PROC / "train.csv")
    test_df  = pd.read_csv(PROC / "test.csv")

    X_train = train_df.drop("target", axis=1).values
    y_train = train_df["target"].values
    X_test  = test_df.drop("target", axis=1).values
    y_test  = test_df["target"].values

    input_dim = X_train.shape[1]

    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='linear'),
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    print(model.summary())

    early_stop = callbacks.EarlyStopping(patience=20, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=200,
        batch_size=32,
        callbacks=[early_stop],
        verbose=2
    )

    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test MAE (scaled): {test_mae:.4f}")

    model.save(MODEL_DIR / "mlp_baseline.h5")
    with open(MODEL_DIR / "mlp_history.json", "w") as f:
        json.dump(history.history, f)

    print("Model and history saved to models/")

if __name__ == "__main__":
    main()
