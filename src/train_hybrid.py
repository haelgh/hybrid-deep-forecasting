"""
train_hybrid.py
---------------
Trains a hybrid GMDH + Neo-Fuzzy Neural Network on the processed CSCO dataset.
Saves model parameters, history, and prints final metrics.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import json

from sklearn.metrics import mean_absolute_error, mean_squared_error

# ------------------- PARAMETERS -----------------
LAGS = 5 
FUZZY_MF = 3 
TEST_RATIO = 0.30  
# ------------------------------------------------

def load_data():
    """
    Loads train.csv та test.csv (scaled).
    Returns:
       X_train_scaled (np.ndarray), y_train_scaled (np.ndarray),
       X_test_scaled (np.ndarray),  y_test_scaled (np.ndarray).
    """
    root = Path(__file__).resolve().parents[1]
    proc = root / "data" / "processed"
    train_df = pd.read_csv(proc / "train.csv")
    test_df  = pd.read_csv(proc / "test.csv")

    X_train = train_df.drop("target", axis=1).values
    y_train = train_df["target"].values.reshape(-1, 1)
    X_test  = test_df.drop("target", axis=1).values
    y_test  = test_df["target"].values.reshape(-1, 1)

    return X_train, y_train, X_test, y_test

# ------- GMDH-block -------
def fit_gmdh(X: np.ndarray, y: np.ndarray, 
             max_layers: int = 10, 
             criterion: str = "mse"):
    """
    A simple implementation of GMDH: 
    Builds polynomial nodes (two features each) layer by layer,
    select the best pairs by MSE until we reach max_layers or MSE grows.

    Returns:
       best_model (list of selected neuronal coefficients for each layer),
       gmdh_history (list of train-mse by layers).A simple implementation of GMDH: 

    """
    # Debug print
    print("🔹 Fit GMDH: start")
    n_samples, n_features = X.shape
    current_X = X.copy() # first - input signs (lags)
    layers_info = []
    history = []

    for layer in range(max_layers):
        pairs = []
        mses = []
        # All pairs of features (i, j), i < j
        for i in range(current_X.shape[1]):
            for j in range(i+1, current_X.shape[1]):
                Xi = current_X[:, i].reshape(-1, 1)
                Xj = current_X[:, j].reshape(-1, 1)
                # Create the matrix [1, Xi, Xj, Xi^2, Xi*Xj, Xj^2]
                P = np.hstack([
                    np.ones((n_samples, 1)), # const
                    Xi,                               
                    Xj,                              
                    Xi * Xi,                          
                    Xi * Xj,                          
                    Xj * Xj                          
                ])  # size (n_samples × 6)

                # Small square (only solve P * w is about y)
                w, residuals, *_ = np.linalg.lstsq(P, y, rcond=None)
                y_pred = P.dot(w)
                mse = np.mean((y - y_pred) ** 2)
                pairs.append(((i, j), w))
                mses.append(mse)
        # Select the top-k nodes (k = number of features in this layer)
        k = current_X.shape[1]
        idx_sorted = np.argsort(mses)[:k]  # indexes of k smallest MSE
        layer_coeffs = []
        new_features = []
        for idx in idx_sorted:
            (i, j), w = pairs[idx]
            # Store the coefficients and a pair of indices
            layer_coeffs.append((i, j, w.ravel().tolist()))
            # form the output of the new node as P.dot(w)
            Xi = current_X[:, i].reshape(-1, 1)
            Xj = current_X[:, j].reshape(-1, 1)
            P = np.hstack([
                np.ones((n_samples, 1)),
                Xi,
                Xj,
                Xi * Xi,
                Xi * Xj,
                Xj * Xj
            ])
            new_feat = P.dot(np.array(w).reshape(-1, 1))
            new_features.append(new_feat.ravel())

        new_features = np.array(new_features).T  # size (n_samples x k)
        # Evaluate the best node in this layer (would have the smallest mse)
        best_mse = np.min(np.array(mses)[idx_sorted])
        history.append(best_mse)
        print(f"  Layer {layer+1}: best MSE = {best_mse:.5e}")

        # If the metric has deteriorated on this layer (or it's time to stop) -> exit
        if layer > 0 and history[-1] >= history[-2]:
            print("MSE did not decrease — GMDH stopped")
            break

        # Remember the layer
        layers_info.append(layer_coeffs)
        # Then the "new matrix" of features from this layer becomes the current X
        current_X = new_features.copy()

    print("Fit GMDH: end")
    return layers_info, history

# ------- Neo-Fuzzy block -------
def compute_fuzzy_membership(X: np.ndarray, n_mfs: int):
    """
    Decompose each feature into n_mfs of linguistic functions of triangular form.
    
    Returns:
    - fuzzy_values: np.ndarray (n_samples × (n_features * n_mfs))
    - mf_params: triangle parameters (for visualization and storage)
    
    Algorithm:
    - For each feature, we choose evenly distributed centers between min and max.
    - The width of the triangle is the minimum distance between the centers.
    - We normalize so that the sum over Lisp functions = 1 (if they intersect).
    
    """
    n_samples, n_features = X.shape
    fuzzy_vals = []
    mf_params = []  # list of lists: for each feature mfs parameters
    for feat_idx in range(n_features):
        col = X[:, feat_idx]
        v_min, v_max = col.min(), col.max()
        centers = np.linspace(v_min, v_max, n_mfs)
        width = (v_max - v_min) / (n_mfs - 1) if n_mfs > 1 else (v_max - v_min + 1e-6)
        feat_mfs = []
        for c in centers:
            # Each triangular function: (x - (c-width)) / width to 1 to ((c+width) - x) / width
            feat_mfs.append((c - width, c, c + width))
        mf_params.append(feat_mfs)

        # Calculate mfs for all x:
        mat = np.zeros((n_samples, n_mfs))
        for i, (a, b, c) in enumerate(feat_mfs):
            # piecewise-linear
            left = (col - a) / (b - a)
            right = (c - col) / (c - b)
            vals = np.minimum(np.maximum(np.minimum(left, right), 0), 1)
            mat[:, i] = vals
        # normalize by row (sum of all MFs for a given feature = 1)
        mat_sum = mat.sum(axis=1, keepdims=True) + 1e-8
        mat = mat / mat_sum
        fuzzy_vals.append(mat)

    # Concatenate along features -> matrix (n_samples × (n_features * n_mfs))
    fuzzy_matrix = np.hstack(fuzzy_vals)
    return fuzzy_matrix, mf_params

def train_hybrid_model(X_train, y_train, layers_info, mf_params):
    """
    We train the Neo-Fuzzy layer on top of GMDH:
    - Input: values ​​from the last GMDH layer (they will become the basis for the fuzzy).
    - Convert each feature into n_mfs memberships (fuzzy_matrix).
    - Neo-Fuzzy: adjust linear weights for each fuzzy-input,
    to minimize the MSE between (we measure f(x) is about y).
    
    Returns:
    - Theta: array of Neo-Fuzzy weights (n_features * n_mfs × 1)
    - hist: history of MSE over epochs (for demonstration only; here you can choose a simple LS).
    
    Algorithm (LS-solution):
    f(x) = Z * Theta, where Z is the fuzzy matrix (n × (d*n_mfs)), Theta is the weight vector.
    Theta = (Z^T Z)^(-1) Z^T y (usual pseudo-inverse solution).
    
    """
    print("Train Neo-Fuzzy: start")
    # 1. We create input for fuzzy: X_hybrid is the last layer of GMDH in the form of np.ndarray
    X_hybrid = X_train.copy()  # here X_train is already ideally = output of the last GMDH layer

    # 2. Generate a fuzzy matrix for training data
    Z_train, _ = compute_fuzzy_membership(X_hybrid, FUZZY_MF)

    # 3. Finding Theta via Pseudoinverse (LS)
    #   Theta = (Z^T Z)^{-1} Z^T y
    ZTZ = Z_train.T.dot(Z_train)
    ZTy = Z_train.T.dot(y_train)
    Theta = np.linalg.pinv(ZTZ).dot(ZTy)  # (d*n_mfs × 1)

    # 4. Calculate train-mse
    y_pred_train = Z_train.dot(Theta)
    train_mse = np.mean((y_train - y_pred_train) ** 2)
    print(f"  Neo-Fuzzy train MSE: {train_mse:.5e}")

    print("🔹 Train Neo-Fuzzy: end")
    return Theta




def main():
    # 1) Loading scaled data
    X_train, y_train, X_test, y_test = load_data()

    root = Path(__file__).resolve().parents[1]
    model_dir = root / "models"
    model_dir.mkdir(exist_ok=True)

    # 2) Train GMDH на train-set
    gmdh_layers, gmdh_history = fit_gmdh(X_train, y_train, max_layers=8)

    # Save for plot_compare.py
    joblib.dump(gmdh_layers, model_dir / "gmdh_layers.pkl")


    # 3) Form the input for the last GMDH layer
    # To do this, we will run through the same code as in fit_gmdh,
    # but we will only store the NUMBERS of the input features of the final layer
    last_layer_info = gmdh_layers[-1]  # list [(i, j, coeffs), …] for the last layer
    print(f"Last GMDH layer info: {last_layer_info}")
    # Create the matrix X_last: (n_samples × len(last_layer_info))
    n_samples = X_train.shape[0]
    X_last = np.zeros((n_samples, len(last_layer_info)))
    for idx, (i, j, w_list) in enumerate(last_layer_info):
        Xi = X_train[:, i].reshape(-1, 1)
        Xj = X_train[:, j].reshape(-1, 1)
        P = np.hstack([
            np.ones((n_samples, 1)),
            Xi,
            Xj,
            Xi * Xi,
            Xi * Xj,
            Xj * Xj
        ])
        w = np.array(w_list).reshape(-1, 1)
        X_last[:, idx] = P.dot(w).ravel()

    # 4) Prepare MF parameters from the training layer (write down how the triangles are arranged)
    # Here we simply take the range of each feature in X_last
    # and calculate the centers for the FUZZY_MF triangular functions
    _, mf_params = compute_fuzzy_membership(X_last, FUZZY_MF)

    # 5) Train Neo-Fuzzy on the outputs GMDH
    Theta = train_hybrid_model(X_last, y_train, gmdh_layers, mf_params)

    # 6) Evaluate on test-set
    # First get X_last_test for test-set
    n_test = X_test.shape[0]
    X_last_test = np.zeros((n_test, len(last_layer_info)))
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

    # Create fuzzy-matrix for X_last_test
    Z_test, _ = compute_fuzzy_membership(X_last_test, FUZZY_MF)
    # Faire the prediction
    y_pred_test = Z_test.dot(Theta)
    # We evaluate in scale (the scale is already applied to y_test, so it is skewed)
    test_mse = mean_squared_error(y_test, y_pred_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)

    print(f"Hybrid Test MSE (scaled): {test_mse:.5e}")
    print(f"Hybrid Test MAE (scaled): {test_mae:.5e}")

    # 7) Save the results
    root = Path(__file__).resolve().parents[1]
    
    np.save(model_dir / "hybrid_theta.npy", Theta)
    with open(model_dir / "hybrid_gmdh_history.json", "w") as f:
        json.dump(gmdh_history, f)

    print("Hybrid model parameters and history saved to models/")

if __name__ == "__main__":
    main()
