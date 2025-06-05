# Hybrid Deep Forecasting (CSCO)

**Topic:** Research of Hybrid Deep Learning Networks (GMDH + Neo-Fuzzy) for Time Series Forecasting and Comparison with a Back-Propagation Neural Network (MLP)

---

## 🧠 Project Summary

This project explores a hybrid deep learning architecture combining **GMDH** (Group Method of Data Handling) with a **Neo-Fuzzy Neural Network** to forecast financial time series. The hybrid system is benchmarked against a traditional **Multilayer Perceptron (MLP)** trained via back-propagation. 

## 📁 Data Source

The raw CSCO OHLC data (2006–2018) is downloaded from Yahoo Finance:

- **Cisco Systems (CSCO) Historical Data:**  
  https://finance.yahoo.com/quote/CSCO/history

You can manually verify or download CSV from that page, or let the `download_data.py` script fetch it automatically via the Yahoo Finance API.


> 🧾 *This work draws methodological inspiration from the scientific contributions of Prof. **Yuriy Zaychenko**. See his profile: [Google Scholar – Yuriy Zaychenko](https://scholar.google.com.ua/citations?user=mzGS8GrJhKEC&hl=ua)*

---

## ⚙️ Architecture Overview

### 🔹 GMDH: Group Method of Data Handling

Each layer of the GMDH network consists of polynomial neurons, where each neuron models a quadratic function:

$$
f(x_i, x_j) = w_0 + w_1x_i + w_2x_j + w_3x_i^2 + w_4x_ix_j + w_5x_j^2
$$

* All feature pairs \$(x\_i, x\_j)\$ are evaluated.
* Least squares regression determines weights \$w\$.
* Neurons with lowest MSE are retained to form the next layer.
* Training stops when validation error increases.

### 🔹 Neo-Fuzzy Neural Layer

On top of GMDH outputs, a fuzzy logic layer applies 3 triangular **membership functions (MFs)** per input:

* For each \$x\$, compute degrees of membership to **low**, **medium**, and **high** regions.
* Normalize to ensure \$, \sum \mu\_i(x) = 1\$.

The output layer solves:

$$
\Theta = (Z^T Z)^{-1} Z^T y
$$

Where \$Z\$ is the fuzzy-transformed feature matrix.

### 🔹 MLP Baseline

Standard fully-connected neural network with two hidden layers:

```
Input → Dense(32, ReLU) → Dense(16, ReLU) → Dense(1, Linear)
```

* Optimizer: Adam
* Loss: MSE
* Metric: MAE
* EarlyStopping (patience=20)

---

## 📊 Results Summary

| Model                         | Test MAE (scaled) | Approx. MAE (USD) |
| ----------------------------- | ----------------- | ----------------- |
| **MLP (Back-Propagation)**    | 0.0297            | \$0.30 – \$0.40   |
| **Hybrid (GMDH + Neo-Fuzzy)** | 0.2276            | \$2.00 – \$3.00   |

* **MLP** shows strong prediction accuracy and generalization.
* **Hybrid** model offers a modular design, though underperforms in this setup.

---

## 🗺️ Project Structure

```
.
├── data/
│   ├── raw/
│   │   └── csco.csv               # Raw OHLC data (CSCO, 2006–2018)
│   └── processed/
│       ├── train.csv              # Scaled training data
│       └── test.csv               # Scaled test data
├── figures/
│   └── compare_plot.png           # Real vs MLP vs Hybrid predictions
├── models/
│   ├── mlp_baseline.h5            # MLP model
│   ├── mlp_history.json           # Training history
│   ├── hybrid_theta.npy           # Neo-Fuzzy weights
│   ├── hybrid_gmdh_history.json   # GMDH MSE history
│   ├── gmdh_layers.pkl            # GMDH structure
│   ├── scaler_X.pkl               # Scaler for inputs
│   └── scaler_y.pkl               # Scaler for target
├── src/
│   ├── download_data.py           # CSCO data from Yahoo
│   ├── preprocess.py              # Feature engineering
│   ├── train_mlp.py               # Train MLP model
│   ├── train_hybrid.py            # Train Hybrid model
│   └── plot_compare.py            # Generate prediction plots
└── README.md    <-- You are here
```

---

## 🔭 Next Steps

1. **Enhance the Hybrid Layer**  
   - Increase/smooth membership functions (e.g., 5–7 Gaussians).  
   - Make MF parameters trainable (ANFIS-style).  
   - Regularize or prune GMDH neurons to reduce overfitting.

2. **Add True Time-Series Models**  
   - **LSTM/GRU:** reshape data to `(samples, timesteps, features)` and compare against MLP & Hybrid.  
   - **ARIMA/Prophet:** model trend/seasonality, then feed residuals to MLP/Neo-Fuzzy.  
   - **TCN (Temporal Conv):** use dilated 1D convolutions to capture longer contexts.

3. **Robust Validation & Comparison**  
   - Implement rolling-window cross-validation instead of a single 70/30 split.  
   - Introduce simple baselines (e.g., naïve forecast, AR(1)).  
   - Experiment with ensembles (MLP + LSTM + Hybrid) to reduce error.

4. **Hyperparameter & Feature Search**  
   - Test larger lag windows (10–20 days) or add technical indicators.  
   - Vary GMDH depth and neuron selection criteria (e.g., AIC/BIC).  
   - Explore different MF types (Gaussian, bell-shaped) and counts.

5. **Extend to Other Datasets & Tasks**  
   - Apply the pipeline to additional tickers (AAPL, MSFT) or indices.  
   - Try multi-step forecasting (5-day or 10-day ahead).  
   - Forecast related targets like volatility or volume in a multi-output setup.

6. **Thesis/Documentation Integration**  
   - Draft a literature review on GMDH, Neo-Fuzzy, and time-series DL methods.  
   - Detail mathematical foundations (GMDH equations, fuzzy membership, back-prop).  
   - Include ablation studies and rolling-window results in the final write-up.

> *These steps outline potential directions—pick a few (e.g., LSTM comparison and rolling-window validation) to implement next.*

---

## 🙋 Questions?

Feel free to open an issue or reach out on [LinkedIn](https://www.linkedin.com/in/olha--tytarenko/) or via [email](olhatytarenko03@gmail.com) if you’re interested in similar research.
