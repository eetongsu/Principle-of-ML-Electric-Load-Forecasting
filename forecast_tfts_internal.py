#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Load forecasting with TFTS (time-series-prediction) — parameters are set INSIDE this file.

What you edit:
- DATA_CSV: path to the Kaggle CSV (downloaded locally)
- CONFIG: lookback/horizon/epochs/etc.
- MODELS: a single model name OR a list of model names to run back-to-back.

Docs:
- https://time-series-prediction.readthedocs.io/en/latest/models.html
- https://time-series-prediction.readthedocs.io/en/latest/api/tfts.trainer.KerasTrainer.html

Install:
  pip install tfts
Run:
  python forecast_tfts_internal.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tfts import AutoConfig, AutoModel
from tfts.trainer import KerasTrainer


# =============================
# 1) USER SETTINGS (EDIT HERE)
# =============================

# Path to your downloaded Kaggle CSV (edit this!)
DATA_CSV = r"./load_forecasting_dataset.csv"

# Easy model switching:
# - Set MODELS to a string: "informer"
# - Or set MODELS to a list: ["rnn", "tcn", "transformer", "informer"]
MODELS: Union[str, List[str]] = "informer"

# Central config (edit as needed)
CONFIG: Dict[str, Any] = dict(
    seed=42,
    train_length=168,   # lookback, e.g., 168 = 7*24 hours
    pred_length=24,     # horizon,  e.g., 24 = next day
    stride=1,

    val_frac=0.15,
    test_frac=0.15,

    epochs=10,
    batch_size=64,
    lr=3e-3,

    output_dir="outputs",
    save_plot=True,     # save one example plot per model
)

# If you want a very quick toggle between common setups:
PRESETS = {
    "day_ahead": dict(train_length=168, pred_length=24),
    "week_ahead": dict(train_length=336, pred_length=168),
    "short_term": dict(train_length=48, pred_length=6),
}
# Example usage (uncomment one line):
# CONFIG.update(PRESETS["day_ahead"])
# CONFIG.update(PRESETS["week_ahead"])
# CONFIG.update(PRESETS["short_term"])


# =============================
# 2) IMPLEMENTATION
# =============================

def set_seed(seed: int = 42) -> None:
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

def _normalize_colname(s: str) -> str:
    return "".join(ch.lower() for ch in s if ch.isalnum())

def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    norm_map = {_normalize_colname(c): c for c in df.columns}
    for cand in candidates:
        key = _normalize_colname(cand)
        if key in norm_map:
            return norm_map[key]
    for cand in candidates:
        key = _normalize_colname(cand)
        for k, v in norm_map.items():
            if key in k:
                return v
    return None

@dataclass
class WindowedData:
    x: np.ndarray                 # (samples, train_length, 1)
    enc_f: np.ndarray             # (samples, train_length, n_features)
    dec_f: np.ndarray             # (samples, pred_length, n_features)
    y: np.ndarray                 # (samples, pred_length, 1)
    future_index: pd.DatetimeIndex  # forecast origin timestamps


def make_windows(
    y: np.ndarray,
    features: np.ndarray,
    index: pd.DatetimeIndex,
    train_length: int,
    pred_length: int,
    stride: int = 1,
) -> WindowedData:
    n = len(y)
    if n != features.shape[0] or n != len(index):
        raise ValueError("y, features, and index must have the same length")

    max_start = n - train_length - pred_length
    if max_start <= 0:
        raise ValueError(f"Not enough data: n={n}, train_length={train_length}, pred_length={pred_length}")

    starts = np.arange(0, max_start + 1, stride, dtype=np.int64)
    m = len(starts)

    x = np.zeros((m, train_length, 1), dtype=np.float32)
    enc_f = np.zeros((m, train_length, features.shape[1]), dtype=np.float32)
    dec_f = np.zeros((m, pred_length, features.shape[1]), dtype=np.float32)
    y_out = np.zeros((m, pred_length, 1), dtype=np.float32)

    future_index = []
    for j, s in enumerate(starts):
        x[j, :, 0] = y[s : s + train_length]
        enc_f[j, :, :] = features[s : s + train_length, :]
        dec_f[j, :, :] = features[s + train_length : s + train_length + pred_length, :]
        y_out[j, :, 0] = y[s + train_length : s + train_length + pred_length]
        future_index.append(index[s + train_length])

    return WindowedData(x=x, enc_f=enc_f, dec_f=dec_f, y=y_out, future_index=pd.DatetimeIndex(future_index))


def to_tf_dataset(w: WindowedData, batch_size: int, shuffle: bool, seed: int) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices(((w.x, w.enc_f, w.dec_f), w.y))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(10000, len(w.y)), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    eps = 1e-8
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mape = float(np.mean(np.abs(err) / (np.abs(y_true) + eps))) * 100.0
    return {"MAE": mae, "RMSE": rmse, "MAPE_%": mape}


def load_and_prepare(csv_path: str) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, str]:
    df = pd.read_csv(csv_path)

    ts_col = _find_col(df, ["Timestamp", "timestamp", "Date", "Datetime"])
    hour_col = _find_col(df, ["Hour", "Hour of D", "Hour of Day", "hour"])
    target_col = _find_col(df, ["Load Demand (kW)", "Load Demand", "LoadDemand", "Load (kW)", "Load"])

    if ts_col is None or target_col is None:
        raise ValueError(f"Missing required columns. Found: {list(df.columns)}")

    dt = pd.to_datetime(df[ts_col], errors="coerce")
    if dt.isna().any():
        raise ValueError(f"Failed to parse some timestamps from '{ts_col}'")

    if hour_col is not None:
        hour = pd.to_numeric(df[hour_col], errors="coerce").fillna(0).astype(int).clip(0, 23)
        dt = dt + pd.to_timedelta(hour, unit="h")

    df["_dt"] = dt
    df = df.sort_values("_dt").reset_index(drop=True)

    # Features: all non-target columns except timestamp/_dt; one-hot encode categoricals (e.g., Season)
    drop_like = {ts_col, "_dt", target_col}
    feature_df = df.drop(columns=[c for c in drop_like if c in df.columns])

    cat_cols = [c for c in feature_df.columns if feature_df[c].dtype == "object"]
    if cat_cols:
        feature_df = pd.get_dummies(feature_df, columns=cat_cols, drop_first=True)

    for c in feature_df.columns:
        feature_df[c] = pd.to_numeric(feature_df[c], errors="coerce")

    feature_df = feature_df.ffill().bfill().fillna(0.0)

    y = pd.to_numeric(df[target_col], errors="coerce").ffill().bfill().astype(float).to_numpy()
    X = feature_df.to_numpy(dtype=float)
    idx = pd.DatetimeIndex(df["_dt"])
    return y, X, idx, target_col


def run_one_model(model_name: str, y: np.ndarray, X: np.ndarray, idx: pd.DatetimeIndex, target_name: str, cfg: Dict[str, Any]) -> None:
    os.makedirs(cfg["output_dir"], exist_ok=True)
    set_seed(int(cfg["seed"]))

    n = len(y)
    test_n = int(round(n * float(cfg["test_frac"])))
    val_n = int(round(n * float(cfg["val_frac"])))
    train_n = n - val_n - test_n

    L = int(cfg["train_length"])
    H = int(cfg["pred_length"])
    stride = int(cfg["stride"])

    if train_n <= L + H:
        raise ValueError("Train split too small. Reduce val/test fractions or window sizes.")

    # Scale (fit on train only)
    y_scaler = StandardScaler()
    X_scaler = StandardScaler()

    y_train_scaled = y_scaler.fit_transform(y[:train_n].reshape(-1, 1)).reshape(-1)
    X_train_scaled = X_scaler.fit_transform(X[:train_n, :])

    y_scaled = y_scaler.transform(y.reshape(-1, 1)).reshape(-1)
    X_scaled = X_scaler.transform(X)

    # Windows
    all_w = make_windows(y_scaled, X_scaled, idx, train_length=L, pred_length=H, stride=stride)

    origin_pos = (np.arange(len(all_w.future_index)) * stride) + L
    train_mask = origin_pos < train_n
    val_mask = (origin_pos >= train_n) & (origin_pos < (train_n + val_n))
    test_mask = origin_pos >= (train_n + val_n)

    def subset(mask: np.ndarray) -> WindowedData:
        return WindowedData(
            x=all_w.x[mask],
            enc_f=all_w.enc_f[mask],
            dec_f=all_w.dec_f[mask],
            y=all_w.y[mask],
            future_index=all_w.future_index[mask],
        )

    w_train = subset(train_mask)
    w_val = subset(val_mask)
    w_test = subset(test_mask)

    if len(w_train.y) < 10 or len(w_test.y) < 10:
        raise ValueError("Too few windows after splitting. Try reducing L/H or val/test fractions.")

    ds_train = to_tf_dataset(w_train, batch_size=int(cfg["batch_size"]), shuffle=True, seed=int(cfg["seed"]))
    ds_val = to_tf_dataset(w_val, batch_size=int(cfg["batch_size"]), shuffle=False, seed=int(cfg["seed"])) if len(w_val.y) else None

    # Model
    config = AutoConfig.for_model(model_name)
    model = AutoModel.from_config(config, predict_sequence_length=H)
    trainer = KerasTrainer(model)

    print(f"\n=============================")
    print(f"Model: {model_name} | L={L} H={H} | train/val/test windows = {len(w_train.y)}/{len(w_val.y)}/{len(w_test.y)}")
    print(f"=============================")

    trainer.train(
        ds_train,
        ds_val,
        loss_fn="mse",
        optimizer=tf.keras.optimizers.Adam(float(cfg["lr"])),
        epochs=int(cfg["epochs"]),
        batch_size=int(cfg["batch_size"]),
        run_eagerly=False,
        verbose=1,
    )

    # Predict
    x_test = (w_test.x, w_test.enc_f, w_test.dec_f)
    y_pred_scaled = np.asarray(trainer.predict(x_test), dtype=np.float32)
    if y_pred_scaled.ndim == 2:
        y_pred_scaled = y_pred_scaled[..., None]

    y_true = y_scaler.inverse_transform(w_test.y.reshape(-1, 1)).reshape(w_test.y.shape)
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).reshape(y_pred_scaled.shape)

    metrics = compute_metrics(y_true, y_pred)
    print("\nTest metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Save predictions
    out = pd.DataFrame({"forecast_origin": w_test.future_index})
    for h in range(H):
        out[f"y_true_t+{h+1}"] = y_true[:, h, 0]
        out[f"y_pred_t+{h+1}"] = y_pred[:, h, 0]

    out_path = os.path.join(cfg["output_dir"], f"predictions_{model_name}_L{L}_H{H}.csv")
    out.to_csv(out_path, index=False)
    print(f"\nSaved predictions: {out_path}")

    # Save model
    model_dir = os.path.join(cfg["output_dir"], f"model_{model_name}_L{L}_H{H}")
    trainer.save_model(model_dir)
    print(f"Saved model:       {model_dir}")

    # Optional plot
    if bool(cfg.get("save_plot", False)):
        import matplotlib.pyplot as plt
        i = 0
        plt.figure()
        plt.plot(np.arange(1, H + 1), y_true[i, :, 0], label="true")
        plt.plot(np.arange(1, H + 1), y_pred[i, :, 0], label="pred")
        plt.xlabel("Horizon step")
        plt.ylabel(target_name)
        plt.title(f"{model_name} forecast (origin={w_test.future_index[i]})")
        plt.legend()
        plot_path = os.path.join(cfg["output_dir"], f"example_forecast_{model_name}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot:        {plot_path}")


def main() -> None:
    if not os.path.exists(DATA_CSV):
        raise FileNotFoundError(f"DATA_CSV not found: {DATA_CSV}\nEdit DATA_CSV at the top of the file.")

    y, X, idx, target_name = load_and_prepare(DATA_CSV)

    models = MODELS if isinstance(MODELS, list) else [MODELS]

    # Simple “switch method”: just put multiple models in MODELS and it will run them sequentially.
    # You can also compare results by checking each model's metrics and prediction CSV.
    for m in models:
        run_one_model(str(m), y, X, idx, target_name, CONFIG)


if __name__ == "__main__":
    main()
