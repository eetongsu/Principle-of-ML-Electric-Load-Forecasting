#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Load forecasting with TFTS (tfts) — OOM-safe version.

Fixes your error:
- The OOM happens during prediction because we were calling the model on ALL test windows at once,
  creating an attention tensor like [num_windows, L, L] (e.g., [37980,168,168]).
- This version predicts in BATCHES using a tf.data.Dataset, so GPU memory stays bounded.

Keeps the reference-style pipeline:
- time features + optional one-hot Season
- chronological train/test split (shuffle=False)
- StandardScaler fit on TRAIN only
- metrics: RMSE/MAPE (step-1 and full-horizon)
- plot Actual vs Predicted (t+1)

Install:
  pip install tfts
Run:
  python forecast_tfts_internal_v3.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tfts import AutoConfig, AutoModel
from tfts.trainer import KerasTrainer


# =============================
# 1) USER SETTINGS (EDIT HERE)
# =============================

DATA_CSV = r"load_forecasting_dataset_corrected.csv"

MODELS: Union[str, List[str]] = "informer"

CONFIG: Dict[str, Any] = dict(
    seed=42,

    train_length=168,   # L
    pred_length=24,     # H
    stride=1,           # increase (e.g., 2/3/6) to reduce windows (and speed up)

    test_size=0.2,
    val_size=0.1,

    epochs=10,
    batch_size=64,          # training batch size
    predict_batch_size=256, # inference batch size (reduce if still OOM)
    lr=3e-3,

    output_dir="outputs",
    save_plot=True,

    # GPU memory behavior:
    enable_memory_growth=True,  # usually helps on consumer GPUs
    mixed_precision=False,      # set True to reduce memory (needs recent TF + GPU)
)

PRESETS = {
    "day_ahead": dict(train_length=168, pred_length=24),
    "week_ahead": dict(train_length=336, pred_length=168),
    "short_term": dict(train_length=48, pred_length=6),
}
# CONFIG.update(PRESETS["day_ahead"])


# =============================
# 2) FEATURE ENGINEERING
# =============================

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    idx = out.index
    out["hour"] = idx.hour
    out["minute"] = idx.minute
    out["dayofweek"] = idx.dayofweek
    out["quarter"] = idx.quarter
    out["month"] = idx.month
    out["day"] = idx.day
    out["year"] = idx.year
    out["season_num"] = (out["month"] % 12) // 3 + 1
    out["dayofyear"] = idx.dayofyear
    out["dayofmonth"] = idx.day
    out["weekofyear"] = idx.isocalendar().week.astype(int)

    out["is_weekend"] = out["dayofweek"].isin([5, 6]).astype(int)
    out["is_month_start"] = (out["dayofmonth"] == 1).astype(int)
    out["is_month_end"] = (out["dayofmonth"] == idx.days_in_month).astype(int)
    out["is_working_day"] = out["dayofweek"].isin([0, 1, 2, 3, 4]).astype(int)
    out["is_business_hours"] = out["hour"].between(9, 17).astype(int)
    out["is_peak_hour"] = out["hour"].isin([8, 12, 18]).astype(int)

    out["minute_of_day"] = out["hour"] * 60 + out["minute"]
    out["minute_of_week"] = out["dayofweek"] * 24 * 60 + out["minute_of_day"]
    return out


# =============================
# 3) CORE PIPELINE
# =============================

def set_seed(seed: int = 42) -> None:
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

def configure_tf(cfg: Dict[str, Any]) -> None:
    if cfg.get("enable_memory_growth", True):
        try:
            gpus = tf.config.list_physical_devices("GPU")
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    if cfg.get("mixed_precision", False):
        try:
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy("mixed_float16")
        except Exception:
            print("Warning: mixed_precision requested but could not be enabled.")

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
    x: np.ndarray
    enc_f: np.ndarray
    dec_f: np.ndarray
    y: np.ndarray
    origin_time: pd.DatetimeIndex

def make_windows(
    y: np.ndarray,
    features: np.ndarray,
    index: pd.DatetimeIndex,
    L: int,
    H: int,
    stride: int = 1,
) -> WindowedData:
    n = len(y)
    max_start = n - L - H
    if max_start <= 0:
        raise ValueError(f"Not enough data for windows: n={n}, L={L}, H={H}")

    starts = np.arange(0, max_start + 1, stride, dtype=np.int64)
    m = len(starts)

    x = np.zeros((m, L, 1), dtype=np.float32)
    enc_f = np.zeros((m, L, features.shape[1]), dtype=np.float32)
    dec_f = np.zeros((m, H, features.shape[1]), dtype=np.float32)
    y_out = np.zeros((m, H, 1), dtype=np.float32)
    origins = []

    for j, t in enumerate(starts):
        x[j, :, 0] = y[t : t + L]
        enc_f[j, :, :] = features[t : t + L, :]
        dec_f[j, :, :] = features[t + L : t + L + H, :]
        y_out[j, :, 0] = y[t + L : t + L + H]
        origins.append(index[t + L])

    return WindowedData(x=x, enc_f=enc_f, dec_f=dec_f, y=y_out, origin_time=pd.DatetimeIndex(origins))

def to_tf_dataset(w: WindowedData, batch_size: int, shuffle: bool, seed: int) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices(((w.x, w.enc_f, w.dec_f), w.y))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(10000, len(w.y)), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def to_tf_x_dataset(x: Tuple[np.ndarray, np.ndarray, np.ndarray], batch_size: int) -> tf.data.Dataset:
    x1, x2, x3 = x
    ds = tf.data.Dataset.from_tensor_slices((x1, x2, x3))
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = y_pred - y_true
    return float(np.sqrt(np.mean(err ** 2)))

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps = 1e-8
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100.0)

def load_prepare(csv_path: str) -> Tuple[pd.DataFrame, str]:
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
    df = df.sort_values("_dt").reset_index(drop=True).set_index("_dt")

    df = create_time_features(df)

    if "Season" in df.columns:
        df = pd.get_dummies(df, columns=["Season"], prefix="season", dtype=int)

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.ffill().bfill().fillna(0.0)

    return df, target_col

def split_chrono(df: pd.DataFrame, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    n_test = int(round(n * float(test_size)))
    n_test = max(1, min(n - 1, n_test))
    train = df.iloc[: n - n_test].copy()
    test = df.iloc[n - n_test :].copy()
    return train, test

def split_val_from_train(train: pd.DataFrame, val_size: float) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    if val_size is None or float(val_size) <= 0:
        return train, None
    n = len(train)
    n_val = int(round(n * float(val_size)))
    n_val = max(1, min(n - 1, n_val))
    tr = train.iloc[: n - n_val].copy()
    va = train.iloc[n - n_val :].copy()
    return tr, va

def predict_in_batches(trainer: KerasTrainer, x_test: Tuple[np.ndarray, np.ndarray, np.ndarray], batch_size: int) -> np.ndarray:
    """
    OOM-safe prediction: iterate over batches.
    Returns: (samples, H, 1) or (samples, H)
    """
    ds = to_tf_x_dataset(x_test, batch_size=batch_size)
    preds = []
    for batch in ds:
        # batch is a tuple (x, enc_f, dec_f)
        yb = trainer.model(batch, training=False)
        preds.append(yb.numpy())
    return np.concatenate(preds, axis=0)

def run_one_model(model_name: str, df: pd.DataFrame, target_col: str, cfg: Dict[str, Any]) -> None:
    os.makedirs(cfg["output_dir"], exist_ok=True)
    set_seed(int(cfg["seed"]))
    configure_tf(cfg)

    L = int(cfg["train_length"])
    H = int(cfg["pred_length"])
    stride = int(cfg["stride"])

    df_train_all, df_test = split_chrono(df, test_size=float(cfg["test_size"]))
    df_train, df_val = split_val_from_train(df_train_all, val_size=float(cfg.get("val_size", 0.0)))

    X_train = df_train.drop(columns=[target_col])
    y_train = df_train[[target_col]]

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train.values)
    y_train_scaled = scaler_y.fit_transform(y_train.values).reshape(-1)

    w_val = None
    if df_val is not None and len(df_val) > (L + H + 1):
        X_val_scaled = scaler_X.transform(df_val.drop(columns=[target_col]).values)
        y_val_scaled = scaler_y.transform(df_val[[target_col]].values).reshape(-1)
        w_val = make_windows(y_val_scaled, X_val_scaled, df_val.index, L=L, H=H, stride=stride)

    w_train = make_windows(y_train_scaled, X_train_scaled, df_train.index, L=L, H=H, stride=stride)

    # test with context
    ctx = df_train_all.tail(L + H + 1)
    df_for_test = pd.concat([ctx, df_test], axis=0)

    X_for_test_scaled = scaler_X.transform(df_for_test.drop(columns=[target_col]).values)
    y_for_test_scaled = scaler_y.transform(df_for_test[[target_col]].values).reshape(-1)
    w_test = make_windows(y_for_test_scaled, X_for_test_scaled, df_for_test.index, L=L, H=H, stride=stride)

    ds_train = to_tf_dataset(w_train, batch_size=int(cfg["batch_size"]), shuffle=True, seed=int(cfg["seed"]))
    ds_val = to_tf_dataset(w_val, batch_size=int(cfg["batch_size"]), shuffle=False, seed=int(cfg["seed"])) if w_val else None

    config = AutoConfig.for_model(model_name)
    model = AutoModel.from_config(config, predict_sequence_length=H)
    trainer = KerasTrainer(model)

    print(f"\n=============================")
    print(f"Model: {model_name}")
    print(f"L={L}, H={H}, stride={stride}")
    print(f"Rows: train={len(df_train)}, val={len(df_val) if df_val is not None else 0}, test={len(df_test)}")
    print(f"Windows: train={len(w_train.y)}, val={len(w_val.y) if w_val else 0}, test={len(w_test.y)}")
    print(f"Predict batch size: {int(cfg.get('predict_batch_size', 256))}")
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

    # OOM-safe predict
    x_test = (w_test.x, w_test.enc_f, w_test.dec_f)
    y_pred_scaled = np.asarray(
        predict_in_batches(trainer, x_test, batch_size=int(cfg.get("predict_batch_size", 256))),
        dtype=np.float32
    )
    if y_pred_scaled.ndim == 2:
        y_pred_scaled = y_pred_scaled[..., None]

    y_true = scaler_y.inverse_transform(w_test.y.reshape(-1, 1)).reshape(w_test.y.shape)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).reshape(y_pred_scaled.shape)

    y_true_step1 = y_true[:, 0, 0]
    y_pred_step1 = y_pred[:, 0, 0]
    rmse_step1 = rmse(y_true_step1, y_pred_step1)
    mape_step1 = mape(y_true_step1, y_pred_step1)

    rmse_all = rmse(y_true.reshape(-1), y_pred.reshape(-1))
    mape_all = mape(y_true.reshape(-1), y_pred.reshape(-1))

    print("\nTest metrics (original scale):")
    print(f"  Step-1 RMSE:  {rmse_step1:.4f}")
    print(f"  Step-1 MAPE:  {mape_step1:.4f}%")
    print(f"  Horizon RMSE: {rmse_all:.4f}")
    print(f"  Horizon MAPE: {mape_all:.4f}%")

    out = pd.DataFrame({"timestamp": w_test.origin_time})
    for h in range(H):
        out[f"y_true_t+{h+1}"] = y_true[:, h, 0]
        out[f"y_pred_t+{h+1}"] = y_pred[:, h, 0]

    out_path = os.path.join(cfg["output_dir"], f"predictions_{model_name}_L{L}_H{H}.csv")
    out.to_csv(out_path, index=False)
    print(f"\nSaved predictions: {out_path}")

    model_dir = os.path.join(cfg["output_dir"], f"model_{model_name}_L{L}_H{H}")
    trainer.save_model(model_dir)
    print(f"Saved model:       {model_dir}")

    keras_dir = os.path.join(cfg["output_dir"], f"keras_savedmodel_{model_name}_L{L}_H{H}")
    trainer.model.save(keras_dir, include_optimizer=False, save_format="tf")
    print(f"Saved keras model: {keras_dir}")

    if bool(cfg.get("save_plot", False)):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        plt.plot(out["timestamp"], out["y_true_t+1"], label="Actual")
        plt.plot(out["timestamp"], out["y_pred_t+1"], label="Predicted")
        plt.title(f"Actual vs Predicted Load Demand (TFTS {model_name})")
        plt.xlabel("Timestamp")
        plt.ylabel(target_col)
        plt.legend()
        plt.tight_layout()
        plot_path = os.path.join(cfg["output_dir"], f"actual_vs_pred_{model_name}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot:        {plot_path}")

def main() -> None:
    if not os.path.exists(DATA_CSV):
        raise FileNotFoundError(f"DATA_CSV not found: {DATA_CSV}\nEdit DATA_CSV at the top of the file.")

    df, target_col = load_prepare(DATA_CSV)
    models = MODELS if isinstance(MODELS, list) else [MODELS]
    for m in models:
        run_one_model(str(m), df, target_col, CONFIG)

if __name__ == "__main__":
    main()
