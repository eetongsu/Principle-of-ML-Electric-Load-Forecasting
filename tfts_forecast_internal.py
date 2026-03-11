#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Load forecasting with TFTS (tfts) — robust version.

Minimal fixes included:
1) Remove these columns from future features (dec_f) only:
   - GDP (LKR)
   - Per Capita Energy Use (kWh)
   - Electricity Price (LKR/kWh)
2) Add weather uncertainty to future meteorological features only:
   - Temperature
   - Humidity
   - Wind Speed
   - Rainfall
   - Solar Irradiance
   with max relative error = 5% of each value.
3) Fix tfts SavedModel export failure caused by DecoderV1 mutating Python input args.
4) Clean output paths before saving to avoid repeated-save collisions.
5) Fix GPU OOM during prediction with adaptive batch shrinking and CPU fallback.
"""

from __future__ import annotations

import os
import shutil
import logging
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple, Union

# These environment variables must be set before importing TensorFlow.
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

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

MODELS: Union[str, List[str]] = ['informer', 'rnn', 'tcn', 'wavenet', 'unet', 'transformer', 'bert', 'tft', 'nbeats']

CONFIG: Dict[str, Any] = dict(
    seed=42,

    train_length=168,   # L
    pred_length=24,     # H
    stride=1,

    test_size=0.2,
    val_size=0.1,

    epochs=10,
    batch_size=32,
    predict_batch_size=16,
    min_predict_batch_size=1,
    lr=3e-3,

    output_dir="outputs",
    save_plot=True,

    enable_memory_growth=True,
    mixed_precision=False,
    force_cpu_predict_on_oom=True,

    weather_relative_error=0.05,

    save_trainer_model=True,
    save_savedmodel=True,
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
    np.random.seed(seed)
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


def get_future_excluded_cols(df: pd.DataFrame) -> List[str]:
    """Identify columns that should be excluded from decoder future features."""
    excluded = []
    for candidates in [
        ["GDP (LKR)", "GDP"],
        ["Per Capita Energy Use (kWh)", "Per Capita Energy Use"],
        ["Electricity Price (LKR/kWh)", "Electricity Price"],
    ]:
        col = _find_col(df, candidates)
        if col is not None:
            excluded.append(col)
    return excluded


def get_weather_cols(df: pd.DataFrame) -> List[str]:
    """Identify meteorological columns."""
    weather = []
    for candidates in [
        ["Temperature (°C)", "Temperature (??C)", "Temperature"],
        ["Humidity (%)", "Humidity"],
        ["Wind Speed (m/s)", "Wind Speed"],
        ["Rainfall (mm)", "Rainfall"],
        ["Solar Irradiance (W/m²)", "Solar Irradiance (W/m??)", "Solar Irradiance"],
    ]:
        col = _find_col(df, candidates)
        if col is not None:
            weather.append(col)
    return weather


def add_relative_noise(
    df: pd.DataFrame,
    cols: List[str],
    max_rel_err: float,
    seed: int = 42,
) -> pd.DataFrame:
    """Add bounded relative noise to selected columns."""
    out = df.copy()
    if not cols or max_rel_err <= 0:
        return out

    rng = np.random.default_rng(seed)
    for c in cols:
        if c in out.columns:
            eps = rng.uniform(-max_rel_err, max_rel_err, size=len(out))
            out[c] = out[c].astype(float) * (1.0 + eps)
    return out


def remove_path(path: str) -> None:
    """Remove an existing file or directory before saving new artifacts."""
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    elif os.path.isfile(path):
        try:
            os.remove(path)
        except OSError:
            pass


def ensure_clean_dir(path: str) -> None:
    """Delete an existing directory and recreate it as an empty directory."""
    remove_path(path)
    os.makedirs(path, exist_ok=True)


def patch_tfts_decoder_for_savedmodel() -> None:
    """
    Monkey-patch tfts DecoderV1.call so it does not mutate Python input arguments
    observed by TensorFlow SavedModel tracing.
    """
    patch_targets = []
    candidate_modules = [
        "tfts.models.seq2seq",
        "tfts.layers.decoder",
        "tfts.models.tft",
    ]

    for mod_name in candidate_modules:
        try:
            module = __import__(mod_name, fromlist=["*"])
            if hasattr(module, "DecoderV1"):
                patch_targets.append(module.DecoderV1)
        except Exception:
            continue

    if not patch_targets:
        return

    for cls in patch_targets:
        if getattr(cls, "_oa_mutation_patch_applied", False):
            continue

        original_call = cls.call

        def patched_call(self, *args, **kwargs):
            if args:
                args = list(args)
                for i, v in enumerate(args):
                    if isinstance(v, list):
                        args[i] = list(v)
                    elif isinstance(v, dict):
                        args[i] = dict(v)
                args = tuple(args)

            if kwargs:
                kwargs = dict(kwargs)
                for k, v in kwargs.items():
                    if isinstance(v, list):
                        kwargs[k] = list(v)
                    elif isinstance(v, dict):
                        kwargs[k] = dict(v)

            if "encoder_outputs" in kwargs and isinstance(kwargs["encoder_outputs"], list):
                kwargs["encoder_outputs"] = list(kwargs["encoder_outputs"])

            return original_call(self, *args, **kwargs)

        cls.call = patched_call
        cls._oa_mutation_patch_applied = True


@dataclass
class WindowedData:
    x: np.ndarray
    enc_f: np.ndarray
    dec_f: np.ndarray
    y: np.ndarray
    origin_time: pd.DatetimeIndex


def make_windows(
    y: np.ndarray,
    enc_features: np.ndarray,
    dec_features: np.ndarray,
    index: pd.DatetimeIndex,
    L: int,
    H: int,
    stride: int = 1,
) -> WindowedData:
    """Create encoder-decoder windows."""
    n = len(y)
    max_start = n - L - H
    if max_start <= 0:
        raise ValueError(f"Not enough data for windows: n={n}, L={L}, H={H}")

    starts = np.arange(0, max_start + 1, stride, dtype=np.int64)
    m = len(starts)

    x = np.zeros((m, L, 1), dtype=np.float32)
    enc_f = np.zeros((m, L, enc_features.shape[1]), dtype=np.float32)
    dec_f = np.zeros((m, H, dec_features.shape[1]), dtype=np.float32)
    y_out = np.zeros((m, H, 1), dtype=np.float32)
    origins = []

    for j, t in enumerate(starts):
        x[j, :, 0] = y[t: t + L]
        enc_f[j, :, :] = enc_features[t: t + L, :]
        dec_f[j, :, :] = dec_features[t + L: t + L + H, :]
        y_out[j, :, 0] = y[t + L: t + L + H]
        origins.append(index[t + L])

    return WindowedData(
        x=x,
        enc_f=enc_f,
        dec_f=dec_f,
        y=y_out,
        origin_time=pd.DatetimeIndex(origins),
    )


def to_tf_dataset(w: WindowedData, batch_size: int, shuffle: bool, seed: int) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices(((w.x, w.enc_f, w.dec_f), w.y))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(10000, len(w.y)), seed=seed, reshuffle_each_iteration=True)
    # Small prefetch is safer than AUTOTUNE for tight GPU memory.
    return ds.batch(batch_size).prefetch(1)


def to_tf_x_dataset(x: Tuple[np.ndarray, np.ndarray, np.ndarray], batch_size: int) -> tf.data.Dataset:
    x1, x2, x3 = x
    ds = tf.data.Dataset.from_tensor_slices((x1, x2, x3))
    return ds.batch(batch_size).prefetch(1)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = y_pred - y_true
    return float(np.sqrt(np.mean(err ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps = 1e-8
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100.0)


def load_prepare(csv_path: str) -> Tuple[pd.DataFrame, str]:
    df = pd.read_csv(csv_path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.set_index("Timestamp")
    df = create_time_features(df)

    if "Season" in df.columns:
        df = pd.get_dummies(df, columns=["Season"], prefix="season", dtype=int)

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.ffill().bfill().fillna(0.0)

    target_col = "Load Demand (kW)"
    return df, target_col


def split_chrono(df: pd.DataFrame, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    n_test = int(round(n * float(test_size)))
    n_test = max(1, min(n - 1, n_test))
    train = df.iloc[: n - n_test].copy()
    test = df.iloc[n - n_test:].copy()
    return train, test


def split_val_from_train(train: pd.DataFrame, val_size: float) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    if val_size is None or float(val_size) <= 0:
        return train, None
    n = len(train)
    n_val = int(round(n * float(val_size)))
    n_val = max(1, min(n - 1, n_val))
    tr = train.iloc[: n - n_val].copy()
    va = train.iloc[n - n_val:].copy()
    return tr, va


def _predict_once(
    model: tf.keras.Model,
    x_test: Tuple[np.ndarray, np.ndarray, np.ndarray],
    batch_size: int,
    use_cpu: bool = False,
) -> np.ndarray:
    """Run prediction once with a fixed batch size on GPU or CPU."""
    ds = to_tf_x_dataset(x_test, batch_size=batch_size)
    preds = []

    device_name = "/CPU:0" if use_cpu else None
    for batch in ds:
        if device_name is None:
            yb = model(batch, training=False)
        else:
            with tf.device(device_name):
                yb = model(batch, training=False)
        preds.append(yb.numpy())

    return np.concatenate(preds, axis=0)


def predict_in_batches(
    trainer: KerasTrainer,
    x_test: Tuple[np.ndarray, np.ndarray, np.ndarray],
    batch_size: int,
    min_batch_size: int = 1,
    force_cpu_on_oom: bool = True,
) -> np.ndarray:
    """
    Predict robustly.

    Strategy:
    1) Try the requested GPU batch size.
    2) If OOM happens, halve the batch size and retry.
    3) If batch size reaches the minimum and GPU still fails, fall back to CPU.
    """
    current_bs = max(int(batch_size), 1)
    min_batch_size = max(int(min_batch_size), 1)
    last_exc: Optional[Exception] = None

    while current_bs >= min_batch_size:
        try:
            print(f"Predicting with batch_size={current_bs} on GPU...")
            return _predict_once(trainer.model, x_test, batch_size=current_bs, use_cpu=False)
        except (tf.errors.ResourceExhaustedError, tf.errors.InternalError) as e:
            last_exc = e
            msg = str(e).lower()
            is_oom_like = (
                "oom" in msg
                or "out of memory" in msg
                or "cudnnrnn" in msg
                or "failed to call thenrnnforward" in msg
            )
            if not is_oom_like:
                raise

            print(f"Warning: GPU prediction OOM at batch_size={current_bs}. Retrying with a smaller batch.")
            current_bs //= 2
            if current_bs < min_batch_size:
                break

    if force_cpu_on_oom:
        print("Warning: GPU prediction still OOM. Falling back to CPU prediction.")
        return _predict_once(trainer.model, x_test, batch_size=min_batch_size, use_cpu=True)

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Prediction failed for an unknown reason.")


def safe_save_trainer_model(trainer: KerasTrainer, model_dir: str) -> bool:
    """Save trainer-managed model artifacts safely."""
    try:
        ensure_clean_dir(model_dir)
        trainer.save_model(model_dir)
        print(f"Saved model:       {model_dir}")
        return True
    except Exception as e:
        logging.exception("Failed to save trainer model to %s: %s", model_dir, e)
        print(f"Warning: failed to save trainer model to {model_dir}: {e}")
        return False


def safe_export_savedmodel(model: tf.keras.Model, export_dir: str) -> bool:
    """Export TensorFlow SavedModel safely."""
    try:
        remove_path(export_dir)
        tf.saved_model.save(model, export_dir)
        print(f"Saved keras model: {export_dir}")
        return True
    except Exception as e:
        logging.exception("Failed to export SavedModel to %s: %s", export_dir, e)
        print(f"Warning: failed to export SavedModel to {export_dir}: {e}")
        return False


def run_one_model(model_name: str, df: pd.DataFrame, target_col: str, cfg: Dict[str, Any]) -> None:
    os.makedirs(cfg["output_dir"], exist_ok=True)
    set_seed(int(cfg["seed"]))
    configure_tf(cfg)
    patch_tfts_decoder_for_savedmodel()

    L = int(cfg["train_length"])
    H = int(cfg["pred_length"])
    stride = int(cfg["stride"])

    df_train_all, df_test = split_chrono(df, test_size=float(cfg["test_size"]))
    df_train, df_val = split_val_from_train(df_train_all, val_size=float(cfg.get("val_size", 0.0)))

    future_excluded_cols = get_future_excluded_cols(df)
    weather_cols_all = get_weather_cols(df)

    X_train_enc = df_train.drop(columns=[target_col]).copy()
    X_train_dec = X_train_enc.drop(columns=future_excluded_cols, errors="ignore").copy()

    scaler_X_enc = StandardScaler()
    scaler_X_dec = StandardScaler()
    scaler_y = StandardScaler()

    X_train_enc_scaled = scaler_X_enc.fit_transform(X_train_enc.values)

    weather_cols_dec = [c for c in weather_cols_all if c in X_train_dec.columns]
    X_train_dec_noisy = add_relative_noise(
        X_train_dec,
        cols=weather_cols_dec,
        max_rel_err=float(cfg.get("weather_relative_error", 0.05)),
        seed=int(cfg["seed"]),
    )
    X_train_dec_scaled = scaler_X_dec.fit_transform(X_train_dec_noisy.values)

    y_train = df_train[[target_col]]
    y_train_scaled = scaler_y.fit_transform(y_train.values).reshape(-1)

    w_val = None
    if df_val is not None and len(df_val) > (L + H + 1):
        X_val_enc = df_val.drop(columns=[target_col]).copy()
        X_val_dec = X_val_enc.drop(columns=future_excluded_cols, errors="ignore").copy()

        X_val_enc_scaled = scaler_X_enc.transform(X_val_enc.values)

        weather_cols_dec_val = [c for c in weather_cols_all if c in X_val_dec.columns]
        X_val_dec_noisy = add_relative_noise(
            X_val_dec,
            cols=weather_cols_dec_val,
            max_rel_err=float(cfg.get("weather_relative_error", 0.05)),
            seed=int(cfg["seed"]) + 1,
        )
        X_val_dec_scaled = scaler_X_dec.transform(X_val_dec_noisy.values)

        y_val_scaled = scaler_y.transform(df_val[[target_col]].values).reshape(-1)

        w_val = make_windows(
            y=y_val_scaled,
            enc_features=X_val_enc_scaled,
            dec_features=X_val_dec_scaled,
            index=df_val.index,
            L=L,
            H=H,
            stride=stride,
        )

    w_train = make_windows(
        y=y_train_scaled,
        enc_features=X_train_enc_scaled,
        dec_features=X_train_dec_scaled,
        index=df_train.index,
        L=L,
        H=H,
        stride=stride,
    )

    ctx = df_train_all.tail(L + H + 1)
    df_for_test = pd.concat([ctx, df_test], axis=0)

    X_test_enc = df_for_test.drop(columns=[target_col]).copy()
    X_test_dec = X_test_enc.drop(columns=future_excluded_cols, errors="ignore").copy()

    X_test_enc_scaled = scaler_X_enc.transform(X_test_enc.values)

    weather_cols_dec_test = [c for c in weather_cols_all if c in X_test_dec.columns]
    X_test_dec_noisy = add_relative_noise(
        X_test_dec,
        cols=weather_cols_dec_test,
        max_rel_err=float(cfg.get("weather_relative_error", 0.05)),
        seed=int(cfg["seed"]) + 2,
    )
    X_test_dec_scaled = scaler_X_dec.transform(X_test_dec_noisy.values)

    y_for_test_scaled = scaler_y.transform(df_for_test[[target_col]].values).reshape(-1)

    w_test = make_windows(
        y=y_for_test_scaled,
        enc_features=X_test_enc_scaled,
        dec_features=X_test_dec_scaled,
        index=df_for_test.index,
        L=L,
        H=H,
        stride=stride,
    )

    ds_train = to_tf_dataset(
        w_train,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        seed=int(cfg["seed"]),
    )
    ds_val = (
        to_tf_dataset(
            w_val,
            batch_size=int(cfg["batch_size"]),
            shuffle=False,
            seed=int(cfg["seed"]),
        )
        if w_val else None
    )

    config = AutoConfig.for_model(model_name)
    model = AutoModel.from_config(config, predict_sequence_length=H)
    trainer = KerasTrainer(model)

    print("\n=============================")
    print(f"Model: {model_name}")
    print(f"L={L}, H={H}, stride={stride}")
    print(f"Rows: train={len(df_train)}, val={len(df_val) if df_val is not None else 0}, test={len(df_test)}")
    print(f"Windows: train={len(w_train.y)}, val={len(w_val.y) if w_val else 0}, test={len(w_test.y)}")
    print(f"Train batch size: {int(cfg['batch_size'])}")
    print(f"Initial predict batch size: {int(cfg.get('predict_batch_size', 16))}")
    print(f"Future excluded cols: {future_excluded_cols}")
    print(f"Weather cols with noise in dec_f: {weather_cols_all}")
    print(f"Weather relative error: {float(cfg.get('weather_relative_error', 0.05)):.2%}")
    print("=============================")

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

    x_test = (w_test.x, w_test.enc_f, w_test.dec_f)
    y_pred_scaled = np.asarray(
        predict_in_batches(
            trainer,
            x_test,
            batch_size=int(cfg.get("predict_batch_size", 16)),
            min_batch_size=int(cfg.get("min_predict_batch_size", 1)),
            force_cpu_on_oom=bool(cfg.get("force_cpu_predict_on_oom", True)),
        ),
        dtype=np.float32,
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

    if bool(cfg.get("save_trainer_model", True)):
        model_dir = os.path.join(cfg["output_dir"], f"model_{model_name}_L{L}_H{H}")
        safe_save_trainer_model(trainer, model_dir)

    if bool(cfg.get("save_savedmodel", True)):
        keras_dir = os.path.join(cfg["output_dir"], f"keras_savedmodel_{model_name}_L{L}_H{H}")
        safe_export_savedmodel(trainer.model, keras_dir)

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

    patch_tfts_decoder_for_savedmodel()

    df, target_col = load_prepare(DATA_CSV)
    models = MODELS if isinstance(MODELS, list) else [MODELS]
    for m in models:
        run_one_model(str(m), df, target_col, CONFIG)


if __name__ == "__main__":
    main()
