#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot typical-day forecasts from a saved TFTS model.

This script is aligned with the training pipeline:
1) Encoder features use all non-target columns.
2) Decoder future features exclude:
   - GDP (LKR)
   - Per Capita Energy Use (kWh)
   - Electricity Price (LKR/kWh)
3) It supports both:
   - Keras model format via tf.keras.models.load_model(...)
   - Plain TensorFlow SavedModel via tf.saved_model.load(...)
4) It handles duplicate timestamps by taking the first matching occurrence.

Edit:
- DATA_CSV
- MODEL_PATH
- CFG
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

# These environment variables must be set before importing TensorFlow.
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf


# =============================
# 1) USER SETTINGS
# =============================

DATA_CSV = r"load_forecasting_dataset_corrected.csv"
MODEL_PATH = r"outputs\keras_savedmodel_informer_L168_H24"

CFG: Dict[str, Any] = dict(
    seed=42,
    train_length=168,          # L
    pred_length=24,            # H
    test_size=0.2,

    typical_days=3,            # Number of typical days to plot
    pick_from="test",          # "test" or "all"
    day_starts_at_hour=0,      # Typical day origin hour

    timezone=None,
    output_dir="outputs_typical_days",
    dpi=160,

    enable_memory_growth=True,
    force_cpu_predict=False,   # Set True if GPU inference causes issues
)


# =============================
# 2) GENERAL HELPERS
# =============================

def set_seed(seed: int = 42) -> None:
    tf.keras.utils.set_random_seed(seed)
    np.random.seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def configure_tf(enable_memory_growth: bool = True) -> None:
    if not enable_memory_growth:
        return
    try:
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
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


# =============================
# 3) FEATURE ENGINEERING
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


def get_future_excluded_cols(df: pd.DataFrame) -> List[str]:
    """Identify columns excluded from decoder future features."""
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


def load_prepare(csv_path: str, timezone: Optional[str] = None) -> Tuple[pd.DataFrame, str]:
    df = pd.read_csv(csv_path)

    ts_col = _find_col(df, ["Timestamp", "timestamp", "Date", "Datetime"])
    target_col = _find_col(df, ["Load Demand (kW)", "Load Demand", "LoadDemand", "Load (kW)", "Load"])

    if ts_col is None or target_col is None:
        raise ValueError(
            f"Missing required columns. Current columns: {list(df.columns)}; "
            f"need a timestamp column and target column."
        )

    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    if df[ts_col].isna().any():
        bad = int(df[ts_col].isna().sum())
        raise ValueError(f"Timestamp column '{ts_col}' has {bad} unparsable rows.")

    df = df.set_index(ts_col).sort_index()

    if timezone:
        try:
            df.index = df.index.tz_localize(timezone, nonexistent="shift_forward", ambiguous="infer")
        except Exception:
            try:
                df.index = df.index.tz_convert(timezone)
            except Exception:
                pass

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
    test = df.iloc[n - n_test:].copy()
    return train, test


# =============================
# 4) WINDOW HELPERS
# =============================

@dataclass
class SingleWindow:
    x: np.ndarray
    enc_f: np.ndarray
    dec_f: np.ndarray
    y_true: np.ndarray
    origin: pd.Timestamp


def make_single_window(
    y: np.ndarray,
    enc_features: np.ndarray,
    dec_features: np.ndarray,
    index: pd.DatetimeIndex,
    origin_pos: int,
    L: int,
    H: int,
) -> SingleWindow:
    start = origin_pos - L
    if start < 0 or origin_pos + H > len(index):
        raise IndexError("Window out of bounds: check L/H or origin_pos")

    x = y[start:origin_pos].astype(np.float32).reshape(1, L, 1)
    enc_f = enc_features[start:origin_pos, :].astype(np.float32).reshape(1, L, -1)
    dec_f = dec_features[origin_pos:origin_pos + H, :].astype(np.float32).reshape(1, H, -1)
    y_true = y[origin_pos:origin_pos + H].astype(np.float32).reshape(1, H, 1)
    origin = index[origin_pos]
    return SingleWindow(x=x, enc_f=enc_f, dec_f=dec_f, y_true=y_true, origin=origin)


def locate_origin_pos(index: pd.DatetimeIndex, ts: pd.Timestamp) -> int:
    """
    Handle duplicate timestamps robustly.
    Always use the first occurrence.
    """
    if ts in index:
        loc = index.get_loc(ts)
        if isinstance(loc, slice):
            return int(loc.start)
        if isinstance(loc, (np.ndarray, list)):
            arr = np.asarray(loc)
            if arr.dtype == bool:
                pos = np.flatnonzero(arr)
                return int(pos[0])
            return int(arr[0])
        return int(loc)
    return int(index.searchsorted(ts))


def pick_typical_day_starts(
    df: pd.DataFrame,
    target_col: str,
    day_start_hour: int,
    how_many: int,
) -> List[pd.Timestamp]:
    """
    Pick representative days based on daily peak load:
    - highest
    - lowest
    - median
    """
    idx = df.index
    shifted_day = (idx - pd.to_timedelta(day_start_hour, unit="h")).normalize()
    daily_max = df[target_col].groupby(shifted_day).max()

    if len(daily_max) == 0:
        return []

    if len(daily_max) < how_many:
        days = list(daily_max.index)
        return [pd.Timestamp(d) + pd.Timedelta(hours=day_start_hour) for d in days]

    hi = daily_max.idxmax()
    lo = daily_max.idxmin()
    med_val = daily_max.sort_values().iloc[len(daily_max) // 2]
    med_day = daily_max[daily_max == med_val].index[0]

    candidates: List[pd.Timestamp] = []
    for d in [hi, lo, med_day]:
        if d not in candidates:
            candidates.append(d)

    if len(candidates) < how_many:
        for d in daily_max.index:
            if d not in candidates:
                candidates.append(d)
            if len(candidates) >= how_many:
                break

    return [pd.Timestamp(d) + pd.Timedelta(hours=day_start_hour) for d in candidates[:how_many]]


def infer_horizon_index(all_index: pd.DatetimeIndex, start_ts: pd.Timestamp, H: int) -> pd.DatetimeIndex:
    freq = pd.infer_freq(all_index)
    if freq is not None:
        return pd.date_range(start=start_ts, periods=H, freq=freq)

    if len(all_index) >= 2:
        deltas = np.diff(all_index.view("int64"))
        step_ns = int(np.median(deltas))
    else:
        step_ns = int(3600 * 1e9)

    return pd.date_range(start=start_ts, periods=H, freq=pd.to_timedelta(step_ns, unit="ns"))


# =============================
# 5) MODEL LOADING
# =============================

def build_tfts_custom_objects() -> Dict[str, object]:
    custom: Dict[str, object] = {}
    try:
        import tfts  # noqa: F401
    except Exception as e:
        print("Warning: failed to import tfts:", e)
        return custom

    candidates = [
        ("tfts.layers.attention_layer", ["Attention", "FullAttention", "ProbAttention", "FastAttention"]),
        ("tfts.layers.attention", ["Attention", "FullAttention", "ProbAttention", "FastAttention"]),
        ("tfts.layers", ["Attention", "FullAttention", "ProbAttention", "FastAttention"]),
    ]

    for module_name, names in candidates:
        try:
            mod = __import__(module_name, fromlist=["*"])
        except Exception:
            continue
        for n in names:
            if hasattr(mod, n):
                custom[n] = getattr(mod, n)

    if "Attention" in custom:
        print("Registered custom object: Attention ->", custom["Attention"])
    else:
        print("Warning: Attention class not found in tfts.")

    return custom


class LoadedForecastModel:
    """
    Unified wrapper for:
    - Keras model loaded by tf.keras.models.load_model
    - Plain TensorFlow SavedModel loaded by tf.saved_model.load
    """

    def __init__(self, backend: str, obj: object):
        self.backend = backend
        self.obj = obj
        self.signature = None
        self.signature_keys: List[str] = []

        if backend == "tf_saved_model":
            sigs = getattr(obj, "signatures", None) or {}
            if "serving_default" in sigs:
                self.signature = sigs["serving_default"]
                try:
                    _, kw = self.signature.structured_input_signature
                    self.signature_keys = list(kw.keys())
                except Exception:
                    self.signature_keys = []

    def summary(self) -> None:
        if self.backend == "keras":
            try:
                self.obj.summary()
            except Exception:
                pass
        else:
            print("Loaded plain TensorFlow SavedModel (non-Keras).")
            if self.signature is not None:
                try:
                    print("Serving signature inputs:", self.signature.structured_input_signature)
                    print("Serving signature outputs:", self.signature.structured_outputs)
                except Exception:
                    pass

    def predict(self, x: np.ndarray, enc_f: np.ndarray, dec_f: np.ndarray, force_cpu: bool = False) -> np.ndarray:
        if self.backend == "keras":
            if force_cpu:
                with tf.device("/CPU:0"):
                    y = self.obj((x, enc_f, dec_f), training=False)
            else:
                y = self.obj((x, enc_f, dec_f), training=False)
            return y.numpy() if hasattr(y, "numpy") else np.asarray(y)

        if self.signature is None:
            raise RuntimeError("SavedModel has no serving_default signature.")

        # Default expected order from training/export:
        # input_0 -> x
        # input_1 -> enc_f
        # input_2 -> dec_f
        tensors = {
            "input_0": tf.convert_to_tensor(x, dtype=tf.float32),
            "input_1": tf.convert_to_tensor(enc_f, dtype=tf.float32),
            "input_2": tf.convert_to_tensor(dec_f, dtype=tf.float32),
        }

        # If the exact standard names are not present, map by discovered key order.
        if self.signature_keys and set(["input_0", "input_1", "input_2"]).issubset(set(self.signature_keys)):
            named_inputs = {
                "input_0": tensors["input_0"],
                "input_1": tensors["input_1"],
                "input_2": tensors["input_2"],
            }
        elif len(self.signature_keys) == 3:
            named_inputs = {
                self.signature_keys[0]: tensors["input_0"],
                self.signature_keys[1]: tensors["input_1"],
                self.signature_keys[2]: tensors["input_2"],
            }
        else:
            raise RuntimeError(
                "Unsupported SavedModel signature. Expected 3 inputs for (x, enc_f, dec_f), "
                f"but got keys: {self.signature_keys}"
            )

        if force_cpu:
            with tf.device("/CPU:0"):
                out = self.signature(**named_inputs)
        else:
            out = self.signature(**named_inputs)

        if isinstance(out, dict):
            first = next(iter(out.values()))
            return first.numpy()
        return out.numpy() if hasattr(out, "numpy") else np.asarray(out)


def load_forecast_model(model_dir: str) -> LoadedForecastModel:
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"MODEL_PATH does not exist: {model_dir}")

    custom_objects = build_tfts_custom_objects()

    try:
        with tf.keras.utils.custom_object_scope(custom_objects):
            model = tf.keras.models.load_model(model_dir, compile=False)
        print(f"Loaded as Keras model: {model_dir}")
        return LoadedForecastModel("keras", model)
    except Exception as e:
        print("Keras load_model failed, falling back to tf.saved_model.load(...)")
        print("Reason:", e)

    obj = tf.saved_model.load(model_dir)
    print(f"Loaded as plain TensorFlow SavedModel: {model_dir}")
    return LoadedForecastModel("tf_saved_model", obj)


# =============================
# 6) MAIN
# =============================

def main() -> None:
    os.makedirs(CFG["output_dir"], exist_ok=True)
    set_seed(int(CFG["seed"]))
    configure_tf(bool(CFG.get("enable_memory_growth", True)))

    L = int(CFG["train_length"])
    H = int(CFG["pred_length"])

    if not os.path.exists(DATA_CSV):
        raise FileNotFoundError(f"DATA_CSV does not exist: {DATA_CSV}")

    df, target_col = load_prepare(DATA_CSV, timezone=CFG.get("timezone"))

    dup = int(df.index.duplicated().sum())
    if dup > 0:
        print(f"Warning: found {dup} duplicate timestamps. The script will use the first occurrence for each typical-day origin.")

    df_train, df_test = split_chrono(df, test_size=float(CFG["test_size"]))
    df_pick = df_test if str(CFG.get("pick_from", "test")).lower() == "test" else df

    # Rebuild features exactly as in training
    future_excluded_cols = get_future_excluded_cols(df)

    X_train_enc = df_train.drop(columns=[target_col]).copy()
    X_train_dec = X_train_enc.drop(columns=future_excluded_cols, errors="ignore").copy()
    y_train = df_train[[target_col]].copy()

    scaler_X_enc = StandardScaler().fit(X_train_enc.values)
    scaler_X_dec = StandardScaler().fit(X_train_dec.values)
    scaler_y = StandardScaler().fit(y_train.values)

    X_all_enc = df.drop(columns=[target_col]).copy()
    X_all_dec = X_all_enc.drop(columns=future_excluded_cols, errors="ignore").copy()

    X_all_enc_scaled = scaler_X_enc.transform(X_all_enc.values)
    X_all_dec_scaled = scaler_X_dec.transform(X_all_dec.values)
    y_all_scaled = scaler_y.transform(df[[target_col]].values).reshape(-1)

    model = load_forecast_model(MODEL_PATH)
    model.summary()

    print(f"Encoder feature dim: {X_all_enc_scaled.shape[1]}")
    print(f"Decoder feature dim: {X_all_dec_scaled.shape[1]}")
    print(f"Excluded decoder future cols: {future_excluded_cols}")

    typical_starts = pick_typical_day_starts(
        df_pick,
        target_col=target_col,
        day_start_hour=int(CFG["day_starts_at_hour"]),
        how_many=int(CFG["typical_days"]),
    )
    if not typical_starts:
        raise RuntimeError("Failed to select typical days. Please check the time index.")

    all_index = df.index
    out_frames: List[pd.DataFrame] = []

    import matplotlib.pyplot as plt

    for day_start in typical_starts:
        origin_pos = locate_origin_pos(all_index, pd.Timestamp(day_start))

        if origin_pos - L < 0 or origin_pos + H > len(all_index):
            print(f"Skip {day_start}: insufficient window length for L={L}, H={H}")
            continue

        w = make_single_window(
            y=y_all_scaled,
            enc_features=X_all_enc_scaled,
            dec_features=X_all_dec_scaled,
            index=all_index,
            origin_pos=origin_pos,
            L=L,
            H=H,
        )

        y_pred_scaled = model.predict(
            w.x,
            w.enc_f,
            w.dec_f,
            force_cpu=bool(CFG.get("force_cpu_predict", False)),
        )

        if y_pred_scaled.ndim == 2:
            y_pred_scaled = y_pred_scaled[..., None]
        elif y_pred_scaled.ndim == 1:
            y_pred_scaled = y_pred_scaled.reshape(1, -1, 1)

        y_true = scaler_y.inverse_transform(w.y_true.reshape(-1, 1)).reshape(w.y_true.shape)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).reshape(y_pred_scaled.shape)

        horizon_idx = infer_horizon_index(all_index, start_ts=w.origin, H=H)

        df_one = pd.DataFrame({
            "timestamp": horizon_idx,
            "y_true": y_true[0, :, 0],
            "y_pred": y_pred[0, :, 0],
            "origin": w.origin,
        })
        out_frames.append(df_one)

        plt.figure(figsize=(12, 5))
        plt.plot(df_one["timestamp"], df_one["y_true"], label="Actual")
        plt.plot(df_one["timestamp"], df_one["y_pred"], label="Predicted")
        plt.title(f"Typical Day Forecast (origin={w.origin})")
        plt.xlabel("Timestamp")
        plt.ylabel(target_col)
        plt.legend()
        plt.tight_layout()

        fname = f"typical_day_{pd.Timestamp(w.origin).strftime('%Y%m%d_%H%M')}.png"
        plot_path = os.path.join(CFG["output_dir"], fname)
        plt.savefig(plot_path, dpi=int(CFG["dpi"]), bbox_inches="tight")
        plt.close()
        print(f"Saved plot: {plot_path}")

    if not out_frames:
        raise RuntimeError(
            "No typical-day plots were generated. "
            "The selected segment may be too short, or L/H may not match the dataset."
        )

    out_all = pd.concat(out_frames, axis=0, ignore_index=True)
    out_csv = os.path.join(CFG["output_dir"], "typical_day_forecasts.csv")
    out_all.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"Saved forecasts CSV: {out_csv}")
    print("Done.")


if __name__ == "__main__":
    main()