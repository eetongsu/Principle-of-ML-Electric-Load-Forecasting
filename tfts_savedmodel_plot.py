#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
读取已保存的 Keras SavedModel（目录形式）TFTS 模型，选择典型日进行预测，并画图保存。
（修复：DatetimeIndex 存在重复时间戳时 get_loc 返回 slice 导致的 TypeError）

依赖安装（建议同训练环境）
  pip install tensorflow==2.10.* tfts pandas numpy scikit-learn matplotlib

运行：
  python forecast_tfts_savedmodel_plot_fix_attention_v2.py

只需修改两处路径：
- DATA_CSV
- MODEL_PATH（SavedModel 目录，非 .h5）
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf


# =============================
# 1) 参数（脚本内全部给定）
# =============================

DATA_CSV = r"load_forecasting_dataset_corrected.csv"
MODEL_PATH = r"outputs\keras_savedmodel_informer_L168_H24"

CFG: Dict[str, Any] = dict(
    seed=42,
    train_length=168,   # L
    pred_length=24,     # H
    test_size=0.2,
    typical_days=3,
    pick_from="test",
    day_starts_at_hour=0,
    timezone=None,
    output_dir="outputs_typical_days",
    dpi=160,
    enable_memory_growth=True,
)


# =============================
# 2) 工具函数
# =============================

def set_seed(seed: int = 42) -> None:
    tf.keras.utils.set_random_seed(seed)
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

def load_prepare(csv_path: str, timezone: Optional[str] = None) -> Tuple[pd.DataFrame, str]:
    df = pd.read_csv(csv_path)

    ts_col = _find_col(df, ["Timestamp", "timestamp", "Date", "Datetime"])
    hour_col = _find_col(df, ["Hour", "Hour of D", "Hour of Day", "hour"])
    target_col = _find_col(df, ["Load Demand (kW)", "Load Demand", "LoadDemand", "Load (kW)", "Load"])

    if ts_col is None or target_col is None:
        raise ValueError(f"缺少必要列。当前列：{list(df.columns)}；需要 Timestamp/Date + Load")

    dt = pd.to_datetime(df[ts_col], errors="coerce")
    if dt.isna().any():
        bad = int(dt.isna().sum())
        raise ValueError(f"时间戳列 '{ts_col}' 有 {bad} 行无法解析。")

    if hour_col is not None:
        hour = pd.to_numeric(df[hour_col], errors="coerce").fillna(0).astype(int).clip(0, 23)
        dt = dt + pd.to_timedelta(hour, unit="h")

    if timezone:
        try:
            dt = dt.dt.tz_localize(timezone, nonexistent="shift_forward", ambiguous="infer")
        except Exception:
            try:
                dt = dt.dt.tz_convert(timezone)
            except Exception:
                pass

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

@dataclass
class SingleWindow:
    x: np.ndarray
    enc_f: np.ndarray
    dec_f: np.ndarray
    y_true: np.ndarray
    origin: pd.Timestamp

def make_single_window(
    y: np.ndarray,
    features: np.ndarray,
    index: pd.DatetimeIndex,
    origin_pos: int,
    L: int,
    H: int
) -> SingleWindow:
    start = origin_pos - L
    if start < 0 or origin_pos + H > len(index):
        raise IndexError("窗口越界：检查 L/H 或 origin_pos")

    x = y[start : origin_pos].astype(np.float32).reshape(1, L, 1)
    enc_f = features[start : origin_pos, :].astype(np.float32).reshape(1, L, -1)
    dec_f = features[origin_pos : origin_pos + H, :].astype(np.float32).reshape(1, H, -1)
    y_true = y[origin_pos : origin_pos + H].astype(np.float32).reshape(1, H, 1)
    origin = index[origin_pos]
    return SingleWindow(x=x, enc_f=enc_f, dec_f=dec_f, y_true=y_true, origin=origin)

def pick_typical_day_starts(
    df: pd.DataFrame,
    target_col: str,
    day_start_hour: int,
    how_many: int,
) -> List[pd.Timestamp]:
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

def build_tfts_custom_objects() -> Dict[str, object]:
    custom: Dict[str, object] = {}
    try:
        import tfts  # noqa: F401
    except Exception as e:
        print("⚠️ 未能 import tfts：", e)
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
        print("✅ Registered custom object: Attention ->", custom["Attention"])
    else:
        print("⚠️ 没有在 tfts 中找到 Attention 类。")

    return custom

def load_model_savedmodel(model_dir: str) -> tf.keras.Model:
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"MODEL_PATH 不存在：{model_dir}")
    custom_objects = build_tfts_custom_objects()
    with tf.keras.utils.custom_object_scope(custom_objects):
        return tf.keras.models.load_model(model_dir, compile=False)

def locate_origin_pos(index: pd.DatetimeIndex, ts: pd.Timestamp) -> int:
    """
    兼容重复时间戳：
    - DatetimeIndex.get_loc(ts) 可能返回 int / slice / bool array
    我们统一取“第一次出现”的位置。
    """
    if ts in index:
        loc = index.get_loc(ts)
        if isinstance(loc, slice):
            return int(loc.start)
        if isinstance(loc, (np.ndarray, list)):
            # bool mask or array of positions
            arr = np.asarray(loc)
            if arr.dtype == bool:
                pos = np.flatnonzero(arr)
                return int(pos[0])
            return int(arr[0])
        return int(loc)
    # 不存在则取第一个 >= ts 的位置
    return int(index.searchsorted(ts))


# =============================
# 3) 主流程
# =============================

def main() -> None:
    os.makedirs(CFG["output_dir"], exist_ok=True)
    set_seed(int(CFG["seed"]))
    configure_tf(bool(CFG.get("enable_memory_growth", True)))

    L = int(CFG["train_length"])
    H = int(CFG["pred_length"])

    if not os.path.exists(DATA_CSV):
        raise FileNotFoundError(f"DATA_CSV 不存在：{DATA_CSV}")

    df, target_col = load_prepare(DATA_CSV, timezone=CFG.get("timezone"))

    # 如果有重复时间戳，这里提示一下（不强制去重，避免影响与训练一致性）
    dup = int(df.index.duplicated().sum())
    if dup > 0:
        print(f"⚠️ 注意：发现重复时间戳 {dup} 条。脚本将对典型日起点取第一次出现的位置。")

    df_train, df_test = split_chrono(df, test_size=float(CFG["test_size"]))
    df_pick = df_test if str(CFG.get("pick_from", "test")).lower() == "test" else df

    X_train = df_train.drop(columns=[target_col])
    y_train = df_train[[target_col]]
    scaler_X = StandardScaler().fit(X_train.values)
    scaler_y = StandardScaler().fit(y_train.values)

    X_all_scaled = scaler_X.transform(df.drop(columns=[target_col]).values)
    y_all_scaled = scaler_y.transform(df[[target_col]].values).reshape(-1)

    model = load_model_savedmodel(MODEL_PATH)
    print(f"✅ Loaded model: {MODEL_PATH}")
    try:
        model.summary()
    except Exception:
        pass

    typical_starts = pick_typical_day_starts(
        df_pick,
        target_col=target_col,
        day_start_hour=int(CFG["day_starts_at_hour"]),
        how_many=int(CFG["typical_days"]),
    )
    if not typical_starts:
        raise RuntimeError("无法挑选典型日：请检查数据时间索引是否正确。")

    all_index = df.index
    out_frames: List[pd.DataFrame] = []

    import matplotlib.pyplot as plt

    for day_start in typical_starts:
        origin_pos = locate_origin_pos(all_index, pd.Timestamp(day_start))

        if origin_pos - L < 0 or origin_pos + H > len(all_index):
            print(f"⚠️ Skip {day_start}: 不满足窗口需求（需要 L={L}, H={H}）")
            continue

        w = make_single_window(
            y=y_all_scaled,
            features=X_all_scaled,
            index=all_index,
            origin_pos=origin_pos,
            L=L,
            H=H,
        )

        y_pred_scaled = model((w.x, w.enc_f, w.dec_f), training=False).numpy()
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
        print(f"✅ Saved plot: {plot_path}")

    if not out_frames:
        raise RuntimeError("没有生成任何典型日图：可能 test 段太短或 L/H 设置不匹配。")

    out_all = pd.concat(out_frames, axis=0, ignore_index=True)
    out_csv = os.path.join(CFG["output_dir"], "typical_day_forecasts.csv")
    out_all.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"✅ Saved forecasts CSV: {out_csv}")
    print("Done.")


if __name__ == "__main__":
    main()
