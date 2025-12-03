# normalize_manual.py  (rev)
from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Dict, Any, Iterable, Tuple

__all__ = [
    "FEATURE_ORDER",
    "MANUAL_NORM_CONF",
    "SKIP_NORM",
    "SHARE_FEATS",
    "normalize_matrix_file",
    "normalize_directory",
]

FEATURE_ORDER: Tuple[str, ...] = (
    "dt", "dp", "dmx", "dmm", "dmt", "dmv", "dvwap", "ddayo", "d5m", "d30m",
    "rng", "efrang", "pctx0", "pctx1", "pctx2", "pctx3", "pctx4",
    "b_vol", "s_vol", "t_vol", "d_vol", "avg_vol",
    "g_b", "g_s", "g_n", "rate_avg", "max_streak_buy", "max_streak_sell",
    "ema_vol_total", "ema_order_rate", "d_ticks_max",
    "vol_per_tick_up_a_mean", "vol_per_tick_up_a_max", "vol_per_tick_up_a_sum",
    "vol_per_tick_dn_a_mean", "vol_per_tick_dn_a_max", "vol_per_tick_dn_a_sum",
    "vol_per_tick_up_b_mean", "vol_per_tick_up_b_max", "vol_per_tick_up_b_sum",
    "vol_per_tick_dn_b_mean", "vol_per_tick_dn_b_max", "vol_per_tick_dn_b_sum",
    "buy_share_wdo", "sell_share_wdo",
    "ease_of_move_a", "ease_of_move_b",
    "absorption_buy_a", "absorption_sell_a",
    "absorption_buy_b", "absorption_sell_b",
    "topB_buy_vol_a", "topB_sell_vol_a",
    "topS_buy_vol_a", "topS_sell_vol_a",
    "topB_buy_vol_b", "topB_sell_vol_b",
    "topS_buy_vol_b", "topS_sell_vol_b",
)

MANUAL_NORM_CONF: Dict[str, Dict[str, Any]] = {

    "dt":   {"enabled": True,  "low":    3, "high":  40,  "gamma": 1,  "invert": False, "log":  True, "log_scale": 0.1, "skew": -1.0,  "softness": 1},
    "dp":   {"enabled": True,  "low":   -2, "high":   2,  "gamma": 1,  "invert": False, "log": False, "log_scale": 0.1, "skew":  0.0,  "softness": 1},
    "dmx":  {"enabled": True,  "low": -2.0, "high": 0.0,  "gamma": 1,  "invert": False, "log": False, "log_scale": 0.1, "skew":  0.0,  "softness": 1},
    "dmm":  {"enabled": True,  "low":  0.0, "high": 2.0,  "gamma": 1,  "invert": False, "log": False, "log_scale": 0.1, "skew":  0.0,  "softness": 1},
    "dmt":  {"enabled": True,  "low": -2.0, "high": 2.0,  "gamma": 1,  "invert": False, "log": False, "log_scale": 0.1, "skew":  0.0,  "softness": 1},
    "dmv":  {"enabled": True,  "low": -2.0, "high": 2.0,  "gamma": 1,  "invert": False, "log": False, "log_scale": 0.1, "skew":  0.0,  "softness": 1},
    "dvwap":  {"enabled": True,  "low": -20.0, "high": 20.0,  "gamma": 1,  "invert": False, "log": False,  "log_scale": 0.1, "skew":  0.0,  "softness": 1},
    "ddayo":  {"enabled": True,  "low": -30.0, "high": 30.0,  "gamma": 1,  "invert": False, "log": False,  "log_scale": 0.1, "skew":  0.0,  "softness": 0.5},
    "d5m":    {"enabled": True,  "low":  -5.0, "high":  5.0,  "gamma": 1,  "invert": False, "log": False,  "log_scale": 0.1, "skew":  0.0,  "softness": 1},
    "d30m":   {"enabled": True,  "low": -10.0, "high": 10.0,  "gamma": 1,  "invert": False, "log": False,  "log_scale": 0.1, "skew":  0.0,  "softness": 1},
    "rng":    {"enabled": True,  "low":   0.5,  "high": 3.0,  "gamma": 1, "invert": False, "log": False,  "log_scale": 0.1, "skew":  0.0,  "softness": 1},
    "efrang": {"enabled": True,  "low":   0.0,  "high": 1.0,  "gamma": 1, "invert": False, "log": False,  "log_scale": 0.1, "skew":  0.0,  "softness": 1},
    "b_vol":   {"enabled": True, "low": 0,     "high": 3000, "gamma": 0.7, "invert": False, "log": True, "log_scale": 20, "skew":  100.0,  "softness": 2},
    "s_vol":   {"enabled": True, "low": 0,     "high": 3000, "gamma": 0.7, "invert": False, "log": True, "log_scale": 20, "skew":  100.0,  "softness": 2},
    "t_vol":   {"enabled": True, "low": 1000,  "high": 10000, "gamma": 1.0, "invert": False, "log": True, "log_scale": 0.1, "skew":  0.0,  "softness": 1},
    "d_vol":   {"enabled": True, "low": -5000, "high":  5000, "gamma": 0.7, "invert": False, "log": False, "log_scale": 0.1, "skew":  0.0,  "softness": 1},
    "avg_vol": {"enabled": True, "low": 5,     "high":    95, "gamma": 0.7, "invert": False, "log":  True, "log_scale": 0.1, "skew":  0.0,  "softness": 1},
    "g_b": {"enabled": True, "low": 5, "high": 50, "gamma": 1.0, "invert": False, "log": True, "log_scale": 0.1, "skew":  0.0,  "softness": 1},
    "g_s": {"enabled": True, "low": 5, "high": 50, "gamma": 1.0, "invert": False, "log": True, "log_scale": 0.1, "skew":  0.0,  "softness": 1},
    "g_n": {"enabled": True, "low": 2, "high": 30, "gamma": 1.0, "invert": False, "log": True, "log_scale": 0.1, "skew":  0.0,  "softness": 1},
    "rate_avg": {"enabled": True, "low": 5, "high": 95, "gamma": 1.0, "invert": False, "log": True, "log_scale": 0.1, "skew":  0.0,  "softness": 1.5},
    "max_streak_buy": {"enabled": True, "low": 0, "high": 10, "gamma": 1.0, "invert": False, "log": False, "log_scale": 0.1, "skew":  0.0,  "softness": 1},
    "max_streak_sell": {"enabled": True, "low": 0, "high": 10, "gamma": 1.0, "invert": False, "log": False, "log_scale": 0.1, "skew":  0.0,  "softness": 1},
    "ema_vol_total": {"enabled": True, "low": 200, "high": 800, "gamma": 1.0, "invert": False, "log": True, "log_scale": 1, "skew":  1.0,  "softness": 1},
    "ema_order_rate": {"enabled": True, "low": 5, "high": 66, "gamma": 1, "invert": False, "log": True, "log_scale": 1, "skew":  0.0,  "softness": 2},
    "d_ticks_max": {"enabled": True, "low": 0, "high": 4, "gamma": 1.0, "invert": False, "log": True, "log_scale": 1, "skew":  0.0,  "softness": 1},
    "vol_per_tick_up_a_mean": {"enabled": True, "low": 0, "high": 200, "gamma": 1.0, "invert": False, "log": True, "log_scale": 1, "skew":  0.0,  "softness": 1},
    "vol_per_tick_up_a_max": {"enabled": True, "low": 0, "high": 200, "gamma": 1.0, "invert": False, "log": True, "log_scale": 1, "skew":  0.0,  "softness": 1},
    "vol_per_tick_up_a_sum": {"enabled": True, "low": 0, "high": 400, "gamma": 1.0, "invert": False, "log": True, "log_scale": 1, "skew":  0.0,  "softness": 1},
    "vol_per_tick_dn_a_mean": {"enabled": True, "low": 0, "high": 200, "gamma": 1.0, "invert": False, "log": True, "log_scale": 1, "skew":  0.0,  "softness": 1},
    "vol_per_tick_dn_a_max": {"enabled": True, "low": 0, "high": 200, "gamma": 1.0, "invert": False, "log": True, "log_scale": 1, "skew":  0.0,  "softness": 1},
    "vol_per_tick_dn_a_sum": {"enabled": True, "low": 0, "high": 400, "gamma": 1.0, "invert": False, "log": True, "log_scale": 1, "skew":  0.0,  "softness": 1},
    "vol_per_tick_up_b_mean": {"enabled": True, "low": 0, "high": 2000, "gamma": 1.0, "invert": False, "log": True, "log_scale": 10, "skew":  0.0,  "softness": 1},
    "vol_per_tick_up_b_max": {"enabled": True, "low": 0, "high": 2000, "gamma": 1.0, "invert": False, "log": True, "log_scale": 10, "skew":  0.0,  "softness": 1},
    "vol_per_tick_up_b_sum": {"enabled": True, "low": 0, "high": 4000, "gamma": 1.0, "invert": False, "log": True, "log_scale": 10, "skew":  0.0,  "softness": 1},
    "vol_per_tick_dn_b_mean": {"enabled": True, "low": 0, "high": 2000, "gamma": 1.0, "invert": False, "log": True, "log_scale": 10, "skew":  0.0,  "softness": 1},
    "vol_per_tick_dn_b_max": {"enabled": True, "low": 0, "high": 2000, "gamma": 1.0, "invert": False, "log": True, "log_scale": 10, "skew":  0.0,  "softness": 1},
    "vol_per_tick_dn_b_sum": {"enabled": True, "low": 0, "high": 4000, "gamma": 1.0, "invert": False, "log": True, "log_scale": 10, "skew":  0.0,  "softness": 1},
    "buy_share_wdo": {"enabled": True, "low": 0, "high": 1, "gamma": 1, "invert": False, "log": True, "log_scale": 0.02, "skew":  0.0,  "softness": 2},
    "sell_share_wdo":{"enabled": True, "low": 0, "high": 1, "gamma": 1, "invert": False, "log": True, "log_scale": 0.02, "skew":  0.0,  "softness": 2},
    "ease_of_move_a": {"enabled": True, "low": -500, "high": 500,  "gamma": 1.0, "invert": False, "log": False, "log_scale": 0.1, "skew":  0.0,  "softness": 2},
    "ease_of_move_b": {"enabled": True, "low": -3000, "high": 3000, "gamma": 1.0, "invert": False, "log": False, "log_scale": 0.1, "skew":  0.0,  "softness": 2},
    "absorption_buy_a": {"enabled": True, "low": 100, "high": 500, "gamma": 1.0, "invert": False, "log": False, "log_scale": 1, "skew":  -10.0,  "softness": 1},
    "absorption_sell_a": {"enabled": True, "low": 100, "high": 500, "gamma": 1.0, "invert": False, "log": False, "log_scale": 1, "skew":  -10.0,  "softness": 1},
    "absorption_buy_b": {"enabled": True, "low": 0, "high": 3000, "gamma": 1.0, "invert": False, "log": False, "log_scale": 0.1, "skew":  0.0,  "softness": 1},
    "absorption_sell_b": {"enabled": True, "low": 0, "high": 3000, "gamma": 1.0, "invert": False, "log": False, "log_scale": 0.1, "skew":  0.0,  "softness": 1},
    "topB_buy_vol_a": {"enabled": True, "low": 0, "high": 800, "gamma": 1.0, "invert": False, "log": True, "log_scale": 10, "skew":  0.0,  "softness": 1.5},
    "topB_sell_vol_a":{"enabled": True, "low": 0, "high": 800, "gamma": 1.0, "invert": False, "log": True, "log_scale": 10, "skew":  0.0,  "softness": 1.5},
    "topS_buy_vol_a": {"enabled": True, "low": 0, "high": 800, "gamma": 1.0, "invert": False, "log": True, "log_scale": 10, "skew":  0.0,  "softness": 1.5},
    "topS_sell_vol_a":{"enabled": True, "low": 0, "high": 800, "gamma": 1.0, "invert": False, "log": True, "log_scale": 10, "skew":  0.0,  "softness": 1.5},
    "topB_buy_vol_b":  {"enabled": True, "low": 0, "high": 3000, "gamma": 1.0, "invert": False, "log": True, "log_scale": 100, "skew":  0.0,  "softness": 1.5},
    "topB_sell_vol_b": {"enabled": True, "low": 0, "high": 3000, "gamma": 1.0, "invert": False, "log": True, "log_scale": 100, "skew":  0.0,  "softness": 1.5},
    "topS_buy_vol_b":  {"enabled": True, "low": 0, "high": 3000, "gamma": 1.0, "invert": False, "log": True, "log_scale": 100, "skew":  0.0,  "softness": 1.5},
    "topS_sell_vol_b": {"enabled": True, "low": 0, "high": 3000, "gamma": 1.0, "invert": False, "log": True, "log_scale": 100, "skew":  0.0,  "softness": 1.5},
}
# Cópia direta e [0,1]→[-1,1]
SKIP_NORM: Iterable[str]  = {"pctx0", "pctx1", "pctx2", "pctx3", "pctx4"}
SHARE_FEATS: Iterable[str] = {}

# === MANUAL NORMALIZATION ===================================================
def _apply_manual_norm(x: np.ndarray, conf: Dict[str, Any]) -> np.ndarray:
    
    x = x.astype(np.float32)

    low, high = conf["low"], conf["high"]

    # 1) log opcional
    if conf.get("log", False):
        s = conf.get("log_scale", 1.0)
        x = np.log1p(x / s)
        low = np.log1p(max(low, 0) / s)
        high = np.log1p(max(high, 0) / s)

    den = high - low

    # 2) escala para [-1, 1]
    y = 2.0 * ((x - low) / den) - 1.0

    # 3) tanh para suavizar bordas
    softness = conf.get("softness", 1.0)
    y = np.tanh(softness * y)

    # 4) skew logarítmico
    skew = conf.get("skew", 0.0)
    if skew != 0.0:
        alpha = abs(skew)
        p = 0.5 * (y + 1.0)
        if skew > 0:
            p = 1.0 - np.log1p(alpha * (1.0 - p)) / np.log1p(alpha)
        else:
            p = np.log1p(alpha * p) / np.log1p(alpha)
        y = 2.0 * p - 1.0

    # 5) gamma
    gamma = conf.get("gamma", 1.0)
    if gamma != 1.0:
        y = np.sign(y) * (np.abs(y) ** gamma)

    # 6) inversão
    if conf.get("invert", False):
        y = -y

    return np.clip(y, -1.0, 1.0).astype(np.float32)

def normalize_matrix(X: np.ndarray) -> np.ndarray:
    if X.shape[1] != len(FEATURE_ORDER):
        raise ValueError(f"Expected {len(FEATURE_ORDER)} features but got {X.shape[1]}")
    Y = np.empty_like(X, dtype=np.float32)
    for j, feat in enumerate(FEATURE_ORDER):
        col = X[:, j]
        if feat in SKIP_NORM:
            Y[:, j] = col.astype(np.float32)
            continue
        conf = MANUAL_NORM_CONF.get(feat, {})
        if conf.get("enabled", False):
            Y[:, j] = _apply_manual_norm(col, conf)
        else:
            # sem dinâmica: copia “as is” se não tiver config manual
            Y[:, j] = col.astype(np.float32)
    return Y

def normalize_matrix_file(input_path: Path, output_path: Path | None = None) -> Path:
    X = np.load(input_path)
    if output_path is None:
        output_path = input_path.with_name(input_path.stem + "_norm" + input_path.suffix)
    Y = normalize_matrix(X)
    np.save(output_path, Y)
    return output_path

def normalize_directory(data_dir: Path, pattern: str = "*.npy") -> None:
    for path in sorted(data_dir.glob(pattern)):
        if not path.name.endswith(".npy"): 
            continue
        if path.stem.endswith("_norm"):
            continue
        out_path = path.with_name(path.stem + "_norm" + path.suffix)
        try:
            normalize_matrix_file(path, out_path)
            print(f"[ok] normalised {path.name} -> {out_path.name}")
        except Exception as exc:
            print(f"[erro] failed to normalise {path.name}: {exc}")

if __name__ == "__main__":
    datadir = Path(r"C:\Users\Aldair\Desktop\eventos_processados")
    normalize_directory(data_dir=datadir)

    # days = ["20250314", "20250310", "20250311", "20250312", "20250313"]
    # for x in days:
    #     input_path = Path(rf"E:\Mercado BMF&BOVESPA\tryd\eventos_processados\{x}_wdo_dol.npy")
    #     output_path = input_path.with_name(input_path.stem + "_norm.npy")
    #     normalize_matrix_file(input_path, output_path)
