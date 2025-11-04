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

# === HELPERS ================================================================
def _soft_clamp01(z: np.ndarray, softness: float) -> np.ndarray:
    """
    Aproxima clip(z, 0, 1) com transição suave.
    Usa fórmula estável para evitar overflow em exp().
    """
    if softness <= 0.0:
        return np.clip(z, 0.0, 1.0, out=z)

    s = float(softness)
    # Evita exp muito grandes: corta a entrada em ±L (L~60 => exp(L) ~ 1e26)
    L = 60.0
    z1 = np.clip(z / s, -L, L)
    z2 = np.clip((z - 1.0) / s, -L, L)

    # softplus(x) = log(1 + exp(x))
    sp1 = np.log1p(np.exp(z1))
    sp2 = np.log1p(np.exp(z2))
    return s * (sp1 - sp2)

def _winsorize(col: np.ndarray, p_low: float, p_high: float) -> np.ndarray:
    lo = float(np.nanpercentile(col, p_low))
    hi = float(np.nanpercentile(col, p_high))
    if hi <= lo:
        return np.clip(col, lo, lo)
    return np.clip(col, lo, hi)

def _finite(col: np.ndarray) -> np.ndarray:
    m = np.isfinite(col)
    return col[m] if np.any(m) else np.array([0.0], dtype=float)

# === CONFIG =================================================================
# Regra: defina todas as features aqui. Use "enabled=False" para delegar ao
# fluxo dinâmico. Quando quiser manual, coloque enabled=True e preencha.
# Novos campos:
# - p_low/p_high   : percentis (0..100) para derivar low/high dos dados
# - winsor_pct     : winsorização simétrica (ex.: 1 -> 1% e 99%)
# - softness       : 0 = clamp duro; 0.02~0.05 costuma ir bem
# - gain           : ganho linear em torno do centro (aumenta std sem mexer no gamma)
# - target_std     : calcula um gain aproximado para atingir este std (aplicado antes do soft-clamp)
# - log            : bool (usa log1p para dados positivos)
# - gamma/invert   : igual ao seu código anterior

MANUAL_NORM_CONF = {

    "dt":   {"enabled": True,  "low": 3,  "high": 30, "gamma": 1.0, "invert": False, "log": True,  "softness": 0.1, "gain": 1.2},
    "dp":   {"enabled": True,  "low": -2,  "high": 2, "gamma": 1.0, "invert": False, "log": False,  "softness": 0.01, "gain": 1.0},
    "dmx":  {"enabled": True,  "low": -2.0, "high": 0.0,  "gamma": 1.0, "invert": True,  "log": False, "softness": 0.05, "gain": 1.0},
    "dmm":  {"enabled": True,  "low":  0.0, "high": 2.0,  "gamma": 1.0, "invert": False,  "log": False, "softness": 0.05, "gain": 1.0},
    "dmt":  {"enabled": True,  "low": -2.0, "high": 2.0,  "gamma": 1.0, "invert": False,  "log": False, "softness": 0.05, "gain": 1.0},
    "dmv":  {"enabled": True,  "low": -2.0, "high": 2.0,  "gamma": 1.0, "invert": False,  "log": False, "softness": 0.05, "gain": 1.0},
    "dvwap":  {"enabled": True,  "low": -20.0, "high": 20.0,  "gamma": 1.0, "invert": False,  "log": False, "softness": 0.1, "gain": 1.0},
    "ddayo":  {"enabled": True,  "low":  -30.0, "high": 30.0,  "gamma": 1.0, "invert": False,  "log": False, "softness": 0.1, "gain": 1.0},
    "d5m":  {"enabled": True,  "low": -5.0, "high": 5.0, "gamma": 1.0, "invert": False,  "log": False, "softness": 0.1, "gain": 1.0},
    "d30m":  {"enabled": True,  "low": -10.0, "high": 10.0, "gamma": 1.0, "invert": False,  "log": False, "softness": 0.1, "gain": 1.0},
    "rng":  {"enabled": True,  "low": 0.5,  "high": 3.0, "gamma": 0.9, "invert": False, "log": True,  "softness": 0.05, "gain": 1.2},
    "efrang": {"enabled": True,"low": 0.0,  "high": 1.0, "gamma": 1.0, "invert": False, "log": False, "softness": 0.02, "gain": 1.0},

    "b_vol": {"enabled": True, "low": 5, "high": 10000, "gamma": 1.0, "invert": False, "log": False, "softness": 0.01, "gain": 1.0},
    "s_vol": {"enabled": True, "low": 5, "high": 10000, "gamma": 1.0, "invert": False, "log": False, "softness": 0.01, "gain": 1.0},
    "t_vol": {"enabled": True, "low": 1000, "high": 10000, "gamma": 1.0, "invert": False, "log": True, "softness": 0.1, "gain": 1.0},
    "d_vol": {"enabled": True, "low": -5000, "high": 5000, "gamma": 0.7, "invert": False, "log": False, "softness": 0.1, "gain": 1.0},
    "avg_vol": {"enabled": True, "low": 5, "high": 95, "gamma": 0.7, "invert": False, "log": True, "softness": 0.05, "gain": 1.0},

    "g_b": {"enabled": True, "low": 5, "high": 50, "gamma": 1.0, "invert": False, "log": True, "softness": 0.1, "gain": 1.0},
    "g_s": {"enabled": True, "low": 5, "high": 50, "gamma": 1.0, "invert": False, "log": True, "softness": 0.1, "gain": 1.0},
    "g_n": {"enabled": True, "low": 2, "high": 30, "gamma": 1.0, "invert": False, "log": True, "softness": 0.1, "gain": 1.0},

    "rate_avg": {"enabled": True, "low": 5, "high": 95, "gamma": 1.0, "invert": False, "log": True, "softness": 0.1, "gain": 1.0},
    
    "max_streak_buy": {"enabled": True, "low": 0, "high": 15, "gamma": 1.0, "invert": False, "log": False, "softness": 0.01, "gain": 1.0},
    "max_streak_sell": {"enabled": True, "low": 0, "high": 15, "gamma": 1.0, "invert": False, "log": False, "softness": 0.01, "gain": 1.0},

    "ema_vol_total": {"enabled": True, "low": 300, "high": 1000, "gamma": 1.0, "invert": False, "log": True, "softness": 0.1, "gain": 1.0},
    "ema_order_rate": {"enabled": True, "low": 5, "high": 70, "gamma": 1.0, "invert": False, "log": True, "softness": 0.03, "gain": 1.2},

    "d_ticks_max": {"enabled": True, "low": 0, "high": 4, "gamma": 1.0, "invert": False, "log": True, "softness": 0.01, "gain": 1.0},

    "vol_per_tick_up_a_mean": {"enabled": True, "low": 0, "high": 200, "gamma": 1.0, "invert": False, "log": True, "softness": 0.01, "gain": 1.0},
    "vol_per_tick_up_a_max": {"enabled": True, "low": 0, "high": 200, "gamma": 1.0, "invert": False, "log": True, "softness": 0.01, "gain": 1.0},
    "vol_per_tick_up_a_sum": {"enabled": True, "low": 0, "high": 400, "gamma": 1.0, "invert": False, "log": True, "softness": 0.01, "gain": 1.0},
    "vol_per_tick_dn_a_mean": {"enabled": True, "low": 0, "high": 200, "gamma": 1.0, "invert": False, "log": True, "softness": 0.01, "gain": 1.0},
    "vol_per_tick_dn_a_max": {"enabled": True, "low": 0, "high": 200, "gamma": 1.0, "invert": False, "log": True, "softness": 0.01, "gain": 1.0},
    "vol_per_tick_dn_a_sum": {"enabled": True, "low": 0, "high": 400, "gamma": 1.0, "invert": False, "log": True, "softness": 0.01, "gain": 1.0},
    "vol_per_tick_up_b_mean": {"enabled": True, "low": 100, "high": 1000, "gamma": 2.0, "invert": False, "log": True, "softness": 0.1, "gain": 1.0},
    "vol_per_tick_up_b_max": {"enabled": True, "low": 0, "high": 100, "gamma": 1.0, "invert": False, "log": True, "softness": 0.01, "gain": 1.0},
    "vol_per_tick_up_b_sum": {"enabled": True, "low": 0, "high": 4000, "gamma": 1.0, "invert": False, "log": True, "softness": 0.01, "gain": 1.0},
    "vol_per_tick_dn_b_mean": {"enabled": True, "low": 0, "high": 2000, "gamma": 1.0, "invert": False, "log": True, "softness": 0.01, "gain": 1.0},
    "vol_per_tick_dn_b_max": {"enabled": True, "low": 0, "high": 2000, "gamma": 1.0, "invert": False, "log": True, "softness": 0.01, "gain": 1.0},
    "vol_per_tick_dn_b_sum": {"enabled": True, "low": 0, "high": 4000, "gamma": 1.0, "invert": False, "log": True, "softness": 0.01, "gain": 1.0},

    "buy_share_wdo": {"enabled": True, "low": 0, "high": 1, "gamma": 1.0, "invert": False, "log": True, "softness": 0.01, "gain": 1.0},
    "sell_share_wdo": {"enabled": True, "low": 0, "high": 1, "gamma": 1.0, "invert": False, "log": True, "softness": 0.01, "gain": 1.0},

    "ease_of_move_a": {"enabled": True, "low": -1000, "high": 1000, "gamma": 1.0, "invert": False, "log": False, "softness": 0.1, "gain": 1.0},
    "ease_of_move_b": {"enabled": True, "low": -3000, "high": 3000, "gamma": 1.0, "invert": False, "log": False, "softness": 0.1, "gain": 1.0},

    "absorption_buy_a": {"enabled": True, "low": 0, "high": 500, "gamma": 1.0, "invert": False, "log": False, "softness": 0.1, "gain": 1.4},
    "absorption_sell_a": {"enabled": True, "low": 0, "high": 500, "gamma": 1.0, "invert": False, "log": False, "softness": 0.1, "gain": 1.4},
    "absorption_buy_b": {"enabled": True, "low": 0, "high": 3000, "gamma": 1.0, "invert": False, "log": False, "softness": 0.1, "gain": 1.4},
    "absorption_sell_b": {"enabled": True, "low": 0, "high": 3000, "gamma": 1.0, "invert": False, "log": False, "softness": 0.1, "gain": 1.4},

    "topB_buy_vol_a": {"enabled": True, "low": 0, "high": 1000, "gamma": 1.0, "invert": False, "log": True, "softness": 0.1, "gain": 1.0},
    "topB_sell_vol_a": {"enabled": True, "low": 0, "high": 1000, "gamma": 1.0, "invert": False, "log": True, "softness": 0.1, "gain": 1.0},
    "topS_buy_vol_a": {"enabled": True, "low": 0, "high": 1000, "gamma": 1.0, "invert": False, "log": True, "softness": 0.1, "gain": 1.0},
    "topS_sell_vol_a": {"enabled": True, "low": 0, "high": 1000, "gamma": 1.0, "invert": False, "log": True, "softness": 0.1, "gain": 1.0},
    
    "topB_buy_vol_b": {"enabled": True, "low": 0, "high": 5000, "gamma": 1.0, "invert": False, "log": True, "softness": 0.1, "gain": 1.0},
    "topB_sell_vol_b": {"enabled": True, "low": 0, "high": 5000, "gamma": 1.0, "invert": False, "log": True, "softness": 0.1, "gain": 1.0},
    "topS_buy_vol_b": {"enabled": True, "low": 0, "high": 5000, "gamma": 1.0, "invert": False, "log": True, "softness": 0.1, "gain": 1.0},
    "topS_sell_vol_b": {"enabled": True, "low": 0, "high": 5000, "gamma": 1.0, "invert": False, "log": True, "softness": 0.1, "gain": 1.0},

}
# Cópia direta e [0,1]→[-1,1]
SKIP_NORM: Iterable[str]  = {"pctx0", "pctx1", "pctx2", "pctx3", "pctx4"}
SHARE_FEATS: Iterable[str] = {}

# === DYNAMIC PARAMS =========================================================

def _compute_dynamic_params(X: np.ndarray) -> Tuple[Dict[str, float], Dict[str, bool]]:
    feat_scale: Dict[str, float] = {}
    feat_use_log: Dict[str, bool] = {}
    n_features = X.shape[1]
    for idx in range(n_features):
        feat = FEATURE_ORDER[idx]
        # Se manual estiver habilitado, não calcula dinâmico
        if MANUAL_NORM_CONF.get(feat, {}).get("enabled", False):
            continue
        if feat in SKIP_NORM or feat in SHARE_FEATS:
            continue
        col = _finite(X[:, idx].astype(float))
        if col.size == 0:
            feat_scale[feat] = 1.0
            feat_use_log[feat] = False
            continue
        if np.min(col) < 0.0:
            scale = float(np.percentile(np.abs(col), 95))
            feat_scale[feat] = scale if scale > 0 else 1.0
            feat_use_log[feat] = False
        else:
            p95 = float(np.percentile(col, 95))
            mx  = float(np.max(col))
            if p95 > 0.0 and mx > 2.0 * p95:
                arr_log = np.log1p(np.clip(col, 0.0, None))
                scale = float(np.percentile(arr_log, 95))
                feat_scale[feat] = scale if scale > 0 else 1.0
                feat_use_log[feat] = True
            else:
                feat_scale[feat] = p95 if p95 > 0 else 1.0
                feat_use_log[feat] = False
    return feat_scale, feat_use_log

# === MANUAL NORMALIZATION ===================================================

def _resolve_bounds(col_raw: np.ndarray, conf: Dict[str, Any], use_log: bool) -> Tuple[float, float, np.ndarray]:
    """Resolve low/high a partir de valores fixos ou percentis; aplica winsor se pedido."""
    col_finite = _finite(col_raw.astype(float))

    # Winsorização opcional (no domínio bruto)
    winsor_pct = float(conf.get("winsor_pct", 0.0))
    if winsor_pct > 0.0:
        col_w = _winsorize(col_finite, winsor_pct, 100.0 - winsor_pct)
    else:
        col_w = col_finite

    # Deriva limites por percentil ou usa low/high fixos
    if "p_low" in conf and "p_high" in conf:
        lo_raw = float(np.nanpercentile(col_w, conf["p_low"]))
        hi_raw = float(np.nanpercentile(col_w, conf["p_high"]))
    else:
        lo_raw = float(conf.get("low", np.nanpercentile(col_w, 5.0)))
        hi_raw = float(conf.get("high", np.nanpercentile(col_w, 95.0)))

    if use_log:
        lo_proc = np.log1p(max(lo_raw, 0.0))
        hi_proc = np.log1p(max(hi_raw, 0.0))
        col_proc = np.log1p(np.clip(col_raw.astype(float), 0.0, None))
    else:
        lo_proc = lo_raw
        hi_proc = hi_raw
        col_proc = col_raw.astype(float)

    # Evita divisão por zero
    if hi_proc <= lo_proc:
        hi_proc = lo_proc + 1e-6
    return lo_proc, hi_proc, col_proc

def _apply_manual_norm(col: np.ndarray, conf: Dict[str, Any]) -> np.ndarray:
    """
    Agora com:
      - percentis para limites
      - winsor opcional
      - soft-clamp nas pontas
      - ganho linear (ou alvo de std) sem mexer no gamma
    """
    gamma   = float(conf.get("gamma", 1.0))
    invert  = bool(conf.get("invert", False))
    use_log = bool(conf.get("log", False))
    softness = float(conf.get("softness", 0.0))
    gain     = float(conf.get("gain", 1.0))
    target_std = conf.get("target_std", None)

    low_proc, high_proc, col_proc = _resolve_bounds(col, conf, use_log)

    # Linear [low, high] → [0,1] (sem clamp)
    denom = (high_proc - low_proc)
    scaled = (col_proc - low_proc) / denom

    # Ajuste de ganho em torno do centro para expandir/contrair sem tocar gamma
    # Se target_std informado, estima um ganho básico (antes de soft e gamma).
    if target_std is not None:
        # estimativa usando mapeamento linear centrado
        base_centered = (scaled - 0.5) * 2.0
        std0 = float(np.nanstd(base_centered))
        if std0 > 1e-9:
            gain_est = float(target_std) / std0
            # limites razoáveis pro ganho automático
            gain = np.clip(gain_est, 0.5, 5.0)
    if gain != 1.0:
        scaled = 0.5 + (scaled - 0.5) * gain

    # Soft-clamp para [0,1] com bordas suaves
    scaled = _soft_clamp01(scaled, softness)

    # [0,1] → [-1,1]
    mapped = scaled * 2.0 - 1.0
    if invert:
        mapped = -mapped

    # Curvatura por gamma (preserva sinal)
    if gamma != 1.0:
        mapped = np.sign(mapped) * (np.abs(mapped) ** gamma)

    return mapped.astype(np.float32)

# === DYNAMIC NORMALIZATION ==================================================

def _apply_dynamic_norm(col: np.ndarray, scale: float, use_log: bool) -> np.ndarray:
    if scale <= 0.0:
        scale = 1.0
    colp = np.log1p(np.clip(col.astype(float), 0.0, None)) if use_log else col.astype(float)
    return np.tanh(colp / scale).astype(np.float32)

# === PUBLIC API =============================================================

def normalize_matrix(X: np.ndarray) -> np.ndarray:
    if X.shape[1] != len(FEATURE_ORDER):
        raise ValueError(f"Expected {len(FEATURE_ORDER)} features but got {X.shape[1]}")

    feat_scale, feat_use_log = _compute_dynamic_params(X)
    Y = np.empty_like(X, dtype=np.float32)

    for j, feat in enumerate(FEATURE_ORDER):
        col = X[:, j]

        if feat in SKIP_NORM:
            Y[:, j] = col.astype(np.float32)
            continue

        if feat in SHARE_FEATS:
            y = 2.0 * col - 1.0
            np.clip(y, -1.0, 1.0, out=y)
            Y[:, j] = y.astype(np.float32)
            continue

        conf = MANUAL_NORM_CONF.get(feat, {})
        if conf.get("enabled", False):
            Y[:, j] = _apply_manual_norm(col, conf)
        else:
            scale = feat_scale.get(feat, 1.0)
            use_log = feat_use_log.get(feat, False)
            Y[:, j] = _apply_dynamic_norm(col, scale, use_log)

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
    days = ["20200309", "20200310", "20200311", "20200312", "20200313", "20200316"]
    for x in days:
        input_path = Path(rf"E:\Mercado BMF&BOVESPA\tryd\eventos_processados\{x}_wdo_dol.npy")
        output_path = input_path.with_name(input_path.stem + "_norm.npy")
        normalize_matrix_file(input_path, output_path)
