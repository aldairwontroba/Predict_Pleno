# analyze_norm_day.py
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy.stats import skew as _skew, kurtosis as _kurtosis
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

def _safe_skew(x: np.ndarray) -> float:
    if _HAS_SCIPY:
        return float(_skew(x, bias=False, nan_policy="omit"))
    x = x[np.isfinite(x)]
    if x.size < 3:
        return float("nan")
    m = x.mean()
    s = x.std(ddof=1)
    if s == 0:
        return 0.0
    n = x.size
    return float((np.sum(((x - m)/s)**3) * n) / ((n-1)*(n-2)))

def _safe_kurtosis(x: np.ndarray) -> float:
    if _HAS_SCIPY:
        # Fisher (excess) by default; aqui queremos "regular" (não-excess)? Vamos usar excess=False.
        return float(_kurtosis(x, fisher=False, bias=False, nan_policy="omit"))
    x = x[np.isfinite(x)]
    if x.size < 4:
        return float("nan")
    m = x.mean()
    s2 = x.var(ddof=1)
    if s2 == 0:
        return 3.0
    n = x.size
    m4 = np.mean((x - m)**4)
    return float(m4 / (s2**2))

def load_day_mats(
    data_dir: Path,
    day: str,
    pair: Tuple[str, str],
    norm_suffix: str = "_norm",
    scaler_pkl: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Carrega X_orig e X_norm para um dia. Se X_norm não existir e scaler_pkl for fornecido,
    normaliza on-the-fly.
    """
    a, b = pair[0].lower(), pair[1].lower()
    base = data_dir / f"{day}_{a}_{b}.npy"
    norm = data_dir / f"{day}_{a}_{b}{norm_suffix}.npy"

    if not base.exists():
        raise FileNotFoundError(f"Arquivo base não encontrado: {base}")
    X = np.load(base).astype(np.float32, copy=False)

    if norm.exists():
        Xn = np.load(norm).astype(np.float32, copy=False)
    elif scaler_pkl is not None and Path(scaler_pkl).exists():
        import joblib
        meta = joblib.load(scaler_pkl)
        scaler = meta["scaler"]
        Xn = scaler.transform(X).astype(np.float32, copy=False)
        print(f"[info] Normalizado on-the-fly usando {scaler_pkl}")
    else:
        raise FileNotFoundError(
            f"Arquivo normalizado não encontrado: {norm} "
            f"e scaler_pkl não fornecido."
        )
    return X, Xn

def load_feature_names(feature_order_json: Optional[Path], n_features: int) -> List[str]:
    if feature_order_json and Path(feature_order_json).exists():
        try:
            names = json.loads(Path(feature_order_json).read_text(encoding="utf-8"))
            if isinstance(names, list) and len(names) == n_features:
                return [str(x) for x in names]
        except Exception:
            pass
    # fallback posicional
    return [f"f{i}" for i in range(n_features)]

def print_norm_stats_table(Xn: np.ndarray, feat_names: List[str], sat_thr: float = 0.995) -> None:
    """
    Imprime tabela: mean, std, skew, kurt, % saturado (|x| >= sat_thr).
    """
    rows = []
    for j, name in enumerate(feat_names):
        y = Xn[:, j]
        y = y[np.isfinite(y)]
        if y.size == 0:
            rows.append((name, np.nan, np.nan, np.nan, np.nan, 0.0))
            continue
        mu = float(np.mean(y))
        sd = float(np.std(y))
        sk = _safe_skew(y)
        ku = _safe_kurtosis(y)
        sat = float(np.mean(np.abs(y) >= sat_thr) * 100.0)
        rows.append((name, mu, sd, sk, ku, sat))

    # ordena por |mean| desc só para destacar os piores centramentos
    # rows.sort(key=lambda r: abs(r[1] if np.isfinite(r[1]) else 0.0), reverse=True)

    print(f"{'feature':<32} {'mean':>8} {'std':>8} {'skew':>8} {'kurt':>8} {'sat%':>8}")
    print("-" * 72)
    for name, mu, sd, sk, ku, sat in rows:
        def _fmt(x):
            return f"{x:8.3f}" if (x == x and np.isfinite(x)) else f"{'nan':>8}"
        print(f"{name:<32} {_fmt(mu)} {_fmt(sd)} {_fmt(sk)} {_fmt(ku)} {sat:8.1f}")

def plot_original_vs_normalized(
    y_orig: np.ndarray,
    y_norm: np.ndarray,
    feat_name: str,
    figsize=(13, 4),
    marker=".",
    linewidth=1.0,
    grid=True,
    bins="auto",              # nº de bins do histograma
) -> None:
    """
    Plota séries (original x normalizado) e, abaixo, a distribuição (histogramas)
    com linhas verticais nos valores mínimo e máximo.
    """
    y_orig = np.asarray(y_orig, dtype=float)
    y_norm = np.asarray(y_norm, dtype=float)

    y0 = y_orig[np.isfinite(y_orig)]
    y1 = y_norm[np.isfinite(y_norm)]

    mu_norm = float(np.nanmean(y1)) if y1.size else float("nan")
    std_norm = float(np.nanstd(y1)) if y1.size else float("nan")

    # limites min/max (ignorando NaNs)
    if y0.size:
        min0, max0 = float(np.nanmin(y0)), float(np.nanmax(y0))
    else:
        min0 = max0 = np.nan
    if y1.size:
        min1, max1 = float(np.nanmin(y1)), float(np.nanmax(y1))
    else:
        min1 = max1 = np.nan

    # figura 2x2 (topo: séries; base: distribuições)
    w, h = figsize
    fig, axes = plt.subplots(2, 2, figsize=(w, h * 2), sharex=False)
    (ax1, ax2), (ax3, ax4) = axes

    # --- topo: séries temporais ---
    ax1.plot(np.arange(y_orig.size), y_orig, marker=marker, linewidth=linewidth)
    ax1.set_title("Original")
    ax1.set_xlabel("amostra")
    ax1.set_ylabel(feat_name)
    if grid: ax1.grid(True, alpha=0.3)

    ax2.plot(np.arange(y_norm.size), y_norm, marker=marker, linewidth=linewidth)
    ax2.set_title("Normalizado")
    ax2.set_xlabel("amostra")
    ax2.set_ylabel(f"{feat_name} (norm.)")
    if grid: ax2.grid(True, alpha=0.3)

    # --- base: distribuições + min/max ---
    if y0.size:
        ax3.hist(y0, bins=bins, range=(min0, max0))
        ax3.axvline(min0, linestyle="--", linewidth=1)
        ax3.axvline(max0, linestyle="--", linewidth=1)
        ax3.set_title("Distribuição (original)")
        ax3.set_xlabel(feat_name)
        ax3.set_ylabel("contagem")
        ax3.text(0.02, 0.95, f"min={min0:.3g}\nmax={max0:.3g}",
                 transform=ax3.transAxes, va="top", ha="left")
        if grid: ax3.grid(True, alpha=0.3)

    if y1.size:
        ax4.hist(y1, bins=bins, range=(min1, max1))
        ax4.axvline(min1, linestyle="--", linewidth=1)
        ax4.axvline(max1, linestyle="--", linewidth=1)
        ax4.set_title("Distribuição (normalizado)")
        ax4.set_xlabel(f"{feat_name} (norm.)")
        ax4.set_ylabel("contagem")
        ax4.text(0.02, 0.95, f"min={min1:.3g}\nmax={max1:.3g}",
                 transform=ax4.transAxes, va="top", ha="left")
        if grid: ax4.grid(True, alpha=0.3)

    fig.suptitle(f"{feat_name} — mean(norm)={mu_norm:.3f}, std(norm)={std_norm:.3f}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def analyze_day(
    data_dir: Path,
    day: str,
    pair: Tuple[str, str],
    feature_order_json: Optional[Path] = None,
    scaler_pkl: Optional[Path] = None,
    norm_suffix: str = "_norm",
    plot_all: bool = True,
    selected: Optional[List[int]] = None
) -> None:
    """
    Carrega X e Xn e faz:
      (1) imprime tabela de estatísticas do normalizado;
      (2) plota, em loop, original vs normalizado para cada coluna (ou subset).
    """
    X, Xn = load_day_mats(data_dir, day, pair, norm_suffix=norm_suffix, scaler_pkl=scaler_pkl)
    feat_names = load_feature_names(feature_order_json, X.shape[1])

    print("\n=== Estatísticas do NORMALIZADO ===")
    print_norm_stats_table(Xn, feat_names, sat_thr=0.995)

    idxs = list(range(X.shape[1])) if plot_all and selected is None else (selected or [])
    for j in idxs:
        if j < 27:
            continue
        name = feat_names[j]
        y0 = X[:, j]
        y1 = Xn[:, j]
        plot_original_vs_normalized(y0, y1, name)

# normalize_manual_simple.py
import numpy as np
from pathlib import Path
from typing import Dict, Any, Iterable, Tuple


FEATURE_ORDER: Tuple[str, ...] = (
    "dt", 
    "dp", 
    "dmx", 
    "dmm", 
    "dmt", 
    "dmv", 
    "dvwap", 
    "ddayo", 
    "d5m", 
    "d30m",
    "rng", 
    "efrang", 
    "pctx0", "pctx1", "pctx2", "pctx3", "pctx4",
    "b_vol", 
    "s_vol", 
    "t_vol", 
    "d_vol", 
    "avg_vol",
    "g_b", 
    "g_s", 
    "g_n", 
    "rate_avg", 
    "max_streak_buy", 
    "max_streak_sell",
    "ema_vol_total", 
    "ema_order_rate", 
    "d_ticks_max",
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

# ---------- SUA CONFIG (mantida) ----------
# Mantive sua tabela como veio; o campo "gain" é ignorado pela implementação.
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

    "ema_vol_total": {"enabled": True, "low": 500, "high": 1000, "gamma": 1.0, "invert": False, "log": True, "log_scale": 10, "skew":  -10.0,  "softness": 2},
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

SKIP_NORM: Iterable[str]  = {"pctx0", "pctx1", "pctx2", "pctx3", "pctx4"}
SHARE_FEATS: Iterable[str] = {}  # não usado aqui

# ---------- NORMALIZAÇÃO MANUAL “PURO” ----------
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

# ---------- API ----------
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
        if not path.name.endswith(".npy") or path.stem.endswith("_norm"):
            continue
        out_path = path.with_name(path.stem + "_norm" + path.suffix)
        try:
            normalize_matrix_file(path, out_path)
            print(f"[ok] normalised {path.name} -> {out_path.name}")
        except Exception as exc:
            print(f"[erro] failed to normalise {path.name}: {exc}")

if __name__ == "__main__":
    # days = ["20200309", "20200310", "20200311", "20200312", "20200313", "20200316"]
    # for x in days:
    #     input_path = Path(rf"E:\Mercado BMF&BOVESPA\tryd\eventos_processados\{x}_wdo_dol.npy")
    #     output_path = input_path.with_name(input_path.stem + "_norm.npy")
    #     normalize_matrix_file(input_path, output_path)

    # ===== CONFIGURAÇÃO RÁPIDA =====
    DATA_DIR = Path(r"E:\Mercado BMF&BOVESPA\tryd\eventos_processados")  # onde estão os .npy
    DAY      = "20250313"                                               # selecione o dia
    PAIR     = ("wdo", "dol")
    # Se você salvou os nomes de features:
    FEATURE_ORDER_JSON = None  # Path(r"E:\...\feature_order.json")
    # Se ainda não existe *_norm.npy, pode apontar para o scaler para normalizar on-the-fly:
    SCALER_PKL = DATA_DIR / "scaler_wdo_dol.pkl"  # ou None se já existir *_norm.npy

    analyze_day(
        data_dir=DATA_DIR,
        day=DAY,
        pair=PAIR,
        feature_order_json=FEATURE_ORDER_JSON,
        scaler_pkl=SCALER_PKL,     # None se já existir {DAY}_{pair}_norm.npy
        norm_suffix="_norm",
        plot_all=True,             # True = plota todas as features
        selected=None              # ou, por exemplo, [0,1,2] para um subset
    )
