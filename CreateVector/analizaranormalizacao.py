# analyze_norm_day.py
from __future__ import annotations
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
    rows.sort(key=lambda r: abs(r[1] if np.isfinite(r[1]) else 0.0), reverse=True)

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
    figsize=(13,4),
    marker=".",
    linewidth=1.0,
    grid=True
) -> None:
    """
    Plota lado-a-lado (original vs normalizado) com média do normalizado no título.
    """
    mu_norm = float(np.nanmean(y_norm)) if y_norm.size else float("nan")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharex=False)

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

    fig.suptitle(f"{feat_name} — mean(norm)={mu_norm:.3f}")
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
        name = feat_names[j]
        y0 = X[:, j]
        y1 = Xn[:, j]
        plot_original_vs_normalized(y0, y1, name)

if __name__ == "__main__":
    # ===== CONFIGURAÇÃO RÁPIDA =====
    DATA_DIR = Path(r"E:\Mercado BMF&BOVESPA\tryd\eventos_processados")  # onde estão os .npy
    DAY      = "20181205"                                               # selecione o dia
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
