# verificar_normalizados.py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# importa a ordem das features do seu arquivo de normalização
from normalize_manual import FEATURE_ORDER

# === CONFIG ÚNICA (ajuste se quiser) ===
DATA_DIR = Path(r"C:\Users\Aldair\Desktop\eventos_processados")
PATTERN = "*_norm_norm.npy"   # só lê arquivos já normalizados
BINS = 60
SAT_THR = 0.99           # |y| >= 0.99 conta como saturado

def _finite(a: np.ndarray) -> np.ndarray:
    a = a[np.isfinite(a)]
    return a if a.size else np.array([0.0], dtype=float)

def _skew_kurt_excess(x: np.ndarray):
    x = _finite(x.astype(float))
    n = x.size
    if n < 3:
        return float("nan"), float("nan")
    mu = x.mean()
    var = x.var(ddof=1)
    if var <= 0:
        return 0.0, -3.0
    s = np.sqrt(var)
    z = (x - mu) / s
    # skew (amostral corrigido) e kurtosis (excesso)
    skew = (np.sqrt(n*(n-1)) / max(n-2,1)) * np.mean(z**3)
    # fórmula simples p/ excesso (sem correção fina de viés)
    kurt_excess = np.mean(z**4) - 3.0
    return float(skew), float(kurt_excess)

def main():
    files = sorted(DATA_DIR.glob(PATTERN))
    if not files:
        print(f"[ERRO] Nenhum arquivo encontrado em {DATA_DIR} com padrão {PATTERN}")
        return

    arrs = []
    for f in files:
        X = np.load(f)
        if X.ndim != 2:
            print(f"[AVISO] {f.name}: shape inesperado {X.shape}; pulando.")
            continue
        arrs.append(X)
    if not arrs:
        print("[ERRO] Nada para analisar.")
        return

    Y = np.vstack(arrs)
    n_feat = Y.shape[1]
    if n_feat != len(FEATURE_ORDER):
        print(f"[AVISO] colunas={n_feat}, FEATURE_ORDER={len(FEATURE_ORDER)}")

    print("=== Estatísticas do NORMALIZADO ===")
    print(f"Total de amostras: {Y.shape[0]}")
    print("feature                            mean     std      skew     kurt     sat%    min     p01     p50     p99     max")
    print("-"*110)

    for j in range(n_feat):
        col = _finite(Y[:, j])
        feat = FEATURE_ORDER[j] if j < len(FEATURE_ORDER) else f"f{j}"

        mean = float(np.mean(col))
        std  = float(np.std(col, ddof=0))
        skew, kurt = _skew_kurt_excess(col)
        sat  = float(np.mean(np.abs(col) >= SAT_THR) * 100.0)

        p01 = float(np.percentile(col, 1))
        p50 = float(np.percentile(col, 50))
        p99 = float(np.percentile(col, 99))
        mn  = float(np.min(col))
        mx  = float(np.max(col))

        print(f"{feat:32s}  {mean:7.3f}  {std:7.3f}  {skew:8.3f}  {kurt:8.3f}  {sat:6.1f}  {mn:7.3f}  {p01:7.3f}  {p50:7.3f}  {p99:7.3f}  {mx:7.3f}")

        # Histograma simples em [-1, 1]
        plt.figure(figsize=(6,4))
        plt.hist(col, bins=BINS)
        plt.title(f"Distribuição — {feat}")
        plt.xlabel("valor normalizado")
        plt.ylabel("contagem")
        # plt.xlim(-1.0, 1.0)
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
