# renormalizar_sklearn.py
import numpy as np
from pathlib import Path

# >>> ajuste aqui se quiser outro diretório
DATA_DIR = Path(r"C:\Users\Aldair\Desktop\eventos_processados")

# ===== escolha do scaler (use UM deles) =====
from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import RobustScaler
# SCALER = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(5.0, 95.0))
SCALER = StandardScaler(with_mean=True, with_std=True)
# ===========================================

PATTERN_IN   = "*_norm.npy"        # lê os já normalizados manualmente
SUFIX_OUT    = "_norm"             # adiciona mais um _norm
SATURATE_OUT = False               # True -> clip em [-1,1] após o sklearn (opcional)

def _load_all_for_fit(files):
    arrs = []
    for f in files:
        X = np.load(f)
        if X.ndim != 2:
            print(f"[aviso] pulando {f.name}: shape {X.shape} não é 2D.")
            continue
        arrs.append(X)
    if not arrs:
        raise RuntimeError("Nenhum arquivo válido para fitar o scaler.")
    Y = np.vstack(arrs)

    # tratar não-finitos: substitui por mediana da coluna
    mask = ~np.isfinite(Y)
    if mask.any():
        col_meds = np.nanmedian(np.where(np.isfinite(Y), Y, np.nan), axis=0)
        # fallback se alguma coluna ficou toda NaN
        col_meds = np.where(np.isfinite(col_meds), col_meds, 0.0)
        Y = np.where(mask, col_meds[np.newaxis, :], Y)
    return Y

def main():
    files = sorted(DATA_DIR.glob(PATTERN_IN))
    # evita reprocessar os que já têm duas vezes _norm
    files = [f for f in files if not f.stem.endswith("_norm_norm")]
    if not files:
        print(f"[ERRO] Nenhum arquivo {PATTERN_IN} para processar em {DATA_DIR}")
        return

    print(f"[INFO] Achados {len(files)} arquivos para fitar scaler global.")
    Yfit = _load_all_for_fit(files)

    print("[INFO] Fitando scaler do scikit-learn no conjunto combinado...")
    scaler = SCALER
    scaler.fit(Yfit)
    import joblib
    joblib.dump(scaler, "scaler_norm2.pkl")
    print("[OK] scaler salvo em scaler_norm2.pkl")


    print("[INFO] Transformando e salvando...")
    for f in files:
        X = np.load(f)

        # mesma imputação simples por arquivo (caso tenham NaNs/Inf)
        mask = ~np.isfinite(X)
        if mask.any():
            col_meds = np.nanmedian(np.where(np.isfinite(X), X, np.nan), axis=0)
            col_meds = np.where(np.isfinite(col_meds), col_meds, 0.0)
            X = np.where(mask, col_meds[np.newaxis, :], X)

        X2 = scaler.transform(X)

        if SATURATE_OUT:
            X2 = np.clip(X2, -1.0, 1.0)

        out_path = f.with_name(f.stem + SUFIX_OUT + f.suffix)  # vira *_norm_norm.npy
        np.save(out_path, X2.astype(np.float32))
        print(f"[ok] {f.name} -> {out_path.name}")

    print("[OK] Tudo pronto.")

if __name__ == "__main__":
    main()
