# normalize_dataset.py
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Dict, Any, Iterable, Optional
import numpy as np
import joblib

# === Escolha do normalizador ===
# 'quantile' (recomendado), 'standard' ou 'robust'
def make_scaler(method: str = "quantile"):
    if method == "quantile":
        from sklearn.preprocessing import QuantileTransformer
        # mapeia quantis -> Normal(0,1), excelente p/ caudas e degraus
        return QuantileTransformer(output_distribution="normal", n_quantiles=1000, subsample=int(1e6), random_state=0)
    elif method == "standard":
        from sklearn.preprocessing import StandardScaler
        return StandardScaler(with_mean=True, with_std=True)
    elif method == "robust":
        from sklearn.preprocessing import RobustScaler
        # robusto a outliers (usa IQR); bom se quiser menos “esticado” que quantile
        return RobustScaler(with_centering=True, with_scaling=True, quantile_range=(10.0, 90.0))
    else:
        raise ValueError(f"method inválido: {method}")

# === Descoberta de arquivos ===
def list_daily_files(folder: Path, pair: Tuple[str, str]) -> List[Path]:
    """
    Lista arquivos diários salvos pelo process_day: YYYYMMDD_{a}_{b}.npy
    Ex.: 20181205_wdo_dol.npy
    """
    a, b = pair[0].lower(), pair[1].lower()
    pat = f"*_{a}_{b}_norm.npy"
    return sorted(folder.glob(pat))

# === Carregamento de um dia ===
def load_day_matrix(path: Path) -> np.ndarray:
    X = np.load(path)
    # garante float32 para economizar RAM/tempo
    if X.dtype != np.float32:
        X = X.astype(np.float32, copy=False)
    return X

# === Monta dataset para fit (stack) ===
def build_stack_for_fit(files: List[Path], max_rows_per_day: Optional[int] = None) -> np.ndarray:
    """
    Empilha linhas de vários dias para ajustar o scaler.
    max_rows_per_day: amostragem por dia (None = usa tudo).
    """
    mats = []
    for p in files:
        X = load_day_matrix(p)
        if X.size == 0:
            continue
        if max_rows_per_day is not None and X.shape[0] > max_rows_per_day:
            # amostra aleatório e estável:
            rng = np.random.default_rng(0)
            idx = rng.choice(X.shape[0], size=max_rows_per_day, replace=False)
            X = X[idx]
        mats.append(X)
    if not mats:
        raise RuntimeError("Nenhuma matriz encontrada para ajuste do scaler.")
    X_all = np.vstack(mats)
    return X_all

# === Fit e salvamento do scaler ===
def fit_and_save_scaler(
    data_folder: Path,
    pair: Tuple[str, str],
    scaler_out: Path,
    method: str = "quantile",
    max_rows_per_day: Optional[int] = None,
    feature_order_json: Optional[Path] = None,
) -> Tuple[Any, List[str]]:
    """
    Ajusta o scaler no conjunto (stack) e salva (scaler + feature_order).
    feature_order_json: opcional; se existir um JSON com a lista de features, salvamos junto.
    """
    files = list_daily_files(data_folder, pair)
    if not files:
        raise RuntimeError(f"Nenhum .npy encontrado em {data_folder} para o par {pair}.")

    # Empilha dados para ajuste
    X_all = build_stack_for_fit(files, max_rows_per_day=max_rows_per_day)

    scaler = make_scaler(method)
    Xn_sample = scaler.fit_transform(X_all)  # apenas para validar

    # feature_order: se você tiver salvo isso em JSON, pode carregar; senão, trabalhamos posicionalmente
    feature_order: List[str]
    if feature_order_json and feature_order_json.exists():
        import json
        feature_order = json.loads(feature_order_json.read_text(encoding="utf-8"))
    else:
        # fallback: nomes posicionais f0..fN-1
        feature_order = [f"f{i}" for i in range(X_all.shape[1])]

    joblib.dump(
        {"scaler": scaler, "feature_order": feature_order, "method": method, "pair": tuple(map(str.lower, pair))},
        scaler_out
    )
    print(f"[ok] scaler salvo em: {scaler_out}  (method={method}, shape={X_all.shape})")
    return scaler, feature_order

# === Aplicar scaler e salvar versões normalizadas ===
def apply_scaler_to_folder(
    data_folder: Path,
    pair: Tuple[str, str],
    scaler_pkl: Path,
    out_suffix: str = "_norm",
    overwrite: bool = False
) -> List[Path]:
    """
    Aplica o scaler salvo a cada arquivo diário da pasta e salva YYYYMMDD_{a}_{b}{out_suffix}.npy
    Retorna a lista de arquivos gerados.
    """
    meta = joblib.load(scaler_pkl)
    scaler = meta["scaler"]
    feature_order = meta["feature_order"]
    files = list_daily_files(data_folder, pair)
    out_files: List[Path] = []

    for p in files:
        X = load_day_matrix(p)
        if X.size == 0:
            continue
        Xn = scaler.transform(X)
        p_out = p.with_name(p.stem + out_suffix + p.suffix)
        if p_out.exists() and not overwrite:
            print(f"[skip] {p_out.name} (já existe)")
        else:
            np.save(p_out, Xn.astype(np.float32, copy=False))
            print(f"[ok] salvo {p_out.name}  shape={Xn.shape}")
        out_files.append(p_out)
    return out_files

# === Normalizar 1 evento isolado (dict -> dict) ===
def normalize_single_event_vector(
    vector: Dict[str, float],
    scaler_pkl: Path,
    feature_order: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Recebe um dict {feature: valor} de um único evento e devolve outro dict normalizado,
    usando o scaler salvo. Se não passar feature_order, tenta ler do pkl.
    """
    meta = joblib.load(scaler_pkl)
    scaler = meta["scaler"]
    feat_order = feature_order or meta["feature_order"]

    x = np.array([[float(vector.get(k, 0.0)) for k in feat_order]], dtype=np.float32)
    xn = scaler.transform(x)[0]
    return {k: float(v) for k, v in zip(feat_order, xn)}

# === CLI simples de exemplo ===
if __name__ == "__main__":
    # Exemplo de uso rápido:
    DATA_DIR = Path(r"E:\Mercado BMF&BOVESPA\tryd\eventos_processados")  # onde estão os .npy
    PAIR     = ("wdo", "dol")
    SCALER_PKL = DATA_DIR / "scaler_wdo_dol.pkl"

    # 1) Ajusta e salva o scaler com base em vários dias
    scaler, feat_order = fit_and_save_scaler(
        data_folder=DATA_DIR,
        pair=PAIR,
        scaler_out=SCALER_PKL,
        method="quantile",        # 'quantile' recomendado
        max_rows_per_day=None,    # amostra p/ acelerar; tire para usar tudo
        feature_order_json=None   # passe um JSON com a ordem se você tiver salvo
    )

    # 2) Aplica a todos os dias e salva *_norm.npy
    apply_scaler_to_folder(
        data_folder=DATA_DIR,
        pair=PAIR,
        scaler_pkl=SCALER_PKL,
        out_suffix="_norm_norm",
        overwrite=False
    )


