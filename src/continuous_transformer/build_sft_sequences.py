"""
Constrói o dataset de SFT para o ContinuousEventTransformer.

Para cada dia, faz janelas deslizantes:
  - past_events  : últimos n_past eventos normalizados (input do modelo)
  - action_label : direção calculada a partir dos próximos n_future eventos crus
  - meta         : metadados (dia da semana, horário, etc.) como floats

O label de ação usa os mesmos critérios de src/agent/build_sequences.py
(compute_future_features + thresholds em pontos reais do ativo).

Saída salva em SFTBuildConfig.output_path como:
  {
    "past_events"   : Tensor [N, n_past, input_dim]   float32
    "action_labels" : Tensor [N]                       long  (0=buy, 1=sell, 2=hold)
    "meta"          : Tensor [N, num_meta]             float32
    "day_keys"      : List[str]
    "day_indices"   : Tensor [N]                       long
    "center_indices": Tensor [N]                       long
    "cfg"           : dict   (configurações usadas)
  }

Uso:
  python -m src.continuous_transformer.build_sft_sequences
"""
from __future__ import annotations

import datetime as _dt
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from src.continuous_transformer.config import SFTBuildConfig, INPUT_DIM


# ============================================================
# Helpers de metadados (mesmo padrão de build_sequences.py)
# ============================================================

def _parse_date(day_key: str) -> _dt.date:
    return _dt.datetime.strptime(day_key, "%Y%m%d").date()


def _is_last_business_day(d: _dt.date) -> bool:
    m, y = d.month, d.year
    next_m = _dt.date(y + 1, 1, 1) if m == 12 else _dt.date(y, m + 1, 1)
    cur = next_m - _dt.timedelta(days=1)
    while cur.weekday() >= 5:
        cur -= _dt.timedelta(days=1)
    return d == cur


def _encode_meta(day_key: str, raw_events: np.ndarray, center_idx: int) -> List[float]:
    """
    Codifica metadados como vetor float [0, 1].
    Slots:
      0: dia_da_semana / 6
      1: bucket_horário / 4  (0=9h, 1=10h, 2=11h, 3=12-16h, 4=>=16h)
      2: is_last_business_day (0 ou 1)
      3: pos_state (sempre 0 = flat por enquanto)
      4-7: reservado (zeros)
    """
    d   = _parse_date(day_key)
    dow = d.weekday()  # 0=seg..6=dom

    # Tempo acumulado a partir do pregão (09:00)
    center_idx = int(max(0, min(center_idx, raw_events.shape[0] - 1)))
    dt_cum     = float(raw_events[: center_idx + 1, 0].sum())
    t_sec      = max(0.0, min(9 * 3600 + dt_cum, 24 * 3600 - 1))
    h          = int(t_sec / 3600)
    if h < 10:
        tod = 0
    elif h < 11:
        tod = 1
    elif h < 12:
        tod = 2
    elif h < 16:
        tod = 3
    else:
        tod = 4

    return [
        dow / 6.0,
        tod / 4.0,
        float(_is_last_business_day(d)),
        0.0,   # pos_state
        0.0, 0.0, 0.0, 0.0,
    ]


# ============================================================
# Cálculo de label de ação (adaptado de src/agent/build_sequences.py)
# ============================================================

def compute_direction_label(
    raw_events: np.ndarray,
    start: int,
    end: int,
    cfg: SFTBuildConfig,
) -> int:
    """
    Retorna 0=buy, 1=sell, 2=hold baseado nos movimentos de preço futuros.

    Usa os dados *crus* (não normalizados) para que os thresholds em pontos
    façam sentido.
    """
    n = raw_events.shape[0]
    if n == 0:
        return 2

    start = int(max(0, min(start, n - 1)))
    end   = int(max(start + 1, min(end, n)))
    if end <= start + 1:
        return 2

    deltas = raw_events[start:end, cfg.price_col].astype(float)
    deltas = deltas[~np.isnan(deltas)]
    if deltas.size == 0:
        return 2

    price_rel = np.concatenate(([0.0], np.cumsum(deltas)))
    max_up    = float(np.max(price_rel))
    max_down  = float(np.min(price_rel))

    # Zero crossings
    sign = np.sign(price_rel)
    last = 0.0
    for i in range(sign.size):
        if sign[i] == 0 and last != 0:
            sign[i] = last
        elif sign[i] != 0:
            last = sign[i]
    zero_crossings = int(np.sum(sign[:-1] * sign[1:] < 0))

    if zero_crossings > cfg.max_crossings:
        return 2  # muito choppy

    T = cfg.thresh_strong
    DR = cfg.thresh_dom_ratio
    OM = cfg.thresh_other_max

    # Regra 1: movimento forte limpo
    for p in price_rel:
        if p >= T:
            return 0   # buy
        if p <= -T:
            return 1   # sell

    # Regra 2: dominância de extremo
    abs_up   = max_up
    abs_down = abs(max_down)
    if abs_up >= T and abs_up >= DR * abs_down and abs_down <= OM:
        return 0
    if abs_down >= T and abs_down >= DR * abs_up and abs_up <= OM:
        return 1

    return 2  # hold


# ============================================================
# Par de arquivos: norm ↔ raw
# ============================================================

def _find_file_pairs(cfg: SFTBuildConfig) -> List[Tuple[str, Path, Path]]:
    """
    Associa cada arquivo *_norm_norm.npy ao seu raw *_wdo_dol.npy.
    Retorna lista de (day_key, norm_path, raw_path).
    """
    norm_files = sorted(Path(cfg.norm_dir).glob(cfg.norm_pattern))
    raw_suffix = cfg.raw_suffix   # e.g. "_wdo_dol.npy"

    pairs = []
    for nf in norm_files:
        # "20250314_wdo_dol_norm_norm.npy" → "20250314"
        name  = nf.name
        # Remove padrão de sufixo conhecidos até chegar na data
        base  = name.replace("_norm_norm.npy", "").replace("_norm.npy", "")
        day_key = base.split("_")[0]

        # Procura raw: mesmo diretório, mesmo prefixo de data
        raw_path = Path(cfg.raw_dir) / (day_key + raw_suffix)
        if not raw_path.exists():
            print(f"[WARN] Raw não encontrado para {nf.name}: {raw_path}")
            continue

        pairs.append((day_key, nf, raw_path))

    return pairs


# ============================================================
# Função principal de construção
# ============================================================

def build_sft_sequences(cfg: Optional[SFTBuildConfig] = None):
    if cfg is None:
        cfg = SFTBuildConfig()

    rng = random.Random(cfg.seed)
    pairs = _find_file_pairs(cfg)
    if not pairs:
        raise RuntimeError(
            f"Nenhum par norm/raw encontrado em {cfg.norm_dir} / {cfg.raw_dir}"
        )

    print(f"[BUILD SFT] {len(pairs)} dias encontrados.")

    all_labels:     List[int]  = []
    all_meta:       List[List[float]] = []
    all_day_idx:    List[int]  = []
    all_centers:    List[int]  = []
    day_keys:       List[str]  = []
    norm_file_paths: List[str] = []   # um caminho por dia (alinhado com day_keys)

    for day_idx, (day_key, norm_path, raw_path) in enumerate(pairs):
        norm_ev = np.load(norm_path).astype(np.float32)  # [L, input_dim]
        raw_ev  = np.load(raw_path).astype(np.float32)   # [L, input_dim]

        if norm_ev.ndim != 2 or raw_ev.ndim != 2:
            print(f"[WARN] {day_key}: shape inválido, pulando.")
            continue

        # Alinha comprimentos (raw e norm podem diferir por pequena margem)
        L = min(norm_ev.shape[0], raw_ev.shape[0])
        norm_ev = norm_ev[:L]
        raw_ev  = raw_ev[:L]

        n_past   = cfg.n_past
        n_future = cfg.n_future
        min_c    = n_past
        max_c    = L - n_future

        if max_c <= min_c:
            print(f"[WARN] {day_key}: dia muito curto ({L} eventos), pulando.")
            continue

        possible = list(range(min_c, max_c, cfg.stride))
        if cfg.max_per_day is not None and len(possible) > cfg.max_per_day:
            possible = sorted(rng.sample(possible, cfg.max_per_day))

        day_keys.append(day_key)
        norm_file_paths.append(str(norm_path))
        n_janelas = 0

        for c in possible:
            label    = compute_direction_label(raw_ev, c, c + n_future, cfg)
            meta_vec = _encode_meta(day_key, raw_ev, c)

            all_labels.append(label)
            all_meta.append(meta_vec)
            all_day_idx.append(len(day_keys) - 1)  # índice no array day_keys
            all_centers.append(c)
            n_janelas += 1

        print(f"  [{day_key}] {n_janelas} janelas geradas")

    if not all_labels:
        raise RuntimeError("Nenhuma janela de SFT foi gerada.")

    labels_t = torch.tensor(all_labels, dtype=torch.long)
    buy  = (labels_t == 0).sum().item()
    sell = (labels_t == 1).sum().item()
    hold = (labels_t == 2).sum().item()
    N    = len(labels_t)

    # Salva apenas labels, meta e índices — SEM os tensores de eventos.
    # O SFTDataset vai ler os arquivos .npy sob demanda no __getitem__.
    result = {
        "action_labels":   labels_t,
        "meta":            torch.tensor(all_meta, dtype=torch.float32),  # [N, num_meta]
        "day_indices":     torch.tensor(all_day_idx,  dtype=torch.long),
        "center_indices":  torch.tensor(all_centers,  dtype=torch.long),
        "day_keys":        day_keys,
        "norm_file_paths": norm_file_paths,  # um path por dia
        "cfg": {
            "n_past":    cfg.n_past,
            "n_future":  cfg.n_future,
            "input_dim": INPUT_DIM,
            "num_meta":  8,
            "n_classes": 3,
            "stride":    cfg.stride,
        },
    }

    print("=" * 60)
    print(f"[BUILD SFT] Total de janelas : {N}")
    print(f"[BUILD SFT] buy  = {buy}  ({100*buy/N:.1f}%)")
    print(f"[BUILD SFT] sell = {sell}  ({100*sell/N:.1f}%)")
    print(f"[BUILD SFT] hold = {hold}  ({100*hold/N:.1f}%)")
    print(f"[BUILD SFT] Arquivo .pt      : labels + meta + índices (sem eventos)")
    print("=" * 60)

    out = cfg.output_path
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, out)
    print(f"[BUILD SFT] Salvo em: {out}")
    return result


if __name__ == "__main__":
    build_sft_sequences()
