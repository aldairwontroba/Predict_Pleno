"""
Dataset de pré-treino: next-event prediction.

Carrega arquivos *_norm_norm.npy (eventos normalizados) e cria janelas
deslizantes de comprimento seq_len+1:
  x = events[t : t+seq_len]        # [seq_len, input_dim]  → input
  y = events[t+1 : t+seq_len+1]    # [seq_len, input_dim]  → target

Isso é o análogo direto do MarketTokenDatasetPacked (src/agent/model.py),
mas para espaço contínuo em vez de tokens discretos.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.continuous_transformer.config import PretrainConfig


def _load_npy_float32(path: Path) -> Optional[torch.Tensor]:
    """Carrega .npy, valida shape 2D e converte para float32 Tensor."""
    arr = np.load(path)
    if arr.ndim != 2:
        print(f"[WARN] {path.name}: shape {arr.shape} não é 2D, pulando.")
        return None
    # substitui não-finitos por zero (segurança)
    if not np.all(np.isfinite(arr)):
        arr = np.where(np.isfinite(arr), arr, 0.0)
    return torch.from_numpy(arr.astype(np.float32))


class PretrainDataset(Dataset):
    """
    Dataset para pré-treino do CET (next-event prediction).

    Parâmetros
    ----------
    data_dir : diretório com os arquivos *_norm_norm.npy
    seq_len  : comprimento da janela de contexto (T)
    pattern  : glob pattern para encontrar os arquivos
    val_ratio: fração dos arquivos reservada para validação
    split    : 'train' ou 'val'
    seed     : seed para o split
    stride   : passo entre janelas (1 = máximo de dados, >1 reduz dataset)
    """

    def __init__(
        self,
        data_dir: str | Path,
        seq_len:  int,
        pattern:  str = "*_norm_norm.npy",
        val_ratio: float = 0.1,
        split:    str = "train",
        seed:     int = 42,
        stride:   int = 1,
    ):
        super().__init__()
        self.seq_len  = seq_len
        self.raw_win  = seq_len + 1   # x=[:seq_len], y=[1:seq_len+1]

        files = sorted(Path(data_dir).glob(pattern))
        if not files:
            raise FileNotFoundError(
                f"Nenhum arquivo '{pattern}' em {data_dir}"
            )

        # Split por arquivo (não mistura dias entre treino/val)
        rng = random.Random(seed)
        files_shuffled = files[:]
        rng.shuffle(files_shuffled)
        n_val = max(1, int(len(files_shuffled) * val_ratio))
        val_files   = set(str(f) for f in files_shuffled[:n_val])
        train_files = [f for f in files_shuffled if str(f) not in val_files]
        val_files_l = [f for f in files_shuffled if str(f) in val_files]

        selected = train_files if split == "train" else val_files_l

        # Carrega todos os arquivos selecionados em memória
        self.tensors: List[torch.Tensor] = []
        for f in selected:
            t = _load_npy_float32(Path(f))
            if t is None or t.shape[0] < self.raw_win:
                continue
            self.tensors.append(t)

        # Índice: (tensor_idx, start_pos)
        self.index_map: List[Tuple[int, int]] = []
        for i, t in enumerate(self.tensors):
            L = t.shape[0]
            for start in range(0, L - self.raw_win + 1, stride):
                self.index_map.append((i, start))

        print("=" * 60)
        print(f"[PretrainDataset] split={split}")
        print(f"[PretrainDataset] arquivos carregados : {len(self.tensors)}")
        print(f"[PretrainDataset] janelas             : {len(self.index_map)}")
        print(f"[PretrainDataset] seq_len             : {seq_len}")
        print(f"[PretrainDataset] stride              : {stride}")
        print("=" * 60)

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ti, start = self.index_map[idx]
        window = self.tensors[ti][start : start + self.raw_win]  # [raw_win, input_dim]
        x = window[:-1]   # [seq_len, input_dim]
        y = window[1:]    # [seq_len, input_dim]
        return x, y
