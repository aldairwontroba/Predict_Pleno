"""
Dataset de SFT para o ContinuousEventTransformer.

Carrega o arquivo gerado por build_sft_sequences.py (apenas labels + índices)
e lê os eventos normalizados sob demanda dos arquivos .npy por dia.

Estratégia de memória:
  - O .pt de índices é pequeno (MBs), independente do tamanho do histórico.
  - Os arquivos .npy de cada dia (~1MB cada) são carregados todos de uma vez
    no __init__ como lista de tensors. Para 1000 dias × 5000 eventos × 59
    features × 4 bytes ≈ 1.2 GB — manejável na máquina de treino.
  - No __getitem__ apenas fatia a janela certa: sem I/O extra.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class SFTDataset(Dataset):
    """
    Dataset de fine-tuning supervisionado.

    Cada item retorna:
      past_events  : [n_past, input_dim]  float32
      meta         : [num_meta]           float32
      action_label : scalar long          0=buy, 1=sell, 2=hold
    """

    def __init__(
        self,
        sequences_path:  str | Path,
        split:           str        = "train",
        val_ratio:       float      = 0.1,
        seed:            int        = 42,
        data_dir_override: str | Path | None = None,
    ):
        """
        data_dir_override: se fornecido, substitui o diretório dos arquivos .npy.
          Útil quando o .pt foi criado no Windows e está sendo usado no Linux
          (RunPod). Ex: data_dir_override="/workspace/data"
        """
        super().__init__()
        data = torch.load(sequences_path, map_location="cpu")

        self.action_labels  = data["action_labels"]    # [N] long
        self.meta           = data["meta"]              # [N, num_meta]
        self.day_indices    = data["day_indices"]       # [N] long
        self.center_indices = data["center_indices"]    # [N] long
        self.n_past         = data["cfg"]["n_past"]
        norm_file_paths: List[str] = data["norm_file_paths"]

        # Remapeia caminhos se data_dir_override foi fornecido
        if data_dir_override is not None:
            override_dir = Path(data_dir_override)
            remapped = []
            for p in norm_file_paths:
                fname = Path(p).name          # só o nome do arquivo
                remapped.append(str(override_dir / fname))
            norm_file_paths = remapped
            print(f"[SFTDataset] data_dir_override → {override_dir}")

        N = self.action_labels.shape[0]

        # Carrega todos os arquivos .npy de eventos em memória (um por dia).
        # Cada tensor tem shape [L_dia, input_dim].
        print(f"[SFTDataset] Carregando {len(norm_file_paths)} arquivos de eventos...")
        self.day_tensors: List[torch.Tensor] = []
        for p in norm_file_paths:
            arr = np.load(p).astype(np.float32)
            if not np.all(np.isfinite(arr)):
                arr = np.where(np.isfinite(arr), arr, 0.0)
            self.day_tensors.append(torch.from_numpy(arr))
        print(f"[SFTDataset] Eventos em memória: ok")

        # Split por índice (reproduzível)
        rng = random.Random(seed)
        indices = list(range(N))
        rng.shuffle(indices)
        n_val = max(1, int(N * val_ratio))
        val_set = set(indices[:n_val])

        self.indices: List[int] = (
            [i for i in range(N) if i not in val_set]
            if split == "train"
            else [i for i in range(N) if i in val_set]
        )

        labels = self.action_labels[self.indices]
        buy  = (labels == 0).sum().item()
        sell = (labels == 1).sum().item()
        hold = (labels == 2).sum().item()
        n    = len(self.indices)

        print("=" * 60)
        print(f"[SFTDataset] split={split}  |  {n} janelas")
        print(f"  buy={buy} ({100*buy/max(n,1):.1f}%)  "
              f"sell={sell} ({100*sell/max(n,1):.1f}%)  "
              f"hold={hold} ({100*hold/max(n,1):.1f}%)")
        print("=" * 60)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        real_idx = self.indices[idx]

        day_i  = int(self.day_indices[real_idx])
        center = int(self.center_indices[real_idx])

        # Fatia a janela de passado do tensor do dia (sem cópia extra)
        past_events = self.day_tensors[day_i][center - self.n_past : center]

        meta         = self.meta[real_idx]
        action_label = self.action_labels[real_idx]

        return past_events, meta, action_label
