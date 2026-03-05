from __future__ import annotations

import torch
from torch.utils.data import Dataset


class AgentSequenceDataset(Dataset):
    """
    Lê o arquivo agent_sequences.pt e faz um split simples
    train/val na dimensão das janelas (N sequences).
    """

    def __init__(self, pt_path: str, split: str, val_ratio: float = 0.1, seed: int = 42):
        super().__init__()

        try:
            data = torch.load(pt_path, map_location="cpu", weights_only=False)
        except TypeError:
            data = torch.load(pt_path, map_location="cpu")

        if "sequences" not in data:
            raise KeyError(f"'sequences' não encontrado em {pt_path}")

        seqs = data["sequences"]  # [N, T]
        if seqs.dim() != 2:
            raise ValueError(f"Esperado tensor [N, T] em 'sequences', recebi {seqs.shape}")

        if seqs.dtype != torch.long:
            seqs = seqs.long()

        N = seqs.size(0)
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(N, generator=g)

        n_val = max(1, int(round(N * val_ratio)))
        if n_val >= N:
            n_val = max(1, N - 1)

        if split == "train":
            idx = perm[n_val:]
        elif split == "val":
            idx = perm[:n_val]
        else:
            raise ValueError(f"split inválido: {split}")

        self.seqs = seqs[idx]

    def __len__(self):
        return self.seqs.size(0)

    def __getitem__(self, idx):
        return self.seqs[idx]
