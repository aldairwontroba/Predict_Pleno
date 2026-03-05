"""
Entry point: SFT do ContinuousEventTransformer.

Passo 0 (uma vez): construir o dataset de SFT
  python -m src.continuous_transformer.build_sft_sequences

Passo 1: rodar o SFT
  python scripts/run_sft_ct.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.continuous_transformer.config import CTConfig, SFTConfig
from src.continuous_transformer.train_sft import train_sft

# ============================================================
# CONFIGURE AQUI
# ============================================================

model_cfg = CTConfig()
sft_cfg   = SFTConfig()

# ============================================================
# PERFIL DEV — GTX 1060 6GB  (comentado)
# ============================================================
# model_cfg.d_model  = 256
# model_cfg.n_heads  = 8
# model_cfg.n_layers = 4      # deve bater com o checkpoint do pré-treino
# model_cfg.seq_len  = 256    # deve bater com o checkpoint do pré-treino
# model_cfg.dropout  = 0.1
#
# sft_cfg.num_epochs  = 5
# sft_cfg.batch_size  = 32
# sft_cfg.accum_steps = 4
# sft_cfg.lr          = 1e-4
# sft_cfg.val_ratio   = 0.1
# sft_cfg.num_workers = 0
# sft_cfg.seed        = 42

# ============================================================
# PERFIL RUNPOD — RTX 4090 24GB  ← ATIVO
# ============================================================
model_cfg.d_model  = 256
model_cfg.n_heads  = 8
model_cfg.n_layers = 6      # deve bater com o pretrain RUNPOD
model_cfg.seq_len  = 512    # deve bater com o pretrain RUNPOD
model_cfg.dropout  = 0.1

sft_cfg.num_epochs  = 10
sft_cfg.batch_size  = 128
sft_cfg.accum_steps = 2    # batch efetivo = 256
sft_cfg.lr          = 1e-4
sft_cfg.val_ratio   = 0.1
sft_cfg.num_workers = 4
sft_cfg.seed        = 42

# Diretório dos dados (para remapear caminhos Windows → Linux)
# No RunPod, os caminhos dentro do ct_sft_sequences.pt são Windows.
# DATA_DIR_OVERRIDE aponta para onde os arquivos *_norm_norm.npy estão no pod.
DATA_DIR_OVERRIDE = "/workspace/data"  # None = usa caminhos originais do .pt

# ============================================================
if __name__ == "__main__":
    train_sft(model_cfg, sft_cfg, data_dir_override=DATA_DIR_OVERRIDE)
