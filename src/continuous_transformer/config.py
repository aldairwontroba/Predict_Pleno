"""
Configurações do Continuous Event Transformer (CET).

Abordagem: sem VQ-VAE, eventos normalizados entram diretamente no Transformer.
  - INPUT_DIM = número de features por evento (vem de FEATURE_ORDER)
  - Fase 1 (pretrain): prevê próximo evento, Huber loss
  - Fase 2 (SFT): classifica ação (buy/sell/hold) no último token do passado
"""
from __future__ import annotations

import os
from pathlib import Path

from src.normalization.normalize_manual import FEATURE_ORDER
from src.config import PATHS

# Dimensão de cada evento normalizado (infere do FEATURE_ORDER para não desincronizar)
INPUT_DIM: int = len(FEATURE_ORDER)  # 59

# Metadados por janela: [dow/6, tod/4, is_last_bd, pos_state, 0, 0, 0, 0]
NUM_META_FEATURES: int = 8

# Ações do agente: 0=buy, 1=sell, 2=hold
NUM_ACTIONS: int = 3
ACTION_NAMES = ["buy", "sell", "hold"]

# Diretório padrão dos dados nesta máquina de desenvolvimento.
# Na máquina de treino (GPU), sobrescreva via env var PLENO_EVENTOS_PROCESSADOS.
_DEFAULT_DATA_DIR = Path(os.getenv("PLENO_EVENTOS_PROCESSADOS",
                                   r"C:\Users\Aldair\OneDrive\MarketData"))


# ============================================================
# CONFIG DO MODELO
# ============================================================

class CTConfig:
    """Hiperparâmetros do ContinuousEventTransformer."""
    input_dim:   int   = INPUT_DIM
    num_meta:    int   = NUM_META_FEATURES
    d_model:     int   = 256
    n_heads:     int   = 8
    n_layers:    int   = 6
    seq_len:     int   = 512   # máximo de eventos na janela de contexto
    dropout:     float = 0.1
    num_actions: int   = NUM_ACTIONS


# ============================================================
# CONFIG DE PRÉ-TREINO
# ============================================================

class PretrainConfig:
    """Configurações do loop de pré-treino (next-event prediction)."""
    # Dados
    data_dir:   Path = _DEFAULT_DATA_DIR
    pattern:    str  = "*_norm_norm.npy"   # arquivos de input

    # Janela deslizante
    seq_len:    int  = CTConfig.seq_len    # tamanho da janela (T)
    stride:     int  = 1                  # passo entre janelas (1 = máximo de dados)

    # Treino
    batch_size: int   = 64
    accum_steps: int  = 4                 # batch efetivo = batch_size * accum_steps
    lr:         float = 3e-4
    num_epochs: int   = 20
    val_ratio:  float = 0.1
    max_grad_norm: float = 1.0
    num_workers: int  = 0                 # 0 = main process (seguro no Windows)
    seed:       int   = 42

    # Checkpoints
    checkpoint_dir: Path = PATHS.checkpoints / "ct_pretrain"
    log_interval:   int  = 50             # batches entre logs
    save_every:     int  = 3000           # salva pretrain_latest.pt a cada N batches

    # Resume: se True, auto-detecta o último checkpoint salvo e continua
    resume: bool = False


# ============================================================
# CONFIG DO SFT DATASET BUILDER
# ============================================================

class SFTBuildConfig:
    """Configurações para build_sft_sequences.py."""
    # Dados normalizados (input do modelo)
    norm_dir:   Path = _DEFAULT_DATA_DIR
    norm_pattern: str = "*_norm_norm.npy"

    # Dados crus (para calcular direction_label com thresholds reais em pontos)
    raw_dir:    Path = _DEFAULT_DATA_DIR
    raw_suffix: str  = "_wdo_dol.npy"    # raw: YYYYMMDD_wdo_dol.npy

    # Janela do agente
    n_past:     int  = 320               # eventos de passado que o modelo vê
    n_future:   int  = 64                # eventos futuros para calcular o label
    stride:     int  = 16                # passo entre janelas dentro de um dia
    max_per_day: int | None = None       # limite de janelas por dia (None = tudo)

    # Thresholds para direction_label (em pontos do ativo, nos dados crus)
    thresh_strong:    float = 5.0
    thresh_dom_ratio: float = 2.0
    thresh_other_max: float = 10.0
    max_crossings:    int   = 4

    # Coluna de preço delta nos dados crus (índice 1 = dp)
    price_col: int = 1
    # Coluna de volume total nos dados crus (índice 19 = t_vol)
    vol_col:   int = 19

    seed: int = 42

    # Saída
    output_path: Path = PATHS.artifacts / "ct_sft_sequences.pt"


# ============================================================
# CONFIG DO SFT
# ============================================================

class SFTConfig:
    """Configurações do loop de fine-tuning supervisionado."""
    # Dataset
    sequences_path: Path = PATHS.artifacts / "ct_sft_sequences.pt"

    # Checkpoint do pré-treino para inicializar (None = treino do zero)
    pretrain_ckpt: Path | None = PATHS.checkpoints / "ct_pretrain" / "pretrain_best.pt"

    # Treino
    batch_size:  int   = 64
    accum_steps: int   = 4
    lr:          float = 1e-4
    num_epochs:  int   = 10
    val_ratio:   float = 0.1
    max_grad_norm: float = 1.0
    num_workers: int   = 0
    seed:        int   = 42

    # Checkpoints
    checkpoint_dir: Path = PATHS.checkpoints / "ct_sft"
    log_interval:   int  = 100
