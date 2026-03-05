"""
Entry point: pré-treino do ContinuousEventTransformer.

Execução:
  python scripts/run_pretrain_ct.py

# ============================================================
# PERFIS DE CONFIGURAÇÃO
# ============================================================
#
#  DEV  (GTX 1060 6GB)                     ← validação local
#    seq_len=256, batch=16, accum=8, stride=16, epochs=3
#    ~250K janelas/época, ~18-20min por época
#    Objetivo: validar pipeline, ver loss cair
#
#  RUNPOD (RTX 4090 24GB)                  ← treino completo
#    seq_len=512, batch=64, accum=4, stride=1, epochs=10
#    ~8M janelas/época, ~2-3h por época → ~20-30h total (~$18)
#    Objetivo: modelo final de qualidade
#
#  Para trocar de perfil: comente o bloco ativo e descomente o outro
# ============================================================
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.continuous_transformer.config import CTConfig, PretrainConfig
from src.continuous_transformer.train_pretrain import train_pretrain

model_cfg = CTConfig()
train_cfg = PretrainConfig()

# ============================================================
# PERFIL DEV — GTX 1060 6GB  (ativo por padrão)
# ============================================================
# model_cfg.d_model  = 256
# model_cfg.n_heads  = 8
# model_cfg.n_layers = 4
# model_cfg.seq_len  = 256
# model_cfg.dropout  = 0.1
#
# train_cfg.seq_len     = model_cfg.seq_len
# train_cfg.num_epochs  = 3
# train_cfg.batch_size  = 16
# train_cfg.accum_steps = 8    # batch efetivo = 128
# train_cfg.lr          = 3e-4
# train_cfg.val_ratio   = 0.1
# train_cfg.stride      = 16
# train_cfg.num_workers = 0
# train_cfg.seed        = 42
# train_cfg.resume      = False

# ============================================================
# PERFIL RUNPOD — RTX 4090 24GB  ← ATIVO
# ============================================================
model_cfg.d_model  = 256
model_cfg.n_heads  = 8
model_cfg.n_layers = 6      # modelo completo
model_cfg.seq_len  = 512    # contexto completo
model_cfg.dropout  = 0.1

train_cfg.seq_len     = model_cfg.seq_len   # SEMPRE igual ao model_cfg.seq_len
train_cfg.num_epochs  = 10
train_cfg.batch_size  = 64
train_cfg.accum_steps = 4   # batch efetivo = 256
train_cfg.lr          = 3e-4
train_cfg.val_ratio   = 0.1
train_cfg.stride      = 1   # usa todos os dados
train_cfg.num_workers = 4
train_cfg.seed        = 42
train_cfg.resume      = True  # retoma automaticamente se interrompido

# Diretório dos dados (env var substitui o padrão do Windows)
# No RunPod: export PLENO_EVENTOS_PROCESSADOS=/workspace/data
# (já configurado em setup_runpod.sh)

# ============================================================
if __name__ == "__main__":
    train_pretrain(model_cfg, train_cfg)
