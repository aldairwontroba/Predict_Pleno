"""
Diagnóstico do pré-treino: compara previsões do modelo com valores reais.

Carrega o checkpoint pretrain_best.pt, pega um arquivo .npy aleatório,
sorteia algumas posições e imprime real vs previsto para todos os 59 features.

Uso:
  python scripts/inspect_pretrain.py
"""
import sys, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.continuous_transformer.config import CTConfig, PretrainConfig
from src.continuous_transformer.model import ContinuousEventTransformer
from src.normalization.normalize_manual import FEATURE_ORDER

# ============================================================
# CONFIG
# ============================================================
CKPT_PATH   = PretrainConfig.checkpoint_dir / "pretrain_best.pt"
DATA_DIR    = PretrainConfig.data_dir
N_POSITIONS = 5      # quantas posições aleatórias mostrar
SEQ_LEN     = 256    # deve bater com o checkpoint (dev=256, full=512)
SEED        = 7

# ============================================================
# Carrega modelo
# ============================================================
print(f"Carregando checkpoint: {CKPT_PATH}")
ckpt = torch.load(CKPT_PATH, map_location="cpu")
cfg_dict = ckpt.get("cfg", {})

cfg = CTConfig()
for k, v in cfg_dict.items():
    if hasattr(cfg, k):
        setattr(cfg, k, v)

model = ContinuousEventTransformer(cfg)
model.load_state_dict(ckpt["model_state"])
model.eval()
print(f"  d_model={cfg.d_model}  n_layers={cfg.n_layers}  seq_len={cfg.seq_len}")
print(f"  val_loss no checkpoint: {ckpt.get('val_loss', 'n/a'):.5f}\n")

# ============================================================
# Carrega um arquivo aleatório
# ============================================================
rng = random.Random(SEED)
files = sorted(Path(DATA_DIR).glob("*_norm_norm.npy"))
chosen = rng.choice(files)
print(f"Arquivo escolhido: {chosen.name}")

arr = np.load(chosen).astype(np.float32)   # [L, 59]
L   = arr.shape[0]
print(f"Eventos no dia   : {L}\n")

# ============================================================
# Sorteia posições onde o modelo vai prever o próximo evento
# ============================================================
# Para posição p: input = arr[p-SEQ_LEN : p], target = arr[p]
min_p = SEQ_LEN
max_p = L - 1
positions = sorted(rng.sample(range(min_p, max_p), N_POSITIONS))

# ============================================================
# Roda inferência e imprime
# ============================================================
FEAT_NAMES = list(FEATURE_ORDER)
COL_W = 22   # largura da coluna de nome

with torch.no_grad():
    for pos in positions:
        x      = torch.from_numpy(arr[pos - SEQ_LEN : pos]).unsqueeze(0)  # [1, SEQ_LEN, 59]
        out    = model(x)
        pred   = out["next_event"][0, -1, :].numpy()   # previsão do último token → próximo evento
        real   = arr[pos]                               # valor real do próximo evento
        mae    = float(np.abs(pred - real).mean())
        var_p  = float(pred.var())
        var_r  = float(real.var())

        print("=" * 75)
        print(f"Posição {pos:5d}  |  MAE={mae:.4f}  |  var_pred={var_p:.4f}  var_real={var_r:.4f}")
        print(f"  {'Feature':<{COL_W}}  {'Real':>8}  {'Previsto':>8}  {'Erro':>8}")
        print(f"  {'-'*COL_W}  {'-'*8}  {'-'*8}  {'-'*8}")

        for i, name in enumerate(FEAT_NAMES):
            r = float(real[i])
            p = float(pred[i])
            e = abs(r - p)
            # marca features com erro alto
            flag = " !" if e > 0.5 else ""
            print(f"  {name:<{COL_W}}  {r:>8.4f}  {p:>8.4f}  {e:>8.4f}{flag}")

        print()

# ============================================================
# Teste de colapso: varia o input e vê se a saída muda
# ============================================================
print("=" * 75)
print("TESTE DE COLAPSO: mesma posição, inputs diferentes")
print("(se todas as previsões forem iguais, o modelo colapsou)\n")

p = positions[0]
base_input = arr[p - SEQ_LEN : p].copy()

for shift in [0, 10, 50, 100]:
    shifted_pos = min(p + shift, L - 1)
    inp = torch.from_numpy(arr[shifted_pos - SEQ_LEN : shifted_pos]).unsqueeze(0)
    with torch.no_grad():
        pred_shifted = model(inp)["next_event"][0, -1, :].numpy()

    # compara com pred da posição base
    with torch.no_grad():
        pred_base = model(
            torch.from_numpy(base_input).unsqueeze(0)
        )["next_event"][0, -1, :].numpy()

    diff = float(np.abs(pred_shifted - pred_base).mean())
    print(f"  shift={shift:3d} eventos  |  MAE vs base={diff:.4f}  "
          f"{'(IGUAL — possível colapso!)' if diff < 1e-4 else '(OK — output varia)'}")

print()
print("Diagnóstico concluído.")
