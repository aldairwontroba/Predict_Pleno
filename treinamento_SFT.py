"""
Treino SFT do agente em cima de agent_sequences.pt

Tarefa: next-token prediction em janelas de 512 tokens que já incluem:
- contexto + estado,
- passado de mercado,
- futuro real (via VQ),
- tokens de análise,
- tokens de ação.

Configuração é feita diretamente no bloco MAIN, sem argparse.
"""

import os
from dataclasses import dataclass
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# ============================================================
# Ajuste estes imports para o seu projeto
# ============================================================

# Exemplo: se você tiver tudo em agent_transformer.py:
# from agent_transformer import AgentTransformer, TransformerConfig, TokenConfig

from agent_transformer import AgentTransformer, TransformerConfig, TokenConfig


# ============================================================
# Dataset em cima de agent_sequences.pt
# ============================================================
class AgentWindowConfig:
    pass

class AgentSequenceDataset(Dataset):
    """
    Lê o arquivo agent_sequences.pt e faz um split simples
    train/val na dimensão das janelas (N sequences).
    """

    def __init__(self, pt_path: str, split: str, val_ratio: float = 0.1, seed: int = 42):
        super().__init__()

        # Esse .pt é um dicionário de dados (não só pesos)
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



# ============================================================
# Configuração de treino
# ============================================================

@dataclass
class AgentTrainConfig:
    batch_size_micro: int = 64
    accum_steps: int = 4          # batch efetivo = batch_size_micro * accum_steps
    num_epochs: int = 3
    lr: float = 3e-4
    val_ratio: float = 0.1
    num_workers: int = 4
    log_interval: int = 100       # batches
    max_grad_norm: float = 1.0    # clip de gradiente


# ============================================================
# Loop de avaliação
# ============================================================

@torch.no_grad()
def evaluate_agent_stf(model, dl_val, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    use_amp = (device.type == "cuda")
    with torch.cuda.amp.autocast(enabled=use_amp):
        for x in dl_val:
            x = x.to(device, non_blocking=True)   # [B, T]
            inp  = x[:, :-1]                      # [B, T-1]
            targ = x[:,  1:]                      # [B, T-1]

            logits = model(inp)                   # [B, T-1, V] (esperado)
            # garante alinhamento:
            if logits.size(1) > targ.size(1):
                logits = logits[:, :targ.size(1), :]
            elif logits.size(1) < targ.size(1):
                targ = targ[:, :logits.size(1)]

            B, Tm1, V = logits.shape
            loss = F.cross_entropy(
                logits.reshape(B * Tm1, V),
                targ.reshape(B * Tm1),
                reduction="sum",
            )
            total_loss += loss.item()
            total_tokens += B * Tm1

    return total_loss / max(1, total_tokens)


# ============================================================
# Loop de treino principal
# ============================================================

def train_agent_stf(
    agent_pt_path: str,
    model: torch.nn.Module,
    cfg: AgentTrainConfig,
    out_dir: str = "checkpoints_agent_stf",
    seed: int = 42,
):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = next(model.parameters()).device
    os.makedirs(out_dir, exist_ok=True)

    ds_train = AgentSequenceDataset(agent_pt_path, split="train", val_ratio=cfg.val_ratio, seed=seed)
    ds_val   = AgentSequenceDataset(agent_pt_path, split="val",   val_ratio=cfg.val_ratio, seed=seed)

    dl_train = DataLoader(
        ds_train,
        batch_size=cfg.batch_size_micro,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=cfg.batch_size_micro,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    # --- inferir vocab_size do modelo ---
    vocab_size = None
    if hasattr(model, "token_emb"):
        vocab_size = model.token_emb.num_embeddings
    elif hasattr(model, "market_head"):
        vocab_size = model.market_head.out_features

    print("============================================================")
    print(f"[STF] device            : {device}")
    print(f"[STF] sequences train   : {len(ds_train)}")
    print(f"[STF] sequences val     : {len(ds_val)}")
    print(f"[STF] batch_size_micro  : {cfg.batch_size_micro}")
    print(f"[STF] accum_steps       : {cfg.accum_steps}")
    print(f"[STF] batch efetivo     : {cfg.batch_size_micro * cfg.accum_steps}")
    if vocab_size is not None:
        print(f"[STF] vocab_size(model) : {vocab_size}")
    print("============================================================")

    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(1, cfg.num_epochs + 1):
        t0 = datetime.now()
        model.train()
        running_loss = 0.0
        accum_counter = 0

        print(f"\n========== EPOCH {epoch:03d}/{cfg.num_epochs:03d} | START {t0} ==========")

        opt.zero_grad(set_to_none=True)

        for i, x in enumerate(dl_train, start=1):
            x = x.to(device, non_blocking=True)  # [B, T]

            # --------- CHECAGEM CRÍTICA AQUI ---------
            if vocab_size is not None:
                x_min = int(x.min().item())
                x_max = int(x.max().item())
                if x_min < 0 or x_max >= vocab_size:
                    raise RuntimeError(
                        f"[ERRO TOKENS] Batch {i}: valores fora do range da embedding.\n"
                        f"    shape x     = {tuple(x.shape)}\n"
                        f"    min token   = {x_min}\n"
                        f"    max token   = {x_max}\n"
                        f"    vocab_size  = {vocab_size}\n"
                        f"Isto vai explodir em self.token_emb(x). "
                        f"Seu dataset tem ids que não cabem no vocab do modelo."
                    )

            inp  = x[:, :-1]                     # [B, T-1]
            targ = x[:,  1:]                     # [B, T-1]

            logits = model(inp)                  # [B, T-1, V]  (assumindo que forward já retorna só o head de mercado)

            # garante que tempo de logits e targ batem
            if logits.size(1) > targ.size(1):
                logits = logits[:, :targ.size(1), :]
            elif logits.size(1) < targ.size(1):
                targ = targ[:, :logits.size(1)]

            B, Tm1, V = logits.shape

            loss = F.cross_entropy(
                logits.reshape(B * Tm1, V),
                targ.reshape(B * Tm1),
                reduction="mean",
            )

            # acumulação de gradiente
            loss = loss / cfg.accum_steps
            loss.backward()

            accum_counter += 1
            running_loss += loss.item() * cfg.accum_steps  # volta pro valor "real" para logging

            if accum_counter == cfg.accum_steps:
                # opcional: clip de gradiente para estabilizar
                if cfg.max_grad_norm is not None and cfg.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)

                opt.step()
                opt.zero_grad(set_to_none=True)

                accum_counter = 0
                global_step += 1

            # logging periódico
            if (i % cfg.log_interval) == 0:
                mean_loss = running_loss / cfg.log_interval
                ppl = torch.exp(torch.tensor(mean_loss)).item()
                now = datetime.now().strftime("%H:%M:%S")
                print(
                    f"[{now}] [EPOCH {epoch:03d}] [BATCH {i:06d}/{len(dl_train):06d}] "
                    f"train_loss_medio={mean_loss:.4f} train_ppl={ppl:.2f}"
                )
                running_loss = 0.0

        # se sobrou gradiente acumulado não aplicado
        if accum_counter > 0:
            if cfg.max_grad_norm is not None and cfg.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            opt.step()
            opt.zero_grad(set_to_none=True)

        # --- validação ---
        val_loss = evaluate_agent_stf(model, dl_val, device)
        val_ppl = torch.exp(torch.tensor(val_loss)).item()
        dt = (datetime.now() - t0).total_seconds() / 60.0

        print("------------------------------------------------------------")
        print(
            f"[{datetime.now()}] [EPOCH {epoch:03d}/{cfg.num_epochs:03d}] "
            f"val_loss={val_loss:.4f} val_ppl={val_ppl:.2f} | epoch_time={dt:.2f} min"
        )
        print("------------------------------------------------------------")

        # checkpoint por época
        ckpt_path = os.path.join(out_dir, f"stf_epoch_{epoch:03d}.pt")
        torch.save(
            {
                "model_state": model.state_dict(),
                "train_cfg": cfg.__dict__,
                "epoch": epoch,
                "val_loss": val_loss,
            },
            ckpt_path,
        )
        print(f"[CHECKPOINT] salvo: {ckpt_path}")

        # melhor modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(out_dir, "stf_best.pt")
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "train_cfg": cfg.__dict__,
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                best_path,
            )
            print(f"[CHECKPOINT] melhor modelo atualizado: {best_path}")


# ============================================================
# MAIN – sem argparse, tudo configurado aqui
# ============================================================

def main():
    # -------------------------------
    # CONFIGURAÇÕES DO TREINO
    # -------------------------------

    # caminho do dataset de janelas de agente
    AGENT_SEQ_PATH = "tokens_out/agent_sequences.pt"

    # diretório para salvar checkpoints de SFT
    OUT_DIR = "checkpoints_agent_stf"

    # checkpoint inicial do modelo pré-treinado (LM puro)
    # deixe como None para treinar SFT do zero,
    # ou coloque algo como "checkpoints_agent_tiny/agent_best.pt"
    # INIT_CKPT = None
    INIT_CKPT = "checkpoints_agent_tiny/agent_best.pt"

    # hiperparâmetros de treino
    N_EPOCHS = 3
    BATCH_SIZE_MICRO = 32
    ACCUM_STEPS = 4
    LR = 3e-4
    VAL_RATIO = 0.1
    NUM_WORKERS = 4
    SEED = 42

    # dispositivo
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("[WARN] CUDA não disponível, usando CPU.")

    # -------------------------------
    # CRIA CONFIG DO MODELO
    # -------------------------------
    # Se TransformerConfig for uma classe com atributos de classe,
    # isso ainda funciona; se for dataclass, melhor ainda.
    model_cfg = TransformerConfig()

    print("============================================================")
    print(f"[MODEL] dim       = {model_cfg.dim}")
    print(f"[MODEL] n_heads   = {model_cfg.n_heads}")
    print(f"[MODEL] n_layers  = {model_cfg.n_layers}")
    print(f"[MODEL] seq_len   = {model_cfg.seq_len}")
    print(f"[MODEL] vocab     = {TokenConfig.FINAL_VOCAB_SIZE}")
    print("============================================================")

    # -------------------------------
    # CRIA MODELO CORRETAMENTE
    # -------------------------------
    model = AgentTransformer(
        cfg=model_cfg,
        vocab_size=TokenConfig.FINAL_VOCAB_SIZE,
    ).to(device)

    # carregar checkpoint inicial (pré-treino LM) se fornecido
    if INIT_CKPT is not None:
        if os.path.isfile(INIT_CKPT):
            print(f"[INIT] carregando checkpoint inicial de: {INIT_CKPT}")
            ckpt = torch.load(INIT_CKPT, map_location="cpu")
            state = ckpt.get("model_state") or ckpt.get("model") or ckpt
            model.load_state_dict(state, strict=False)
        else:
            print(f"[WARN] INIT_CKPT '{INIT_CKPT}' não encontrado. Treinando SFT do zero.")

    # cfg de treino
    cfg = AgentTrainConfig(
        batch_size_micro=BATCH_SIZE_MICRO,
        accum_steps=ACCUM_STEPS,
        num_epochs=N_EPOCHS,
        lr=LR,
        val_ratio=VAL_RATIO,
        num_workers=NUM_WORKERS,
    )

    # roda treino
    train_agent_stf(
        agent_pt_path=AGENT_SEQ_PATH,
        model=model,
        cfg=cfg,
        out_dir=OUT_DIR,
        seed=SEED,
    )


if __name__ == "__main__":
    main()
