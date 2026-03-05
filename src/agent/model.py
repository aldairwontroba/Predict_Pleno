import os
import math
import random
from typing import List, Optional
import time
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from src.config import PATHS


# =========================================================
# CONFIGURAÇÕES DO VOCAB E TOKENS RESERVADOS
# =========================================================

class TokenConfig:
    # tokens do VQ-VAE
    VQ_TOKENS = 4096

    # reservas futuras
    NUM_ACTION_TOKENS = 16
    NUM_CONTEXT_TOKENS = 64
    NUM_AUX_TOKENS = 32

    VOCAB_SIZE = (
        VQ_TOKENS +
        NUM_ACTION_TOKENS +
        NUM_CONTEXT_TOKENS +
        NUM_AUX_TOKENS
    )

    RANGE_VQ = (0, VQ_TOKENS - 1)
    RANGE_ACTION = (VQ_TOKENS, VQ_TOKENS + NUM_ACTION_TOKENS - 1)
    RANGE_CONTEXT = (RANGE_ACTION[1] + 1, RANGE_ACTION[1] + NUM_CONTEXT_TOKENS)
    RANGE_AUX = (RANGE_CONTEXT[1] + 1, VOCAB_SIZE - 1)

    # token extra só para padding (se/quando você quiser usar)
    PAD_TOKEN = VOCAB_SIZE
    FINAL_VOCAB_SIZE = VOCAB_SIZE + 1  # inclui PAD_TOKEN


# =========================================================
# CONFIG DO TRANSFORMER
# =========================================================

class TransformerConfig:
    # MODELO MENOR
    dim = 128          # antes era 256
    n_heads = 4        # 4 cabeças ainda é ok
    n_layers = 4       # menos camadas
    seq_len = 512      # contexto menor → MUITO mais rápido
    dropout = 0.1

# =========================================================
# DATASET – LENDO tokens_by_day.pt
# =========================================================

class MarketTokenDatasetPacked(Dataset):
    """
    Dataset em cima do dicionário salvo em tokens_by_day.pt:

    {
        "day_keys": [...],
        "tokens_by_day": [tensor_1d, tensor_1d, ...],
        "meta": {...}
    }

    - Cada entrada de tokens_by_day é um dia (não mistura dias).
    - Usa janelas de comprimento raw_window = context_len + 1.
      x = window[:-1], y = window[1:], ambos com tamanho context_len.
    - Apenas usa dias com pelo menos raw_window tokens.
    - day_ids: índice(s) dos dias a incluir (sobre o vetor original).
    """

    def __init__(self, data: dict, context_len: int, day_ids: Optional[List[int]] = None):
        super().__init__()

        self.context_len = context_len
        self.raw_window = context_len + 1  # tamanho da janela "bruta"

        self.day_keys: List[str] = data["day_keys"]
        all_days: List[torch.Tensor] = data["tokens_by_day"]

        if day_ids is None:
            day_ids = list(range(len(all_days)))

        self.day_ids: List[int] = []   # ids originais
        self.days: List[torch.Tensor] = []

        # Seleciona e filtra dias muito curtos
        for day_id in day_ids:
            toks = all_days[day_id].view(-1).long()
            if toks.numel() < self.raw_window:
                continue
            self.day_ids.append(day_id)
            self.days.append(toks)

        # Mapeia janelas: (local_day_idx, start_idx)
        self.index_map: List[tuple] = []
        for local_idx, toks in enumerate(self.days):
            L = toks.numel()
            for start in range(0, L - self.raw_window + 1):
                self.index_map.append((local_idx, start))

        print("============================================================")
        print("[DATASET] context_len      :", self.context_len)
        print("[DATASET] raw_window       :", self.raw_window)
        print("[DATASET] dias selecionados:", len(self.day_ids))
        print("[DATASET] janelas totais   :", len(self.index_map))
        if len(self.index_map) == 0:
            print("[WARN] Dataset ficou vazio! Verifique seq_len ou dados.")
        print("============================================================")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        local_idx, start = self.index_map[idx]
        tokens = self.days[local_idx]

        window = tokens[start:start + self.raw_window]  # [raw_window]
        assert window.numel() == self.raw_window

        x = window[:-1]   # [context_len]
        y = window[1:]    # [context_len]

        return x, y


# =========================================================
# TRANSFORMER
# =========================================================

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, n_heads, dropout=0.1):
        super().__init__()
        assert dim % n_heads == 0

        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        B, T, C = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if mask is not None:
            att = att.masked_fill(mask == 0, float("-inf"))

        att = torch.softmax(att, dim=-1)
        att = self.dropout(att)

        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.dim)
        self.attn = MultiHeadSelfAttention(cfg.dim, cfg.n_heads, cfg.dropout)
        self.ln2 = nn.LayerNorm(cfg.dim)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.dim, 4 * cfg.dim),
            nn.GELU(),
            nn.Linear(4 * cfg.dim, cfg.dim),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x, mask):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return x


class AgentTransformer(nn.Module):
    """
    Modelo Transformer para pré-treino:
    - Embedding para todos os tokens (VQ + reservados + PAD).
    - Cabeça de mercado (previsão de próximo token).
    - Cabeça de ação já preparada para futuro RL (não usada no pré-treino).
    """
    def __init__(self, cfg: TransformerConfig, vocab_size: int):
        super().__init__()
        self.cfg = cfg

        self.token_emb = nn.Embedding(vocab_size, cfg.dim)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.dim)

        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )

        self.ln_final = nn.LayerNorm(cfg.dim)

        # Previsão de próximo token de mercado
        self.market_head = nn.Linear(cfg.dim, vocab_size)

        # Futuro RL: logits de ação (não usados no pré-treino)
        self.action_head = nn.Linear(cfg.dim, TokenConfig.NUM_ACTION_TOKENS)

    def forward(self, x):
        """
        x: [B, T] com T = cfg.seq_len
        retorna: logits de mercado [B, T, vocab_size]
        """
        B, T = x.shape
        assert T <= self.cfg.seq_len, "comprimento da sequência > cfg.seq_len"

        pos = torch.arange(T, device=x.device).unsqueeze(0)  # [1, T]

        h = self.token_emb(x) + self.pos_emb(pos)  # [B, T, C]

        # máscara causal [1, 1, T, T]
        mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)

        for blk in self.blocks:
            h = blk(h, mask)

        h = self.ln_final(h)

        logits = self.market_head(h)  # [B, T, vocab]
        return logits


# =========================================================
# TREINAMENTO PROFISSIONALIZADO
# =========================================================


def train_agent_transformer(
    packed_path: str = "tokens_out/tokens_by_day.pt",
    checkpoint_dir: str = str(PATHS.checkpoints / "checkpoints_agent"),
    num_epochs: int = 10,
    batch_size: int = 16,        # micro-batch
    val_ratio: float = 0.1,
    lr: float = 2e-4,
    num_workers: int = 0,
    seed: int = 42,
    accum_steps: int = 8,        # grad accumulation
    max_train_batches: int | None = None,  # limite opcional por época
    max_val_batches: int | None = None,    # idem p/ validação
):
    # -- seeds
    torch.manual_seed(seed)
    random.seed(seed)

    # -- carrega dados
    data = torch.load(packed_path, map_location="cpu")
    day_keys: List[str] = data["day_keys"]
    # número total de dias
    num_days = len(day_keys)

    # quantos dias vão para validação (mesma lógica de antes)
    val_days = max(1, int(round(num_days * val_ratio)))
    if val_days >= num_days:
        val_days = max(1, num_days - 1)

    # todos os índices de dias
    all_day_ids = list(range(num_days))

    # sorteia dias aleatórios para validação (sem reposição)
    val_day_ids = sorted(random.sample(all_day_ids, val_days))

    # o restante vai para treino
    train_day_ids = [i for i in all_day_ids if i not in val_day_ids]

    print("============================================================")
    print(f"[SPLIT] num_days      : {num_days}")
    print(f"[SPLIT] train_day_ids : {train_day_ids[:10]}{' ...' if len(train_day_ids) > 10 else ''}")
    print("============================================================")

    # -- modelo
    cfg = TransformerConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("============================================================")
    print(f"[INFO] device           : {device}")
    print(f"[INFO] context_len      : {cfg.seq_len}")
    print(f"[INFO] dim              : {cfg.dim}")
    print(f"[INFO] n_layers         : {cfg.n_layers}")
    print(f"[INFO] n_heads          : {cfg.n_heads}")
    print(f"[INFO] batch_size (micro): {batch_size}")
    print(f"[INFO] accum_steps      : {accum_steps}")
    print(f"[INFO] batch efetivo    : {batch_size * accum_steps}")
    print("============================================================")

    model = AgentTransformer(cfg, TokenConfig.FINAL_VOCAB_SIZE).to(device)

    # -- datasets / loaders
    train_ds = MarketTokenDatasetPacked(
        data=data,
        context_len=cfg.seq_len,
        day_ids=train_day_ids,
    )
    val_ds = MarketTokenDatasetPacked(
        data=data,
        context_len=cfg.seq_len,
        day_ids=val_day_ids,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        drop_last=False,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_loss = float("inf")

    # =====================================================
    # LOOP DE TREINAMENTO
    # =====================================================
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n========== EPOCH {epoch:03d}/{num_epochs:03d} | START {now_str} ==========")

        # --------------------------
        # TREINO
        # --------------------------
        model.train()
        total_train_loss = 0.0
        total_train_tokens = 0

        optimizer.zero_grad()
        num_train_batches = len(train_loader)
        if max_train_batches is not None:
            num_train_batches = min(num_train_batches, max_train_batches)

        for batch_idx, (x, y) in enumerate(train_loader):
            if max_train_batches is not None and batch_idx >= max_train_batches:
                break

            x = x.to(device)  # [B, T]
            y = y.to(device)  # [B, T]
            B, T = x.shape

            logits = model(x)  # [B, T, V]
            V = logits.size(-1)

            raw_loss = loss_fn(
                logits.view(B * T, V),
                y.view(B * T),
            )

            total_train_loss += raw_loss.item() * (B * T)
            total_train_tokens += (B * T)

            loss = raw_loss / accum_steps
            loss.backward()

            # step a cada accum_steps
            if ((batch_idx + 1) % accum_steps == 0) or ((batch_idx + 1) == num_train_batches):
                optimizer.step()
                optimizer.zero_grad()

            # logs mais espaçados para não poluir
            if (batch_idx + 1) % 1000 == 0 or (batch_idx + 1) == num_train_batches:
                avg_loss = total_train_loss / max(total_train_tokens, 1)
                now_str = datetime.now().strftime("%H:%M:%S")
                print(
                    f"[{now_str}] [EPOCH {epoch:03d}] "
                    f"[BATCH {batch_idx + 1:06d}/{num_train_batches:06d}] "
                    f"train_loss_medio={avg_loss:.4f}"
                )

        train_loss = total_train_loss / max(total_train_tokens, 1)
        train_ppl = math.exp(min(train_loss, 20.0))

        # --------------------------
        # VALIDAÇÃO
        # --------------------------
        model.eval()
        total_val_loss = 0.0
        total_val_tokens = 0

        num_val_batches = len(val_loader)
        if max_val_batches is not None:
            num_val_batches = min(num_val_batches, max_val_batches)

        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(val_loader):
                if max_val_batches is not None and batch_idx >= max_val_batches:
                    break

                x = x.to(device)
                y = y.to(device)
                B, T = x.shape

                logits = model(x)
                V = logits.size(-1)

                val_loss_batch = loss_fn(
                    logits.view(B * T, V),
                    y.view(B * T),
                )

                total_val_loss += val_loss_batch.item() * (B * T)
                total_val_tokens += (B * T)

        val_loss = total_val_loss / max(total_val_tokens, 1)
        val_ppl = math.exp(min(val_loss, 20.0))

        epoch_time = time.time() - epoch_start
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("------------------------------------------------------------")
        print(
            f"[{now_str}] [EPOCH {epoch:03d}/{num_epochs:03d}] "
            f"train_loss={train_loss:.4f} train_ppl={train_ppl:.2f} | "
            f"val_loss={val_loss:.4f} val_ppl={val_ppl:.2f} | "
            f"epoch_time={epoch_time/60:.2f} min"
        )
        print("------------------------------------------------------------")

        # --------------------------
        # CHECKPOINTS
        # --------------------------
        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": vars(cfg),
            "token_config": {
                k: v
                for k, v in TokenConfig.__dict__.items()
                if not k.startswith("__")
            },
            "train_day_ids": train_day_ids,
            "val_day_ids": val_day_ids,
            "packed_path": packed_path,
            "accum_steps": accum_steps,
            "batch_size": batch_size,
        }

        epoch_path = os.path.join(checkpoint_dir, f"agent_epoch_{epoch:03d}.pt")
        torch.save(ckpt, epoch_path)
        print(f"[CHECKPOINT] salvo: {epoch_path}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(checkpoint_dir, "agent_best.pt")
            torch.save(ckpt, best_path)
            print(f"[CHECKPOINT] melhor modelo atualizado: {best_path}")

    print("[TREINO] Finalizado.")

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    train_agent_transformer(
        packed_path=str(PATHS.tokens_out / "tokens_by_day.pt"),
        checkpoint_dir=str(PATHS.checkpoints / "checkpoints_agent_tiny"),
        num_epochs=10,
        batch_size=32,
        val_ratio=0.05,
        lr=3e-4,
        num_workers=0,
        seed=42,
        accum_steps=8,
    )
