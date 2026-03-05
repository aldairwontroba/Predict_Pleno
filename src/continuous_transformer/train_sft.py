"""
Fase 2: Fine-tuning supervisionado (SFT) do ContinuousEventTransformer.

Tarefa: dado os últimos n_past eventos normalizados, prever a ação correta
        (buy=0 / sell=1 / hold=2) no último token do passado.

Loss  : CrossEntropy (classificação)
Init  : carrega pesos do pré-treino (recomendado) ou treina do zero.

Uso:
  python -m src.continuous_transformer.train_sft
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.continuous_transformer.config import CTConfig, SFTConfig, ACTION_NAMES
from src.continuous_transformer.model import ContinuousEventTransformer
from src.continuous_transformer.dataset_sft import SFTDataset


# ============================================================
# Validação
# ============================================================

@torch.no_grad()
def evaluate_sft(
    model:      ContinuousEventTransformer,
    loader:     DataLoader,
    device:     torch.device,
    use_amp:    bool,
    model_cfg:  CTConfig,
) -> tuple[float, float]:
    """Retorna (val_loss, accuracy)."""
    model.eval()
    total_loss    = 0.0
    total_correct = 0
    total_samples = 0

    with torch.cuda.amp.autocast(enabled=use_amp):
        for past, meta, labels in loader:
            past   = past.to(device, non_blocking=True)    # [B, n_past, input_dim]
            meta   = meta.to(device, non_blocking=True)    # [B, num_meta]
            labels = labels.to(device, non_blocking=True)  # [B]

            if past.shape[1] > model_cfg.seq_len:
                past = past[:, -model_cfg.seq_len:, :]

            out    = model(past, meta)
            logits = out["action"]                          # [B, num_actions]

            loss = F.cross_entropy(logits, labels, reduction="sum")
            total_loss    += loss.item()
            total_correct += (logits.argmax(dim=-1) == labels).sum().item()
            total_samples += labels.shape[0]

    avg_loss = total_loss / max(1, total_samples)
    accuracy = total_correct / max(1, total_samples)
    return avg_loss, accuracy


# ============================================================
# Treino principal
# ============================================================

def train_sft(
    model_cfg:         CTConfig   | None = None,
    sft_cfg:           SFTConfig  | None = None,
    data_dir_override: str | None        = None,
):
    if model_cfg is None:
        model_cfg = CTConfig()
    if sft_cfg is None:
        sft_cfg = SFTConfig()

    torch.manual_seed(sft_cfg.seed)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda")

    # Datasets
    ds_train = SFTDataset(
        sequences_path    = sft_cfg.sequences_path,
        split             = "train",
        val_ratio         = sft_cfg.val_ratio,
        seed              = sft_cfg.seed,
        data_dir_override = data_dir_override,
    )
    ds_val = SFTDataset(
        sequences_path    = sft_cfg.sequences_path,
        split             = "val",
        val_ratio         = sft_cfg.val_ratio,
        seed              = sft_cfg.seed,
        data_dir_override = data_dir_override,
    )

    dl_train = DataLoader(
        ds_train,
        batch_size  = sft_cfg.batch_size,
        shuffle     = True,
        num_workers = sft_cfg.num_workers,
        pin_memory  = use_amp,
        drop_last   = True,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size  = sft_cfg.batch_size,
        shuffle     = False,
        num_workers = sft_cfg.num_workers,
        pin_memory  = use_amp,
        drop_last   = False,
    )

    # Modelo
    model = ContinuousEventTransformer(model_cfg).to(device)

    # Carrega pesos do pré-treino se disponível
    pretrain_ckpt = sft_cfg.pretrain_ckpt
    if pretrain_ckpt is not None and Path(pretrain_ckpt).exists():
        print(f"[SFT] Carregando pré-treino de: {pretrain_ckpt}")
        ckpt = torch.load(pretrain_ckpt, map_location="cpu")
        state = ckpt.get("model_state") or ckpt
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[SFT] Keys ausentes (ok se só action_head): {missing[:5]}")
        if unexpected:
            print(f"[SFT] Keys extras ignoradas: {unexpected[:5]}")
    else:
        print("[SFT] Sem pré-treino: treinando SFT do zero.")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("=" * 60)
    print(f"[SFT] device            : {device}")
    print(f"[SFT] parâmetros        : {n_params:,}")
    print(f"[SFT] sequences_path    : {sft_cfg.sequences_path}")
    print(f"[SFT] batch_size        : {sft_cfg.batch_size}")
    print(f"[SFT] accum_steps       : {sft_cfg.accum_steps}")
    print(f"[SFT] batch efetivo     : {sft_cfg.batch_size * sft_cfg.accum_steps}")
    print(f"[SFT] ações             : {ACTION_NAMES}")
    print("=" * 60)

    # Class weights: inverso da frequência, normalizados para média=1
    # buy≈26%, sell≈27%, hold≈47%  →  upweight buy/sell, downweight hold
    labels_all = ds_train.action_labels[ds_train.indices]
    n_total = len(labels_all)
    freq = torch.tensor([
        (labels_all == c).sum().item() / n_total
        for c in range(model_cfg.num_actions)
    ], dtype=torch.float32)
    class_weights = (1.0 / freq.clamp(min=1e-6))
    class_weights = class_weights / class_weights.mean()   # normaliza: média = 1
    class_weights = class_weights.to(device)
    print(f"[SFT] class_weights     : buy={class_weights[0]:.2f}  "
          f"sell={class_weights[1]:.2f}  hold={class_weights[2]:.2f}")

    opt    = torch.optim.AdamW(model.parameters(), lr=sft_cfg.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    out_dir = sft_cfg.checkpoint_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    accum = sft_cfg.accum_steps

    for epoch in range(1, sft_cfg.num_epochs + 1):
        t0  = datetime.now()
        now = t0.strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n===== EPOCH {epoch:03d}/{sft_cfg.num_epochs:03d} | START {now} =====")

        model.train()
        run_loss    = 0.0
        run_correct = 0
        run_total   = 0
        log_loss    = 0.0
        log_count   = 0
        opt.zero_grad(set_to_none=True)

        for i, (past, meta, labels) in enumerate(dl_train, start=1):
            past   = past.to(device, non_blocking=True)
            meta   = meta.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Trunca para seq_len caso n_past > seq_len (pega eventos mais recentes)
            if past.shape[1] > model_cfg.seq_len:
                past = past[:, -model_cfg.seq_len:, :]

            with torch.cuda.amp.autocast(enabled=use_amp):
                out    = model(past, meta)
                logits = out["action"]                 # [B, num_actions]
                loss   = F.cross_entropy(logits, labels, weight=class_weights, reduction="mean")

            scaler.scale(loss / accum).backward()

            B = labels.shape[0]
            run_loss    += loss.item() * B
            run_correct += (logits.detach().argmax(dim=-1) == labels).sum().item()
            run_total   += B
            log_loss    += loss.item()
            log_count   += 1

            if i % accum == 0:
                if sft_cfg.max_grad_norm > 0:
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), sft_cfg.max_grad_norm)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            if i % sft_cfg.log_interval == 0:
                avg  = log_loss / log_count
                acc  = run_correct / max(1, run_total)
                now_s = datetime.now().strftime("%H:%M:%S")
                print(
                    f"[{now_s}] ep={epoch:03d} batch={i:05d}/{len(dl_train):05d} "
                    f"loss={avg:.4f} acc={acc:.3f}"
                )
                log_loss  = 0.0
                log_count = 0

        # Gradiente restante
        if (len(dl_train) % accum) != 0:
            if sft_cfg.max_grad_norm > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), sft_cfg.max_grad_norm)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

        train_loss = run_loss  / max(1, run_total)
        train_acc  = run_correct / max(1, run_total)
        val_loss, val_acc = evaluate_sft(model, dl_val, device, use_amp, model_cfg)
        dt_min = (datetime.now() - t0).total_seconds() / 60.0

        print("-" * 60)
        print(
            f"ep={epoch:03d}  "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f}  |  "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}  "
            f"elapsed={dt_min:.1f}min"
        )
        print("-" * 60)

        ckpt = {
            "epoch":       epoch,
            "model_state": model.state_dict(),
            "val_loss":    val_loss,
            "val_acc":     val_acc,
            "cfg": {
                "input_dim":   model_cfg.input_dim,
                "num_meta":    model_cfg.num_meta,
                "d_model":     model_cfg.d_model,
                "n_heads":     model_cfg.n_heads,
                "n_layers":    model_cfg.n_layers,
                "seq_len":     model_cfg.seq_len,
                "dropout":     model_cfg.dropout,
                "num_actions": model_cfg.num_actions,
            },
        }

        epoch_path = out_dir / f"sft_epoch_{epoch:03d}.pt"
        torch.save(ckpt, epoch_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = out_dir / "sft_best.pt"
            torch.save(ckpt, best_path)
            print(f"[CHECKPOINT] melhor modelo → {best_path}  (val_acc={val_acc:.3f})")

    print("[SFT] Finalizado.")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    model_cfg = CTConfig()
    sft_cfg   = SFTConfig()

    train_sft(model_cfg, sft_cfg)
