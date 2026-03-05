"""
Fase 1: Pré-treino do ContinuousEventTransformer.

Tarefa: next-event prediction (autoregresso em espaço contínuo).
Loss  : Huber (smooth L1) – mais robusta que MSE para dados financeiros ruidosos.

Analogia com o pipeline existente:
  - Antes: VQ-VAE tokenizava → Transformer fazia next-token (CrossEntropy)
  - Agora: Transformer prediz diretamente o próximo vetor de features (Huber)

Uso:
  python -m src.continuous_transformer.train_pretrain
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.continuous_transformer.config import CTConfig, PretrainConfig
from src.continuous_transformer.model import ContinuousEventTransformer
from src.continuous_transformer.dataset_pretrain import PretrainDataset


# ============================================================
# Validação
# ============================================================

@torch.no_grad()
def evaluate_pretrain(
    model: ContinuousEventTransformer,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> float:
    model.eval()
    total_loss = 0.0
    total_elems = 0

    with torch.cuda.amp.autocast(enabled=use_amp):
        for x, y in loader:
            x = x.to(device, non_blocking=True)  # [B, T, input_dim]
            y = y.to(device, non_blocking=True)
            out = model(x)
            pred = out["next_event"]              # [B, T, input_dim]
            loss = F.huber_loss(pred, y, reduction="sum", delta=1.0)
            total_loss  += loss.item()
            total_elems += x.numel()              # B * T * input_dim

    return total_loss / max(1, total_elems)


# ============================================================
# Treino principal
# ============================================================

def train_pretrain(
    model_cfg:   CTConfig       | None = None,
    train_cfg:   PretrainConfig | None = None,
    resume_ckpt: str | Path | None    = None,
):
    if model_cfg is None:
        model_cfg = CTConfig()
    if train_cfg is None:
        train_cfg = PretrainConfig()

    torch.manual_seed(train_cfg.seed)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda")

    print("=" * 60)
    print(f"[PRETRAIN] device      : {device}")
    print(f"[PRETRAIN] input_dim   : {model_cfg.input_dim}")
    print(f"[PRETRAIN] d_model     : {model_cfg.d_model}")
    print(f"[PRETRAIN] n_layers    : {model_cfg.n_layers}")
    print(f"[PRETRAIN] n_heads     : {model_cfg.n_heads}")
    print(f"[PRETRAIN] seq_len     : {model_cfg.seq_len}")
    print(f"[PRETRAIN] data_dir    : {train_cfg.data_dir}")
    print("=" * 60)

    # Datasets
    ds_train = PretrainDataset(
        data_dir  = train_cfg.data_dir,
        seq_len   = train_cfg.seq_len,
        pattern   = train_cfg.pattern,
        val_ratio = train_cfg.val_ratio,
        split     = "train",
        seed      = train_cfg.seed,
        stride    = train_cfg.stride,
    )
    ds_val = PretrainDataset(
        data_dir  = train_cfg.data_dir,
        seq_len   = train_cfg.seq_len,
        pattern   = train_cfg.pattern,
        val_ratio = train_cfg.val_ratio,
        split     = "val",
        seed      = train_cfg.seed,
        stride    = train_cfg.stride,
    )

    dl_train = DataLoader(
        ds_train,
        batch_size  = train_cfg.batch_size,
        shuffle     = True,
        num_workers = train_cfg.num_workers,
        pin_memory  = use_amp,
        drop_last   = True,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size  = train_cfg.batch_size,
        shuffle     = False,
        num_workers = train_cfg.num_workers,
        pin_memory  = use_amp,
        drop_last   = False,
    )

    # Modelo
    model = ContinuousEventTransformer(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[PRETRAIN] parâmetros treináveis: {n_params:,}")

    opt    = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    out_dir = train_cfg.checkpoint_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    start_epoch   = 1
    accum = train_cfg.accum_steps

    # ── Resume ──────────────────────────────────────────────────
    # Aceita: resume_ckpt explícito OU auto-detecta o último checkpoint
    if resume_ckpt is None and train_cfg.resume:
        # Prioridade: latest (mid-epoch) > epoch mais alto
        latest_path = out_dir / "pretrain_latest.pt"
        if latest_path.exists():
            resume_ckpt = latest_path
        else:
            existing = sorted(out_dir.glob("pretrain_epoch_*.pt"))
            if existing:
                resume_ckpt = existing[-1]

    if resume_ckpt is not None and Path(resume_ckpt).exists():
        print(f"[PRETRAIN] Retomando de: {resume_ckpt}")
        saved = torch.load(resume_ckpt, map_location="cpu")
        model.load_state_dict(saved["model_state"])
        opt.load_state_dict(saved["opt_state"])
        # Mover estado do optimizer para o device correto
        for state in opt.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        start_epoch   = saved["epoch"] + 1
        best_val_loss = saved.get("val_loss", float("inf"))
        print(f"[PRETRAIN] Continuando da época {start_epoch}  "
              f"(best_val_loss anterior={best_val_loss:.5f})")
    elif resume_ckpt is not None:
        print(f"[PRETRAIN] AVISO: resume_ckpt não encontrado ({resume_ckpt}), "
              "iniciando do zero.")
    # ────────────────────────────────────────────────────────────

    for epoch in range(start_epoch, train_cfg.num_epochs + 1):
        t0  = datetime.now()
        now = t0.strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n===== EPOCH {epoch:03d}/{train_cfg.num_epochs:03d} | START {now} =====")

        model.train()
        run_loss    = 0.0
        run_elems   = 0
        log_loss    = 0.0
        log_count   = 0
        opt.zero_grad(set_to_none=True)

        for i, (x, y) in enumerate(dl_train, start=1):
            x = x.to(device, non_blocking=True)   # [B, T, input_dim]
            y = y.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                out  = model(x)
                pred = out["next_event"]           # [B, T, input_dim]
                loss = F.huber_loss(pred, y, reduction="mean", delta=1.0)

            scaler.scale(loss / accum).backward()

            run_loss  += loss.item() * x.numel()
            run_elems += x.numel()
            log_loss  += loss.item()
            log_count += 1

            if i % accum == 0:
                if train_cfg.max_grad_norm > 0:
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            if i % train_cfg.log_interval == 0:
                avg = log_loss / log_count
                now_s = datetime.now().strftime("%H:%M:%S")
                print(
                    f"[{now_s}] ep={epoch:03d} batch={i:06d}/{len(dl_train):06d} "
                    f"train_loss={avg:.5f}"
                )
                log_loss  = 0.0
                log_count = 0

            # Checkpoint mid-epoch (sobrescreve pretrain_latest.pt)
            if train_cfg.save_every > 0 and i % train_cfg.save_every == 0:
                latest_ckpt = {
                    "epoch":       epoch,
                    "model_state": model.state_dict(),
                    "opt_state":   opt.state_dict(),
                    "val_loss":    best_val_loss,
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
                torch.save(latest_ckpt, out_dir / "pretrain_latest.pt")
                now_s = datetime.now().strftime("%H:%M:%S")
                print(f"[{now_s}] [CKPT-MID] pretrain_latest.pt salvo (ep={epoch}, batch={i})")

        # Gradiente acumulado restante
        if (len(dl_train) % accum) != 0:
            if train_cfg.max_grad_norm > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

        train_loss = run_loss / max(1, run_elems)
        val_loss   = evaluate_pretrain(model, dl_val, device, use_amp)
        dt_min     = (datetime.now() - t0).total_seconds() / 60.0

        print("-" * 60)
        print(
            f"ep={epoch:03d}  train_loss={train_loss:.5f}  "
            f"val_loss={val_loss:.5f}  elapsed={dt_min:.1f}min"
        )
        print("-" * 60)

        # Checkpoint
        ckpt = {
            "epoch":       epoch,
            "model_state": model.state_dict(),
            "opt_state":   opt.state_dict(),
            "val_loss":    val_loss,
            "cfg":         {k: v for k, v in vars(model_cfg).items() if not k.startswith("_")},
        }
        # Salva direto os atributos de classe (CTConfig usa atributos de classe)
        ckpt["cfg"] = {
            "input_dim":   model_cfg.input_dim,
            "num_meta":    model_cfg.num_meta,
            "d_model":     model_cfg.d_model,
            "n_heads":     model_cfg.n_heads,
            "n_layers":    model_cfg.n_layers,
            "seq_len":     model_cfg.seq_len,
            "dropout":     model_cfg.dropout,
            "num_actions": model_cfg.num_actions,
        }

        epoch_path = out_dir / f"pretrain_epoch_{epoch:03d}.pt"
        torch.save(ckpt, epoch_path)
        # Remove latest mid-epoch (época concluída com sucesso)
        latest_path = out_dir / "pretrain_latest.pt"
        if latest_path.exists():
            latest_path.unlink()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = out_dir / "pretrain_best.pt"
            torch.save(ckpt, best_path)
            print(f"[CHECKPOINT] melhor modelo → {best_path}")

    print("[PRETRAIN] Finalizado.")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    model_cfg = CTConfig()
    # Ajuste hiperparâmetros aqui se quiser testar variações:
    # model_cfg.d_model  = 128
    # model_cfg.n_layers = 4

    train_cfg = PretrainConfig()
    # train_cfg.num_epochs = 5
    # train_cfg.batch_size = 32

    train_pretrain(model_cfg, train_cfg)
