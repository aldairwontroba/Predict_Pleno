# -*- coding: utf-8 -*-
"""
Função chamável para tokenizar TODOS os arquivos .npy usando o VQ-VAE.

Uso:
    from tokenize_all_events import tokenize_all_events

    out = tokenize_all_events(
        data_dir="C:/Users/Aldair/Desktop/eventos_processados",
        checkpoint="vq_out/vqvae_best_1311.pt",
        out="tokens_out/tokens_by_day.pt",
        batch_size=8192,
        device="cuda",
    )
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import vqvae_train_new as vqmod   # seu arquivo original


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Extrair chave do dia
# -----------------------------------------------------------------------------
def extract_day_key(path: Path) -> str:
    name = path.name

    # Busca padrão YYYY-MM-DD
    m = re.search(r"(\d{4}-\d{2}-\d{2})", name)
    if m:
        return m.group(1)

    # Busca padrão YYYYMMDD
    m = re.search(r"(\d{8})", name)
    if m:
        return m.group(1)

    # fallback
    return path.stem


# -----------------------------------------------------------------------------
# Carregar modelo
# -----------------------------------------------------------------------------
def load_vqvae(checkpoint_path: Path, device: torch.device):
    logger.info(f"[LOAD] Carregando VQVAE de {checkpoint_path}")
    model = vqmod.VQVAE().to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)

    model.eval()
    logger.info("[LOAD] VQ-VAE carregado.")
    return model


# -----------------------------------------------------------------------------
# Tokenizar um arquivo
# -----------------------------------------------------------------------------
def tokenize_file(model, npy_path: Path, device, batch_size: int) -> torch.Tensor:
    logger.info(f"[FILE] Lendo {npy_path.name} ...")
    arr = np.load(npy_path, mmap_mode="r")

    cols = np.array(vqmod.DL, dtype=np.int64)
    x_np = arr[:, cols].astype("float32")
    n = x_np.shape[0]

    logger.info(f"[FILE] {n} eventos → usando cols DL → shape final {x_np.shape}")

    tokens_list = []

    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = torch.from_numpy(x_np[i:i+batch_size]).to(device)
            z_e = model.enc_ln(model.encoder(batch)) * model.enc_scale

            if hasattr(model.vq, "levels"):
                _, codes_list, _ = model.vq(z_e)
                codes = codes_list[0]
            else:
                _, codes, _ = model.vq(z_e)

            tokens_list.append(codes.view(-1).cpu().long())

    tokens = torch.cat(tokens_list, dim=0)
    logger.info(f"[FILE] Tokens gerados: {tokens.numel()}")
    return tokens


# -----------------------------------------------------------------------------
# Função principal (chamável)
# -----------------------------------------------------------------------------
def tokenize_all_events(
    data_dir: str,
    checkpoint: str,
    out: Optional[str] = None,
    batch_size: int = 8192,
    device: str = "cuda"
):
    """
    Tokeniza todos os arquivos .npy do diretório e salva tudo agrupado por dia.

    Retorna um dicionário:
    {
        "day_keys": [...],
        "tokens_by_day": [...],
        "meta": {...}
    }
    """
    setup_logging()

    data_dir = Path(data_dir)
    checkpoint_path = Path(checkpoint)

    device_t = torch.device(device)

    logger.info("============================================================")
    logger.info(" TOKENIZAÇÃO INICIADA ")
    logger.info("============================================================")
    logger.info(f"DATA_DIR   = {data_dir}")
    logger.info(f"CHECKPOINT = {checkpoint_path}")
    logger.info(f"OUT        = {out}")
    logger.info(f"DEVICE     = {device_t}")
    logger.info(f"BATCH_SIZE = {batch_size}")

    if not data_dir.exists():
        raise FileNotFoundError(f"DATA_DIR não existe: {data_dir}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint não existe: {checkpoint_path}")

    # Carrega modelo
    model = load_vqvae(checkpoint_path, device_t)

    # Lista arquivos
    npy_files = sorted(data_dir.glob("*_norm_norm.npy"))

    if not npy_files:
        raise RuntimeError(f"Nenhum .npy encontrado em '{data_dir}'")

    logger.info(f"[SCAN] {len(npy_files)} arquivos encontrados.")

    # Para agrupar tokens por dia
    day_tokens: Dict[str, List[torch.Tensor]] = {}

    for f in npy_files:
        day_key = extract_day_key(f)
        logger.info(f"[DAY] {f.name} → dia '{day_key}'")

        tokens = tokenize_file(model, f, device_t, batch_size)

        if day_key not in day_tokens:
            day_tokens[day_key] = []
        day_tokens[day_key].append(tokens)

    # Consolidar por dia
    final_day_keys = []
    final_tokens_by_day = []

    for dk in sorted(day_tokens.keys()):
        toks = torch.cat(day_tokens[dk], dim=0)
        final_day_keys.append(dk)
        final_tokens_by_day.append(toks)
        logger.info(f"[FINAL] Dia {dk}: {toks.numel()} tokens")

    # Estatísticas globais
    all_tokens = torch.cat(final_tokens_by_day)
    vocab_est = int(all_tokens.max().item()) + 1

    logger.info("------------------------------------------------------------")
    logger.info(f"Total global de tokens: {all_tokens.numel()}")
    logger.info(f"Token mínimo: {int(all_tokens.min())}")
    logger.info(f"Token máximo: {int(all_tokens.max())}")
    logger.info(f"Vocabulário estimado: {vocab_est}")
    logger.info("------------------------------------------------------------")

    # Monta dicionário final
    out_dict = {
        "day_keys": final_day_keys,
        "tokens_by_day": final_tokens_by_day,
        "meta": {
            "data_dir": str(data_dir),
            "checkpoint": str(checkpoint),
            "batch_size": batch_size,
            "vocab_est": vocab_est,
            "num_days": len(final_day_keys),
            "DL": list(map(int, vqmod.DL)),
        }
    }

    # Salvar se pedido
    if out is not None:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(out_dict, out_path)
        logger.info(f"[SAVE] Tokens salvos em: {out_path}")

    logger.info("============================================================")
    logger.info(" TOKENIZAÇÃO FINALIZADA ")
    logger.info("============================================================")

    return out_dict

from tokenize_all_events import tokenize_all_events

result = tokenize_all_events(
    data_dir="C:/Users/Aldair/Desktop/eventos_processados",
    checkpoint="vq_out/vqvae_best_1311.pt",
    out="tokens_out/tokens_by_day.pt",
    batch_size=4096,
    device="cuda"
)

breakpoint()