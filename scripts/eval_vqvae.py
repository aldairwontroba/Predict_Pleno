#!/usr/bin/env python
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.config import PATHS

# ============================================================
# CONFIGS FIXAS DA AVALIAÇÃO
# ============================================================
DATA_DIR = str(PATHS.eventos_processados)
CHECKPOINT_PATH = str(PATHS.models / "vq_out" / "vqvae_ep019.pt")   # ajuste qual ep quer avaliar

# mesmo DL do treino:
DL = np.r_[0:4, 6:11, 12:23, 28, 31, 34, 37:59]
INPUT_DIM = len(DL)   # deve bater com cfg["input_dim"] no checkpoint

DEVICE_STR = "cuda"                 # "cuda" ou "cpu"
BATCH_SIZE = 2048
NUM_WORKERS = 4
MAX_FILES = 32                      # máx. de arquivos de teste
MAX_EVENTS = 200_000                # máx. de eventos
MAX_BATCHES = None                  # None = usa todos os batches

# Valores default de fallback (se por algum motivo o ckpt não tiver config)
NUM_EMBEDDINGS_DEFAULT = 2048
EMBEDDING_DIM_DEFAULT = 128
COMMIT_BETA_DEFAULT = 0.25

# ============================================================
# 1) LOGGING BÁSICO
# ============================================================
def setup_logger():
    logger = logging.getLogger("eval_recon")
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    ch.setFormatter(fmt)
    if not logger.handlers:
        logger.addHandler(ch)
    return logger

logger = setup_logger()

# ============================================================
# 2) DATASET SIMPLES PARA AVALIAÇÃO
# ============================================================

class EventsEvalDataset(Dataset):
    """
    Carrega um subconjunto de eventos de uma pasta.

    Agora **somente** arquivos que terminam com '_norm_norm.npy' são usados,
    que é o mesmo padrão do treino do VQ-VAE.

    Cada arquivo deve ser um .npy com array shape [N, D] já pré-processado
    e normalizado da MESMA FORMA que no treino.
    """
    def __init__(
        self,
        data_dir: str,
        max_files: int = 32,
        max_events: int = 200_000,
        pattern_suffix: str = "_norm_norm.npy",
        shuffle_files: bool = True,
    ):
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Pasta de dados não encontrada: {self.data_dir}")

        # <<< AQUI: só pega arquivos *_norm_norm.npy >>>
        all_files: List[Path] = sorted(self.data_dir.glob(f"*{pattern_suffix}"))

        if not all_files:
            raise RuntimeError(
                f"Nenhum arquivo encontrado em {self.data_dir} com padrão '*{pattern_suffix}'"
            )

        if shuffle_files:
            rng = np.random.default_rng(42)
            rng.shuffle(all_files)

        self.files = all_files[:max_files]
        logger.info(f"[DATA] Usando {len(self.files)} arquivo(s) para avaliação")

        # Carrega um subset de eventos em memória
        xs = []
        total = 0
        for f in self.files:
            arr = self._load_file(f)
            if arr.ndim != 2:
                raise RuntimeError(f"Array em {f} não é 2D: shape={arr.shape}")
            n = arr.shape[0]
            if total + n > max_events:
                n_keep = max_events - total
                if n_keep <= 0:
                    break
                xs.append(arr[:n_keep])
                total += n_keep
                break
            else:
                xs.append(arr)
                total += n
            if total >= max_events:
                break

        self.data = np.concatenate(xs, axis=0)
        logger.info(
            f"[DATA] Total de eventos carregados para avaliação: "
            f"{self.data.shape[0]} (dim={self.data.shape[1]})"
        )

    @staticmethod
    def _load_file(path: Path) -> np.ndarray:
        # aqui agora só esperamos .npy com *_norm_norm.npy
        if path.suffix != ".npy":
            raise RuntimeError(f"Extensão não suportada para avaliação: {path}")
        arr = np.load(path)
        arr = arr[:, DL]
        return arr.astype(np.float32)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]



# ============================================================
# 3) IMPORTA O MODELO
# ============================================================

try:
    from src.vqvae.train import VQVAE  # ajuste o nome do arquivo se for diferente
except Exception as e:
    logger.error("Falha ao importar VQVAE de src.vqvae.train: %s", e)
    VQVAE = None


# ============================================================
# 4) FUNÇÃO ADAPTADORA DA API DO MODELO
# ============================================================

def forward_with_indices(model: torch.nn.Module, x: torch.Tensor):
    """
    Adapta a saída do seu VQVAE para:
      - x_recon : [B, D]
      - indices : [B, L]  (L = nº de níveis que vamos contar; aqui 1)
    Pelo código que você mandou, o forward é:
        x_hat, vq_loss, recon, codes, z_e, z_q = model(x)
    e 'codes' é [B] (códigos do nível 0).
    """
    x_hat, vq_loss, recon, codes, z_e, z_q = model(x)

    # garante formato [B, L] para o contador
    if codes.ndim == 1:
        indices = codes.view(-1, 1)   # [B, 1]
    elif codes.ndim == 2:
        indices = codes               # [B, L]
    else:
        raise RuntimeError(f"Formato inesperado de codes: {codes.shape}")

    return x_hat, indices


# ============================================================
# 5) LOOP DE AVALIAÇÃO
# ============================================================

def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_embeddings: int,
    max_batches: int = None,
    num_feat_summary: int = 4,
):
    model.eval()

    n_total = 0

    sum_x = None
    sum_x2 = None
    sum_sq_err = None
    sum_abs_err = None

    code_counts = None         # [L, K]
    code_feat_sums = None      # [L, K, F]
    num_levels = None
    feat_dim = None

    sample_orig = None
    sample_recon = None

    with torch.no_grad():
        for b_idx, batch in enumerate(dataloader):
            if max_batches is not None and b_idx >= max_batches:
                break

            x = batch.to(device=device, dtype=torch.float32)   # [B, D]
            if x.ndim != 2:
                raise RuntimeError(f"Esperado x com shape [B, D], mas veio {x.shape}")

            x_recon, indices = forward_with_indices(model, x)
            if x_recon.shape != x.shape:
                raise RuntimeError(f"x_recon com shape {x_recon.shape}, esperado {x.shape}")

            B, D = x.shape

            if sum_x is None:
                feat_dim = D
                sum_x = torch.zeros(D, dtype=torch.float64)
                sum_x2 = torch.zeros(D, dtype=torch.float64)
                sum_sq_err = torch.zeros(D, dtype=torch.float64)
                sum_abs_err = torch.zeros(D, dtype=torch.float64)

            x_cpu = x.detach().cpu().to(torch.float64)
            x_recon_cpu = x_recon.detach().cpu().to(torch.float64)
            diff = x_recon_cpu - x_cpu

            sum_x += x_cpu.sum(dim=0)
            sum_x2 += (x_cpu ** 2).sum(dim=0)
            sum_sq_err += (diff ** 2).sum(dim=0)
            sum_abs_err += diff.abs().sum(dim=0)
            n_total += B

            # guarda algumas amostras para exibir depois
            if sample_orig is None:
                sample_orig = x_cpu[:5].clone()
                sample_recon = x_recon_cpu[:5].clone()

            # === Tokens / códigos ===
            idx_t = indices.detach().cpu()  # vamos trabalhar em CPU aqui

            if idx_t.ndim == 2:
                # [B, L]
                idx_flat = idx_t  # [N=B, L]
                x_flat = x_cpu    # [N=B, D]
            else:
                raise RuntimeError(f"Dimensão de indices não suportada: {idx_t.shape}")

            N, L = idx_flat.shape

            if num_levels is None:
                num_levels = L

            if code_counts is None:
                code_counts = torch.zeros(L, num_embeddings, dtype=torch.long)
                code_feat_sums = torch.zeros(L, num_embeddings, num_feat_summary, dtype=torch.float64)

            F = min(num_feat_summary, feat_dim)
            x_feat = x_flat[:, :F]  # [N, F]

            for lvl in range(L):
                codes_lvl = idx_flat[:, lvl].view(-1)  # [N]
                if codes_lvl.min() < 0 or codes_lvl.max() >= num_embeddings:
                    raise RuntimeError(
                        f"Códigos fora de faixa em nível {lvl}: "
                        f"[{int(codes_lvl.min())}, {int(codes_lvl.max())}] "
                        f"com num_embeddings={num_embeddings}"
                    )

                # contagens
                code_counts[lvl].index_add_(
                    0,
                    codes_lvl,
                    torch.ones_like(codes_lvl, dtype=torch.long)
                )

                # somas de features
                for f_idx in range(F):
                    vals = x_feat[:, f_idx]
                    code_feat_sums[lvl, :, f_idx].index_add_(0, codes_lvl, vals)

    if n_total == 0:
        logger.error("Nenhum dado avaliado (n_total == 0). Verifique dataloader/max_batches.")
        return

    # ========================================================
    # 5.1) Métricas de reconstrução
    # ========================================================
    mean_x = (sum_x / n_total).numpy()                 # [D]
    var_x = (sum_x2 / n_total) - (sum_x / n_total) ** 2
    var_x = torch.clamp(var_x, min=0.0)
    std_x = var_x.sqrt().numpy()

    mse = (sum_sq_err / n_total).numpy()
    rmse = np.sqrt(mse)
    mae = (sum_abs_err / n_total).numpy()
    nrmse = rmse / (std_x + 1e-8)

    logger.info("=== MÉTRICAS DE RECONSTRUÇÃO POR FEATURE ===")
    header = f"{'feat':>4} | {'mean':>10} {'std':>10} {'RMSE':>10} {'NRMSE':>10} {'MAE':>10}"
    logger.info(header)
    logger.info("-" * len(header))
    for d in range(feat_dim):
        logger.info(
            f"{d:4d} | {mean_x[d]:10.4f} {std_x[d]:10.4f} "
            f"{rmse[d]:10.4f} {nrmse[d]:10.4f} {mae[d]:10.4f}"
        )

    # ========================================================
    # 5.2) Estatística dos tokens
    # ========================================================
    logger.info("")
    logger.info("=== ESTATÍSTICAS DOS TOKENS POR NÍVEL ===")

    for lvl in range(num_levels):
        counts = code_counts[lvl].to(torch.float64)  # [K]
        total = counts.sum()
        if total == 0:
            logger.warning(f"Nível {lvl}: nenhuma contagem (?)")
            continue

        used = int((counts > 0).sum().item())
        p = counts / total
        p_nonzero = p[p > 0]
        entropy = -(p_nonzero * p_nonzero.log()).sum().item()
        perplexity = float(np.exp(entropy))

        mean_usage = counts.mean().item()
        median_usage = counts.median().item()
        p5 = float(np.percentile(counts.numpy(), 5))
        p95 = float(np.percentile(counts.numpy(), 95))

        logger.info(
            f"[LVL {lvl}] used={used}/{num_embeddings} "
            f"Perplx={perplexity:.1f} "
            f"count{{mean={mean_usage:.1f},med={median_usage:.1f},p5={p5:.1f},p95={p95:.1f}}}"
        )

    # ========================================================
    # 5.3) Médias condicionais de features por código (TOP K)
    # ========================================================
    logger.info("")
    logger.info("=== MÉDIA DE FEATURES POR CÓDIGO (TOP 10 MAIS FREQUENTES) ===")

    top_k = 10
    F = min(num_feat_summary, feat_dim)

    for lvl in range(num_levels):
        logger.info(f"--- Nível {lvl} ---")
        counts = code_counts[lvl].to(torch.float64)  # [K]
        total = counts.sum()
        if total == 0:
            logger.info("  Nenhum código usado.")
            continue

        top_codes = torch.argsort(counts, descending=True)[:top_k]
        for c in top_codes:
            c = int(c.item())
            c_count = int(counts[c].item())
            if c_count == 0:
                continue
            means_feat = (code_feat_sums[lvl, c, :F] / counts[c]).numpy()
            feat_str = " ".join([f"{means_feat[f]:+.3f}" for f in range(F)])
            logger.info(
                f"  code={c:4d} count={c_count:7d} | "
                f"mean_features[0:{F}] = {feat_str}"
            )

    # ========================================================
    # 5.4) Exemplo de reconstrução
    # ========================================================
    if sample_orig is not None and sample_recon is not None:
        logger.info("")
        important_idx = [0, 1, 4, 6, 8, 10, 12, 29, 30, 31]
        # garante que não vamos acessar índice fora do range
        idx = [j for j in important_idx if j < feat_dim]

        logger.info("=== EXEMPLO DE RECONSTRUÇÃO (5 amostras, features selecionadas) ===")
        logger.info(f"features = {idx}")

        num_samples = min(5, sample_orig.shape[0])

        for i in range(num_samples):
            orig_vals = sample_orig[i, idx].numpy()
            recon_vals = sample_recon[i, idx].numpy()
            orig_str = " ".join(f"{v:+.3f}" for v in orig_vals)
            recon_str = " ".join(f"{v:+.3f}" for v in recon_vals)
            logger.info(f"amostra {i:02d} | x[feat]      = {orig_str}")
            logger.info(f"           | x_hat[feat] = {recon_str}")


# ============================================================
# 6) MAIN
# ============================================================
def main():
    if VQVAE is None:
        raise RuntimeError("VQVAE não pôde ser importado. Ajuste o import no topo do arquivo.")

    # dispositivo
    if torch.cuda.is_available() and DEVICE_STR == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"[INFO] Dispositivo de avaliação: {device}")

    # dataset / dataloader
    ds = EventsEvalDataset(
        data_dir=DATA_DIR,
        max_files=MAX_FILES,
        max_events=MAX_EVENTS,
    )
    dl = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # checkpoint
    ckpt_path = Path(CHECKPOINT_PATH)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint não encontrado: {ckpt_path}")

    logger.info(f"[INFO] Carregando checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    cfg = ckpt.get("config", {})

    input_dim   = int(cfg.get("input_dim",  INPUT_DIM))
    hidden      = cfg.get("hidden_dims", None)
    emb_dim     = int(cfg.get("embedding_dim",  EMBEDDING_DIM_DEFAULT))
    num_emb_all = int(cfg.get("num_embeddings", NUM_EMBEDDINGS_DEFAULT))
    commit_beta = float(cfg.get("commit_beta",  COMMIT_BETA_DEFAULT))

    if hidden is None:
        raise RuntimeError("hidden_dims não encontrado em ckpt['config']. Ajuste se necessário.")

    logger.info(
        "[CFG] input_dim=%d hidden=%s emb_dim=%d num_emb_total=%d commit_beta=%.4f",
        input_dim, str(hidden), emb_dim, num_emb_all, commit_beta
    )

    # instancia modelo
    model = VQVAE(
        input_dim=input_dim,
        hidden=hidden,
        emb_dim=emb_dim,
        num_emb=num_emb_all,
        commit_beta=commit_beta,
    )

    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    epoch = ckpt.get("epoch", None)
    perp = ckpt.get("perplexity", None)
    if epoch is not None and perp is not None:
        logger.info("[INFO] Modelo carregado (epoch=%d, perplexity=%.2f)", epoch, perp)
    elif epoch is not None:
        logger.info("[INFO] Modelo carregado (epoch=%d)", epoch)
    else:
        logger.info("[INFO] Modelo carregado do checkpoint.")

    # K por nível (no seu RVQ: num_emb_all / n_levels)
    if hasattr(model, "vq") and hasattr(model.vq, "levels"):
        k_per_level = int(model.vq.levels[0].num_embeddings)
        n_levels = len(model.vq.levels)
    else:
        k_per_level = num_emb_all
        n_levels = 1

    logger.info(f"[INFO] RVQ: níveis={n_levels}, K_por_nível={k_per_level}")

    # avaliação
    logger.info("[INFO] Iniciando avaliação de reconstrução e tokens...")
    evaluate(
        model=model,
        dataloader=dl,
        device=device,
        num_embeddings=k_per_level,
        max_batches=MAX_BATCHES,
        num_feat_summary=4,
    )
    logger.info("[INFO] Avaliação concluída.")


if __name__ == "__main__":
    main()
