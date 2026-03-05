# vqvae_train_fast.py
import time, random, os
import logging
from pathlib import Path
from typing import List, Tuple, Iterator

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import IterableDataset, DataLoader

import matplotlib.pyplot as plt

# ======= LOGGING CONFIGURATION =======
# Configure logging once. The default level is INFO.  Set the environment
# variable `PYTHONLOGLEVEL=DEBUG` or call
# `logging.getLogger(__name__).setLevel(logging.DEBUG)` to see more details.
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Obter logger padrão para este módulo.
logger = logging.getLogger(__name__)

# ======= CONFIG =======
from src.config import PATHS

# Caminho onde estão armazenados os arquivos .npy de eventos.  Ajuste conforme sua estrutura.
DATA_DIR = PATHS.eventos_processados

# Padrões de arquivo a tentar em ordem.  O primeiro padrão encontrado é usado.
PATTERNS = ["*_norm_norm.npy", "*_norm.npy"]

# Cada evento é um vetor de 45 características já normalizadas (mas alguns
# valores podem ultrapassar o intervalo [-1, 1], chegando a ~2–3).  Para
# simplificar o carregamento utilizamos todas as 45 colunas do vetor.
DL = np.r_[0:4,6:11,12:23,28,31,34,37:59]  # utiliza todas as colunas disponíveis
INPUT_DIM = len(DL)

# Ajuste de lote para 6 milhões de eventos: um batch moderado ajuda a não
# esgotar memória GPU/CPU.  O valor efetivo de batch por passo será
# BATCH_SIZE * ACC_STEPS.  Com ACC_STEPS=16 temos um lote efetivo de ~131k
# eventos.  Esses valores funcionam bem com 6M eventos e 45 features.
BATCH_SIZE = 8192 * 4
ACCUMULATE_ALL = False
ACC_STEPS = 16 // 2

# Taxa de aprendizado base.  A taxa para o encoder é reduzida (ver otimizador).
LR = 3e-4

# Total de embeddings do código.  Com 4 níveis de RVQ (abaixo), 2048 total
# significa 512 embeddings por nível.  Para conjuntos muito grandes você
# pode aumentar esse valor (ex: 4096) para melhorar a capacidade.
NUM_EMBEDDINGS = 4096 * 4
EMBED_DIM = 256
HIDDEN_DIMS = (2048, 1024, 512, 256)

NUM_WORKERS = 4
GRAD_CLIP = 0.3
EMA_DECAY = 0.97
EMA_EPS = 1e-5
COMMIT_BETA = 0.5

# Número de épocas.  Com 6M eventos, 10–15 épocas costumam ser suficientes.
NUM_EPOCHS = 30

WEIGHT_DECAY = 1e-6
SEED = 1337
PERSISTENT_WORKERS = True
PRINT_PER_EPOCH = True

# tau schedule (anneal)
TAU_START = 1.8
TAU_END = 1.0

# EMA warmup epochs
EMA_WARMUP_EPOCHS = 5
EMA_WARMUP_DECAY = 0.95

# resurrect params
RESURRECT_MIN_COUNT = 1.0
Z_BUF_MAX = 16384   # máximo de vetores z_e a guardar para re-seed

# output dir
OUT_DIR = PATHS.models / "vq_out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ======= RUNTIME TWEAKS =======
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

from sklearn.cluster import MiniBatchKMeans
def init_codebook_with_kmeans(model, dl, device,
                              n_samples: int = 100000,
                              batch_collect: int = 4096,
                              random_state: int = 1337):
    """
    Coleta z_e do encoder (modo eval) e roda MiniBatchKMeans para inicializar
    model.vq.embedding.weight. Também inicializa ema_w e ema_cluster_size.
    - model : a instância do VQVAE (deve ter .encoder, .enc_ln, .enc_scale, .vq)
    - dl    : DataLoader (iterável que produz batches torch.Tensor)
    - device: device onde o model está (cuda/cpu)
    - n_samples: número total de vetores z_e a coletar (ex: 50k..200k)
    - batch_collect: batch size interno do kmeans (MiniBatchKMeans)
    """
    model.eval()
    collected = []
    got = 0
    with torch.no_grad():
        for i, xb in enumerate(dl):
            xb = xb.to(device, non_blocking=True)
            # forward encoder (compatível com sua implementação)
            z = model.encoder(xb)
            z = model.enc_ln(z)
            z = torch.tanh(z) * model.enc_scale
            arr = z.detach().cpu().numpy()
            collected.append(arr)
            got += arr.shape[0]
            if got >= n_samples:
                break

    if len(collected) == 0:
        raise RuntimeError("init_codebook_with_kmeans: nenhum z coletado (verifique dl).")

    Z = np.concatenate(collected, axis=0)
    Z = Z[:n_samples].astype(np.float32)
    logger.info(f"[kmeans] coletado {Z.shape[0]} vetores z para kmeans (emb_dim={Z.shape[1]})")

    # MiniBatchKMeans
    kmeans = MiniBatchKMeans(n_clusters=model.vq.num_embeddings,
                             batch_size=max(1024, batch_collect),
                             n_init=3,
                             random_state=random_state)
    kmeans.fit(Z)
    centroids = kmeans.cluster_centers_.astype(np.float32)   # shape [K, emb_dim]

    # copy centroids to embedding and init EMA buffers
    dev = model.vq.embedding.weight.device
    with torch.no_grad():
        model.vq.embedding.weight.data.copy_(torch.from_numpy(centroids).to(dev))

        # initialize ema_w and ema_cluster_size so EMA-based reassignments are stable
        # set ema_cluster_size proportional to average count (avoid zeros)
        avg_count = float(Z.shape[0]) / float(model.vq.num_embeddings)
        csize = torch.full_like(model.vq.ema_cluster_size, fill_value=avg_count, device=dev)
        model.vq.ema_cluster_size.copy_(csize)

        # ema_w should be cluster_centers * cluster_size (so embed_norm = ema_w / cluster_size = centroids)
        ew = torch.from_numpy(centroids).to(dev) * csize.unsqueeze(1)
        model.vq.ema_w.copy_(ew)

    logger.info("[kmeans] codebook e buffers EMA inicializados.")
    model.train()

def set_seed(sd: int = 1337):
    random.seed(sd)
    np.random.seed(sd)
    torch.manual_seed(sd)
    torch.cuda.manual_seed_all(sd)

def list_first_nonempty(data_dir: Path, patterns: List[str]) -> List[Path]:
    for pat in patterns:
        fs = sorted(data_dir.glob(pat))
        if fs: return fs
    return []

def memmap_shape(path: Path) -> Tuple[int,int]:
    x = np.load(path, mmap_mode='r')
    if x.ndim != 2 or x.shape[1] != INPUT_DIM:
        raise ValueError(f"{path.name}: shape esperado (N,{INPUT_DIM}), veio {x.shape}")
    return x.shape

# ======= DATASET por CHUNKS =======
DROP_LAST = False

class NpyChunkDataset(IterableDataset):
    def __init__(self, files, dl, batch_size: int, shuffle: bool = True):
        super().__init__()
        self.files = files
        self.bs = int(batch_size)
        self.shuffle = shuffle
        self.cols = DL

        self.shapes = []
        for f in files:
            arr = np.load(f, mmap_mode='r')
            ar = arr[:, DL]
            if ar.ndim != 2:
                raise ValueError(f"{f.name}: esperado 2D, veio {ar.shape}")
            self.shapes.append(ar.shape[0])

        if sum(self.shapes) == 0:
            raise RuntimeError("Sem dados.")

        self.tasks_master = []
        for fid, n in enumerate(self.shapes):
            for start in range(0, n, self.bs):
                end = min(start + self.bs, n)
                self.tasks_master.append((fid, start, end))

    def __len__(self):
        return sum(self.shapes)

    def __iter__(self):
        tasks = self.tasks_master.copy()
        if self.shuffle:
            random.shuffle(tasks)

        mems = [np.load(f, mmap_mode='r') for f in self.files]

        buf = None

        for fid, start, end in tasks:
            chunk = mems[fid][start:end]
            if self.cols is not None:
                chunk = chunk[:, self.cols]
            chunk = np.ascontiguousarray(chunk)
            if buf is None:
                buf = chunk
            else:
                buf = np.concatenate([buf, chunk], axis=0)

            while buf.shape[0] >= self.bs:
                out = buf[:self.bs]
                buf = buf[self.bs:]
                yield torch.from_numpy(np.ascontiguousarray(out)).float()

        if buf is not None and buf.shape[0] > 0 and not DROP_LAST:
            yield torch.from_numpy(np.ascontiguousarray(buf)).float()

# ======= VQ EMA (Cosine) =======
class VectorQuantizerEMA(nn.Module):
    """
    Cosine-VQ com EMA:
    - matching por similaridade do cosseno (z e codebook normalizados)
    - EMA acumula vetores normalizados (unit-norm)
    - codebook mantido normalizado
    - z_q preserva a norma original de z_e: z_q = ||z_e|| * e_idx_unit
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, decay: float = 0.99, eps: float = 1e-5):
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.decay = float(decay)
        self.eps = float(eps)

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=1.0)
        with torch.no_grad():
            # mantém unit-norm já no início
            self.embedding.weight.data = F.normalize(self.embedding.weight.data, dim=1)

        self.register_buffer("ema_cluster_size", torch.zeros(self.num_embeddings))
        self.register_buffer("ema_w", torch.zeros(self.num_embeddings, self.embedding_dim))

    def forward(self, z_e: torch.Tensor, tau: float = 1.0):
        # z_e: (B, D)
        B, D = z_e.shape
        zf = z_e.detach().float()

        # normaliza amostras e codebook p/ matching por cosseno
        z_n = F.normalize(zf, dim=1, eps=self.eps)                          # (B, D)
        e_n = F.normalize(self.embedding.weight.float(), dim=1, eps=self.eps)  # (K, D)

        # similaridade cos: sim = z_n · e_n^T
        sim = z_n @ e_n.t()                                                 # (B, K)
        if tau != 1.0:
            sim = sim / float(tau)  # escala de temperatura (aplana ou agudiza as diferenças)

        # seleciona o código por argmax da similaridade
        codes = torch.argmax(sim, dim=1)                                    # (B,)

        # === EMA updates ===
        with torch.no_grad():
            # one-hot em device do input (rápido); buffers serão atualizados no device dos buffers
            oh = torch.zeros(B, self.num_embeddings, device=z_e.device, dtype=torch.float)
            oh.scatter_(1, codes.view(-1, 1), 1.0)

            # contagem com decaimento
            self.ema_cluster_size.mul_(self.decay).add_(
                oh.sum(0).to(self.ema_cluster_size.device) * (1.0 - self.decay)
            )

            # acumula vetores normalizados (direções)
            dw = oh.t().to(z_n.device) @ z_n                                # (K, D), em device do z_n
            self.ema_w.mul_(self.decay).add_(dw.to(self.ema_w.device) * (1.0 - self.decay))

            # atualiza codebook como média dos vetores unit-norm e re-normaliza cada embedding
            n = float(self.ema_cluster_size.sum().item())
            cluster_size = (self.ema_cluster_size + self.eps) / (n + float(self.num_embeddings) * self.eps) * n
            embed_mean = self.ema_w / cluster_size.unsqueeze(1).clamp_min(self.eps)  # (K, D)
            embed_unit = F.normalize(embed_mean, dim=1, eps=self.eps)
            self.embedding.weight.data.copy_(embed_unit.to(self.embedding.weight.dtype, non_blocking=True))

        # === quantização: preserva a norma de z_e ===
        with torch.no_grad():
            e_sel = F.embedding(codes, self.embedding.weight)               # (B, D), unit-norm
            z_norm = zf.norm(p=2, dim=1, keepdim=True).clamp_min(self.eps)  # (B, 1)
            z_q = e_sel * z_norm                                            # (B, D)

        # straight-through
        z_q_st = z_e + (z_q - z_e).detach()

        # "distância" como 1 - cos (opcional, útil p/ logs)
        dists = 1.0 - (z_n * F.normalize(e_sel.float(), dim=1, eps=self.eps)).sum(dim=1)

        return z_q_st, codes, dists  # dists ~ [0, 2]; menor é melhor (cos mais alto)

    @torch.no_grad()
    def resurrect_dead(self, z_sample: torch.Tensor, min_count: float = 1.0):
        """
        Reinicializa embeddings cuja contagem EMA < min_count.
        Usa direções (unit-norm) de z_sample.
        """
        if z_sample is None or z_sample.numel() == 0:
            return 0
        dead = (self.ema_cluster_size < float(min_count)).to(self.ema_cluster_size.device)
        n_dead = int(dead.sum().item())
        if n_dead == 0:
            return 0

        # escolhe amostras aleatórias e normaliza direção
        z_sample = z_sample.detach()
        ns = z_sample.shape[0]
        idxs = torch.randint(0, ns, (n_dead,), device=z_sample.device)
        new_vecs = z_sample[idxs]
        new_vecs = F.normalize(new_vecs, dim=1, eps=self.eps)

        # aplica nos embeddings e zera estatísticas desses códigos
        dev_emb = self.embedding.weight.device
        self.embedding.weight.data[dead.to(dev_emb)] = new_vecs.to(self.embedding.weight.dtype).to(dev_emb)
        self.ema_w[dead] = 0.0
        self.ema_cluster_size[dead] = 0.0
        return n_dead

# ======= RVQ (residual) usando seus quantizadores EMA =======
class ResidualVectorQuantizerEMA(nn.Module):
    def __init__(self, n_levels: int, num_embeddings: int, embedding_dim: int,
                 decay: float = 0.995, eps: float = 1e-5):
        super().__init__()
        self.levels = nn.ModuleList([
            VectorQuantizerEMA(num_embeddings, embedding_dim, decay=decay, eps=eps)
            for _ in range(n_levels)
        ])
        # ---- Compat layer p/ manter init_codebook_with_kmeans funcionando ----
        self.num_levels = n_levels
        self.num_embeddings = num_embeddings          # por nível (compat)
        self.embedding_dim = embedding_dim
        # a seguir, proxies para o nível 0 (o init antigo mexe nestes nomes)
        self.embedding = self.levels[0].embedding
        self.ema_w = self.levels[0].ema_w
        self.ema_cluster_size = self.levels[0].ema_cluster_size
        # ----------------------------------------------------------------------

    def forward(self, z_e: torch.Tensor, tau: float = 1.0):
        residual = z_e
        zq_list, codes_list, dists_list = [], [], []
        for vq in self.levels:
            z_q, codes, dists = vq(residual, tau=tau)
            zq_list.append(z_q)
            codes_list.append(codes)
            dists_list.append(dists)
            residual = residual - z_q.detach()
        z_q_sum = torch.stack(zq_list, dim=0).sum(dim=0)
        return z_q_sum, codes_list, dists_list

    @torch.no_grad()
    def resurrect_dead(self, z_sample: torch.Tensor, min_count: float = 1.0):
        tot = 0
        for vq in self.levels:
            tot += vq.resurrect_dead(z_sample, min_count)
        return tot

    # helpers úteis (opcionais)
    def get_level(self, i: int) -> VectorQuantizerEMA:
        return self.levels[i]


# ======= VQVAE com RVQ (2 níveis de 1024) =======
class VQVAE(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden=HIDDEN_DIMS, emb_dim=EMBED_DIM,
                 num_emb=NUM_EMBEDDINGS, commit_beta=COMMIT_BETA):
        super().__init__()
        self.encoder = self._mlp(input_dim, list(hidden) + [emb_dim])
        self.enc_ln = nn.LayerNorm(emb_dim)
        self.enc_scale = nn.Parameter(torch.tensor(1.0))

        # Configura o número de níveis para RVQ.  Com 4 níveis e
        # NUM_EMBEDDINGS=2048, cada nível terá 512 embeddings.  A soma das
        # quantizações residuais melhora a capacidade de representação.
        n_levels = 4
        self.vq = ResidualVectorQuantizerEMA(
            n_levels=n_levels,
            num_embeddings=max(1, num_emb // n_levels),
            embedding_dim=emb_dim,
            decay=EMA_DECAY,
            eps=EMA_EPS,
        )

        self.decoder = self._mlp(emb_dim, list(reversed(hidden)) + [input_dim], act_last=False)
        self.commit_beta = float(commit_beta)   # 1.0–1.5 funciona bem com RVQ + commit coseno
        self.current_tau = 1.0

    @staticmethod
    def _mlp(inp, layers, act_last=False):
        mods=[]; d=inp
        for i,o in enumerate(layers):
            mods.append(nn.Linear(d,o))
            if i < len(layers)-1 or act_last:
                mods.append(nn.ReLU(inplace=True))
            d=o
        return nn.Sequential(*mods)

    def forward(self, x: torch.Tensor):
        # Encoder
        z_e = self.encoder(x)
        z_e = self.enc_ln(z_e)
        z_e = torch.tanh(z_e) * self.enc_scale

        # Detecta se é RVQ (tem .levels) ou VQ simples
        is_rvq = hasattr(self.vq, "levels")

        if not is_rvq:
            # ===== VQ simples =====
            z_q, codes, _ = self.vq(z_e, tau=getattr(self, "current_tau", 1.0))
            # perdas
            recon = F.mse_loss(self.decoder(z_q), x, reduction="mean")
            commit = F.mse_loss(z_e, z_q.detach(), reduction="mean")
            vq_loss = self.commit_beta * commit
            x_hat = self.decoder(z_q)

            # ativo (compat)
            with torch.no_grad():
                _ = codes.unique().numel()

            return x_hat, vq_loss, recon, codes, z_e, z_q

        else:
            # ===== RVQ (multi-níveis) =====
            residual = z_e
            z_q_sum = 0.0
            commit_total = 0.0
            codes_all = []
            commit_per_level = []

            tau = getattr(self, "current_tau", 1.0)

            for lvl, vq_lvl in enumerate(self.vq.levels):
                z_q_l, codes_l, _ = vq_lvl(residual, tau=tau)
                # acumula saída quantizada
                z_q_sum = z_q_sum + z_q_l
                # commit em cima do resíduo atual (stop-grad em z_q_l)
                commit_l = F.mse_loss(residual, z_q_l.detach(), reduction="mean")
                commit_total = commit_total + commit_l
                commit_per_level.append(commit_l.detach())
                codes_all.append(codes_l)
                # atualiza resíduo
                residual = residual - z_q_l.detach()

            # reconstrução com soma dos níveis
            x_hat = self.decoder(z_q_sum)
            # recon = F.mse_loss(x_hat, x, reduction="mean")
            recon = F.smooth_l1_loss(x_hat, x, beta=0.5, reduction="mean")
            vq_weight = self.commit_beta / len(self.vq.levels)  # média por nível
            vq_loss  = vq_weight * commit_total


            # === Compatibilidade com treino atual ===
            # Retorna apenas codes do nível 0 (mesmo shape antigo: [B])
            codes = codes_all[0]
            z_q = z_q_sum

            # (Opcional) guardar diagnósticos por nível para uso externo
            self._last_codes_all = codes_all            # lista de [B] por nível
            self._last_commit_per_level = commit_per_level  # lista de escalares

            return x_hat, vq_loss, recon, codes, z_e, z_q



# ======= TREINO =======
def main():
    """
    Treinamento principal do VQ-VAE com RVQ.  Este método configura o
    ambiente, carrega os dados, inicializa o modelo e executa o loop de
    treinamento.  Logs informativos são emitidos através do módulo
    `logging` para acompanhar o progresso.
    """
    # inicializa semente de aleatoriedade
    set_seed(SEED)
    logger = logging.getLogger(__name__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Dispositivo de treinamento: {device}")

    files = list_first_nonempty(DATA_DIR, PATTERNS)
    if not files:
        raise FileNotFoundError(f"Nenhum arquivo encontrado em {DATA_DIR}")
    logger.info(f"{len(files)} arquivo(s) encontrado(s) em {DATA_DIR}")

    ds = NpyChunkDataset(files, DL, batch_size=BATCH_SIZE, shuffle=True)
    dl = DataLoader(
        ds,
        batch_size=None,
        num_workers=NUM_WORKERS,          # já está >0
        pin_memory=True,
        persistent_workers=PERSISTENT_WORKERS,
        prefetch_factor=8,                # <-- adicione isto
    )

    # log tamanho aproximado do dataset
    try:
        dataset_size = len(ds)
        logger.info(f"Dataset contém aproximadamente {dataset_size:,} eventos")
    except Exception:
        logger.info("Dataset é iterável, tamanho total não determinado")

    model = VQVAE().to(device)
    # model = torch.compile(model, backend="aot_eager")  # usa AOT Autograd; dispensa Triton
    logger.info(f"Modelo inicializado com {len(model.vq.levels)} níveis de RVQ")

    # inicializa o codebook com KMeans (executar somente uma vez)
    logger.info("Iniciando inicialização do codebook com KMeans.  Isso pode levar alguns minutos...")
    init_codebook_with_kmeans(model, dl, device, n_samples=100000, batch_collect=4096)
    logger.info("Inicialização do codebook concluída.  Iniciando treinamento.")
    # agora continue com opt, scaler e loop de treino

    opt = torch.optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': LR*0.05},   # reduzir lr do encoder
        {'params': model.decoder.parameters(), 'lr': LR},
        {'params': [model.enc_scale], 'lr': LR*0.05}
    ], lr=LR, weight_decay=WEIGHT_DECAY)

    # 3) clamp enc_scale every N steps (colocar no loop de treino, após forward)
    with torch.no_grad():
        if hasattr(model, "enc_scale"):
            model.enc_scale.data.clamp_(0.01, 5.0)   # ajusta max conforme quiser

    # GradScaler/Autocast: habilitar somente se houver CUDA (mais seguro)
    # use torch.amp (nova API) — device_type 'cuda' quando GPU
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    amp_ctx = torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda"))
    # accumulate steps
    if ACCUMULATE_ALL:
        total_events = len(ds)
        microbatches = int(np.ceil(total_events / BATCH_SIZE))
        acc_steps = max(1, microbatches)
    else:
        acc_steps = max(1, int(ACC_STEPS))

    logger.info(f"acc_steps configurado para {acc_steps} — batch efetivo ≈ {acc_steps*BATCH_SIZE:,} eventos")

    # buffer for z samples to resurrect dead codes
    z_buf_list = []  # list of tensors (cpu)
    z_buf_count = 0

    # ensure diagnostics dir exists
    (OUT_DIR / "vq_diagnostics").mkdir(parents=True, exist_ok=True)

    num_e = model.vq.num_embeddings

    def _p5_p95(arr: np.ndarray):
        if arr.size == 0:
            return 0.0, 0.0
        return float(np.percentile(arr, 5)), float(np.percentile(arr, 95))

    for epoch in range(NUM_EPOCHS):
        # tau anneal and EMA warmup
        model.current_tau = TAU_START + (TAU_END - TAU_START) * (epoch / max(1, NUM_EPOCHS - 1))
        model.vq.decay = EMA_WARMUP_DECAY if epoch < EMA_WARMUP_EPOCHS else float(EMA_DECAY)

        model.train()
        total_loss = total_vq = total_mse = 0.0
        step_in_epoch = 0
        t0 = time.time()

        opt.zero_grad(set_to_none=True)

        # epoch diagnostics (CPU-friendly)
        epoch_used_hist = torch.zeros(num_e, dtype=torch.long)
        sums_dist = torch.zeros(num_e, dtype=torch.float)   # soma das distâncias por código
        total_samples_epoch = 0

        # agregados leves por época (médias)
        agg_batches = 0
        agg_mean_znorm = 0.0
        agg_mean_dist = 0.0

        for i, batch in enumerate(dl):
            x = torch.nan_to_num(batch, nan=0.0, posinf=0.0, neginf=0.0).to(device, non_blocking=True)

            with amp_ctx:
                # expected: x_hat, vq_loss, recon_loss, codes, z_e, z_q
                x_hat, vq_loss, recon_loss, codes, z_e, z_q = model(x)
                # divide por acc_steps para grad accumulation
                loss = (recon_loss + vq_loss) / acc_steps

            # ====== DIAGNÓSTICOS BARATOS (CPU) ======
            with torch.no_grad():
                # médias do batch
                mean_norm = z_e.norm(dim=1).mean().item()
                mean_dist = (z_e.sub(z_q).pow(2).sum(dim=1).sqrt()).mean().item()

                agg_batches += 1
                agg_mean_znorm += mean_norm
                agg_mean_dist += mean_dist

                # histograma em CPU
                codes_cpu = codes.detach().cpu()
                epoch_used_hist += torch.bincount(codes_cpu, minlength=num_e)

                # distâncias por amostra -> somatório por código (para média por código no fim)
                dists_cpu = (z_e.sub(z_q).pow(2).sum(dim=1).sqrt()).detach().cpu()
                sums_dist += torch.bincount(codes_cpu, minlength=num_e, weights=dists_cpu).float()

                total_samples_epoch += codes_cpu.numel()

                # debug leve só no começo da 1ª época
                if epoch == 0 and i < 2:
                    m, s = x.mean().item(), x.std().item()
                    mn, mx = x.min().item(), x.max().item()
                    logger.debug(f"[batch{i:04d}] x: mean={m:.3f} std={s:.3f} min={mn:.3f} max={mx:.3f}")


            # ====== BACKWARD / OPT ======
            scaler.scale(loss).backward()

            if (i + 1) % acc_steps == 0:
                if GRAD_CLIP and GRAD_CLIP > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(opt)
                scaler.update()
                with torch.no_grad():
                    model.enc_scale.clamp_(0.05, 1.0)   # só ajuste de faixa, sem log extra
                opt.zero_grad(set_to_none=True)
                step_in_epoch += 1

            total_loss += float(loss.item()) * acc_steps
            total_vq   += float(vq_loss.item())
            total_mse  += float(recon_loss.item())

        # flush final se loop acabou no meio de um acc_steps
        if (i + 1) % acc_steps != 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            step_in_epoch += 1

        dt = time.time() - t0

        # ====== RESUMO POR ÉPOCA ======
        # uso do codebook na época
        hist = epoch_used_hist.float()
        total_counts = hist.sum().clamp_min(1.0)
        p_epoch = (hist / total_counts).clamp_min(1e-12)
        entropy_epoch = - (p_epoch * p_epoch.log()).sum().item()
        perplexity = float(np.exp(entropy_epoch))
        ativos_epoch = int((hist > 0).sum().item())

        # dist média por código (para CSV/plot, não printar tudo)
        hist_np = hist.numpy()
        sums_np = sums_dist.numpy()
        avg_dist_per_code = np.zeros(num_e, dtype=float)
        nz = hist_np > 0
        avg_dist_per_code[nz] = sums_np[nz] / hist_np[nz]

        # agregados médios do batch
        mean_znorm_epoch = (agg_mean_znorm / max(agg_batches, 1))
        mean_dist_epoch  = (agg_mean_dist / max(agg_batches, 1))

        # estatísticas compactas do histograma
        h_mean = float(hist_np.mean())
        h_median = float(np.median(hist_np))
        h_p5, h_p95 = _p5_p95(hist_np[nz]) if nz.any() else (0.0, 0.0)

        # ====== PRINT ENXUTO: SÓ 2 LINHAS ======
        try:
            evs = len(ds)
        except Exception:
            evs = total_samples_epoch

        logger.info(
            f"[{epoch+1}/{NUM_EPOCHS}] "
            f"Loss={total_loss:.3f} | Recon={total_mse:.3f} | VQ={total_vq:.3f} | "
            f"Perplx={perplexity:.1f} | Ativos={ativos_epoch}/{num_e} | "
            f"||z_e||={mean_znorm_epoch:.3f} | ||z_e-z_q||={mean_dist_epoch:.3f} | "
            f"{evs / max(dt, 1e-6):.0f} ev/s"
        )
        logger.info(
            "[EMB] used=%d/%d count{mean=%.1f,med=%.1f,p5=%.1f,p95=%.1f}"
            % (ativos_epoch, num_e, h_mean, h_median, h_p5, h_p95)
        )

        # ====== ARQUIVOS (sem poluir console) ======
        # salva CSV compacto por época (count e dist média por código)
        try:
            (OUT_DIR / "vq_diagnostics").mkdir(parents=True, exist_ok=True)
            np.savetxt(
                OUT_DIR / f"vq_diagnostics/epoch_{epoch}.csv",
                np.vstack([hist_np, avg_dist_per_code]).T,
                delimiter=",",
                header="count,avg_dist",
                fmt="%.6f",
            )
        except Exception as e:
            logger.warning(f"Falha ao salvar CSV de diagnóstico: {e}")

        # salva figura do uso (log1p(count))
        try:
            plt.figure(figsize=(8, 3))
            plt.plot(np.log1p(hist_np))
            plt.title(f"Code usage epoch {epoch+1} — active {ativos_epoch}/{num_e} — perplexity {perplexity:.1f}")
            plt.xlabel("code index"); plt.ylabel("log(1+count)")
            plt.tight_layout()
            plt.savefig(str(OUT_DIR / f"code_usage_ep{epoch+1:03d}.png"))
            plt.close()
        except Exception as e:
            logger.warning(f"Falha ao plotar histograma: {e}")

        # resurrect (se houver buffer)
        if z_buf_count > 0:
            try:
                z_sample = torch.cat(z_buf_list, dim=0)
                if z_sample.shape[0] > 4096:
                    idx = torch.randperm(z_sample.shape[0])[:4096]
                    z_sample = z_sample[idx]
                n_res = model.vq.resurrect_dead(z_sample, min_count=RESURRECT_MIN_COUNT)
                if n_res > 0:
                    logger.info(f"[epoch {epoch+1}] resurrected {n_res} dead embeddings")
            except Exception as e:
                logger.warning(f"Falha ao ressuscitar embeddings: {e}")

        # checkpoint (sem spam extra)
        ckpt = {
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'epoch_used_hist': epoch_used_hist.numpy(),
            'perplexity': perplexity,
            'ativos': ativos_epoch,
            'config': {
                'input_dim': INPUT_DIM,
                'hidden_dims': HIDDEN_DIMS,
                'embedding_dim': EMBED_DIM,
                'num_embeddings': NUM_EMBEDDINGS,
                'ema_decay': float(model.vq.decay),
                'ema_eps': EMA_EPS,
                'commit_beta': COMMIT_BETA,
                'lr': LR,
                'batch_size': BATCH_SIZE,
                'accumulate_all': ACCUMULATE_ALL,
                'acc_steps': acc_steps,
                'epochs': NUM_EPOCHS,
            }
        }
        torch.save(ckpt, OUT_DIR / f"vqvae_ep{epoch+1:03d}.pt")

    # final save
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'config': ckpt['config'],
        },
        OUT_DIR / "vqvae_final.pt",
    )
    logger.info("Modelo final salvo em vqvae_final.pt")



if __name__ == "__main__":
    main()
