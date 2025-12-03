import time, random, os
from pathlib import Path
from typing import List, Tuple, Iterator
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import IterableDataset, DataLoader
import matplotlib.pyplot as plt

# ======= CONFIG =======
DATA_DIR = Path(r"C:\Users\Aldair\Desktop\eventos_processados")
PATTERNS = ["*_norm_norm.npy", "*_norm.npy"]   # tenta primeiro _norm_norm
DL = np.r_[0:4,6:11,12:23,28,31,34,37:59]
INPUT_DIM = len(DL)

BATCH_SIZE = 2048 * 32
ACCUMULATE_ALL = False
ACC_STEPS = 32
LR = 2e-4
NUM_EMBEDDINGS = 512 * 4
EMBED_DIM = 128
HIDDEN_DIMS = (2048, 1024, 512)
NUM_WORKERS = 4
GRAD_CLIP = 0.3
EMA_DECAY = 0.995
EMA_EPS = 1e-5
COMMIT_BETA = 1.5
NUM_EPOCHS = 20
WEIGHT_DECAY = 1e-6
SEED = 1337
PERSISTENT_WORKERS = True
PRINT_PER_EPOCH = True
DROP_LAST = False

# tau schedule (anneal)
TAU_START = 1.8
TAU_END = 1.0

# EMA warmup epochs
EMA_WARMUP_EPOCHS = 2
EMA_WARMUP_DECAY = 0.95

# resurrect params
RESURRECT_MIN_COUNT = 1.0
Z_BUF_MAX = 16384   # máximo de vetores z_e a guardar para re-seed

# output dir
OUT_DIR = Path("./vq_out")
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
    print(f"[kmeans] collected {Z.shape[0]} z vectors for kmeans (emb_dim={Z.shape[1]})")

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

    print("[kmeans] codebook and EMA buffers initialized.")
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

# ======= DATASET por CHUNKS =======
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

        # 2 níveis × 1024 = 2048 totais (ajuste se quiser 3 níveis, etc.)
        n_levels = 2
        self.vq = ResidualVectorQuantizerEMA(
            n_levels=n_levels,
            num_embeddings=num_emb // n_levels,
            embedding_dim=emb_dim,
            decay=EMA_DECAY,
            eps=EMA_EPS
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
            recon = F.mse_loss(x_hat, x, reduction="mean")
            vq_loss = self.commit_beta * commit_total

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
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")

    files = list_first_nonempty(DATA_DIR, PATTERNS)
    if not files:
        raise FileNotFoundError(f"Nenhum arquivo em {DATA_DIR}")
    print(f"[INFO] arquivos: {len(files)}")

    ds = NpyChunkDataset(files, DL, batch_size=BATCH_SIZE, shuffle=True)
    dl = DataLoader(
        ds, batch_size=None, num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=PERSISTENT_WORKERS
    )

    model = VQVAE().to(device)

    # chamando init (executar APENAS UMA VEZ)
    print("[INIT] inicializando codebook com kmeans. isso pode levar alguns segs/mins...")
    init_codebook_with_kmeans(model, dl, device, n_samples=200000, batch_collect=4096)
    print("[INIT] feito. prosseguindo para o treinamento.")
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

    print(f"[INFO] acc_steps: {acc_steps} (batch efetivo ≈ {acc_steps*BATCH_SIZE:,} eventos)")

    # buffer for z samples to resurrect dead codes
    z_buf_list = []  # list of tensors (cpu)
    z_buf_count = 0

    # ensure diagnostics dir exists
    (OUT_DIR / "vq_diagnostics").mkdir(parents=True, exist_ok=True)

    num_e = model.vq.num_embeddings

    for epoch in range(NUM_EPOCHS):
        # tau anneal and EMA warmup
        model.current_tau = TAU_START + (TAU_END - TAU_START) * (epoch / max(1, NUM_EPOCHS - 1))
        if epoch < EMA_WARMUP_EPOCHS:
            model.vq.decay = EMA_WARMUP_DECAY
        else:
            model.vq.decay = float(EMA_DECAY)

        model.train()
        total_loss = total_vq = total_mse = 0.0
        step_in_epoch = 0
        t0 = time.time()

        opt.zero_grad(set_to_none=True)

        # histogram/counts and sum of distances for this epoch (CPU tensors)
        epoch_used_hist = torch.zeros(num_e, dtype=torch.long)
        sums_dist = torch.zeros(num_e, dtype=torch.float)  # accumulate sum of distances per code
        total_samples_epoch = 0

        for i, batch in enumerate(dl):
            x = torch.nan_to_num(batch, nan=0.0, posinf=0.0, neginf=0.0).to(device, non_blocking=True)

            with amp_ctx:
                # expected return: x_hat, vq_loss, recon_loss, codes, z_e, z_q
                # make sure your model.forward returns z_q (quantized vectors) and codes
                x_hat, vq_loss, recon_loss, codes, z_e, z_q = model(x)

                # divide by acc_steps because we accumulate gradients
                loss = (recon_loss + vq_loss) / acc_steps

            # diagnostics (cheap, execute em CPU quando possível)
            with torch.no_grad():
                mean_norm = z_e.norm(dim=1).mean().item()
                mean_dist = ((z_e - z_q).pow(2).sum(dim=1).sqrt()).mean().item()

                # bincount em CPU (evita mem overhead na GPU)
                codes_cpu = codes.detach().cpu()
                counts = torch.bincount(codes_cpu, minlength=num_e).float()
                p = counts / counts.sum() if counts.sum() > 0 else counts
                # perplexity numeric stable
                mask = p > 0
                entropy = - (p[mask] * (p[mask].log())).sum().item() if mask.any() else 0.0
                perplexity = float(np.exp(entropy)) if entropy > 0 else 0.0
                active = int((counts > 0).sum().item())

                if i % 100 == 0:  # ajuste frequência
                    print(f"[DIAG] mean_z_norm={mean_norm:.3f} mean_dist={mean_dist:.4f} active={active}/{num_e} perp={perplexity:.1f}")

            # accumulate z_e sample buffer (cpu)
            if z_e is not None:
                ze_cpu = z_e.detach().cpu()
                if z_buf_count + ze_cpu.shape[0] <= Z_BUF_MAX:
                    z_buf_list.append(ze_cpu)
                    z_buf_count += ze_cpu.shape[0]
                else:
                    needed = max(0, Z_BUF_MAX - z_buf_count)
                    if needed > 0:
                        z_buf_list.append(ze_cpu[:needed])
                        z_buf_count += needed
                    # skip the rest to avoid heavy ops

            # accumulate code histogram AND sums of distances (per-code)
            cb = codes.detach().cpu()
            epoch_used_hist += torch.bincount(cb, minlength=num_e)

            # compute distances per sample and accumulate sums via bincount weights
            dists_cpu = ((z_e - z_q).pow(2).sum(dim=1).sqrt()).detach().cpu()
            sums_dist += torch.bincount(cb, minlength=num_e, weights=dists_cpu).float()

            total_samples_epoch += cb.numel()

            # backward
            scaler.scale(loss).backward()

            # debug small
            if epoch == 0 and i < 3:
                m, s = x.mean().item(), x.std().item()
                mn, mx = x.min().item(), x.max().item()
                print(f"[debug] batch stats: mean={m:.3f} std={s:.3f} min={mn:.3f} max={mx:.3f}")

            if (i + 1) % acc_steps == 0:
                if GRAD_CLIP and GRAD_CLIP > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                step_in_epoch += 1

            total_loss += float(loss.item()) * acc_steps
            total_vq += float(vq_loss.item())
            total_mse += float(recon_loss.item())

        # flush if ended without step (partial accumulation)
        if (i + 1) % acc_steps != 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            step_in_epoch += 1

        dt = time.time() - t0

        # compute epoch-level perplexity and active codes from epoch_used_hist
        hist = epoch_used_hist.float()
        total_counts = hist.sum().clamp_min(1.0)
        p_epoch = (hist / total_counts).clamp_min(1e-12)
        entropy_epoch = - (p_epoch * p_epoch.log()).sum().item()
        perplexity = float(np.exp(entropy_epoch))
        ativos_epoch = int((hist > 0).sum().item())

        if PRINT_PER_EPOCH:
            evs = len(ds)
            print(f"[{epoch+1}/{NUM_EPOCHS}] "
                  f"Loss: {total_loss:.6f} | Recon: {total_mse:.6f} | VQ: {total_vq:.6f} | "
                  f"Ativos: {ativos_epoch}/{num_e} | perplexity:{perplexity:.1f} | "
                  f"{evs/max(dt,1e-6):.0f} ev/s")

            # compute avg_dist_per_code (safe division)
            hist_np = hist.numpy()
            sums_np = sums_dist.numpy()
            avg_dist_per_code = np.zeros(num_e, dtype=float)
            nz = hist_np > 0
            avg_dist_per_code[nz] = sums_np[nz] / hist_np[nz]

            print("[EMB_STATS] used=%d/%d mean_count=%.2f median_count=%.1f"
                  % (int((hist_np>0).sum()), num_e, hist_np.mean(), float(np.median(hist_np))))
            np.savetxt(OUT_DIR / f"vq_diagnostics/epoch_{epoch}.csv",
                       np.vstack([hist_np, avg_dist_per_code]).T,
                       delimiter=",", header="count,avg_dist", fmt="%.6f")

        # plot histogram (log1p)
        try:
            hnp = epoch_used_hist.numpy()
            plt.figure(figsize=(8,3))
            plt.plot(np.log1p(hnp))
            plt.title(f"Code usage epoch {epoch+1} — active {ativos_epoch}/{num_e} — perplexity {perplexity:.1f}")
            plt.xlabel("code index")
            plt.ylabel("log(1+count)")
            plt.tight_layout()
            plt.savefig(str(OUT_DIR / f"code_usage_ep{epoch+1:03d}.png"))
            plt.close()
        except Exception as e:
            print("plot hist failed:", e)

        # resurrect dead codes using z_buf (if any)
        if z_buf_count > 0:
            try:
                z_sample = torch.cat(z_buf_list, dim=0)
                # shuffle and maybe subsample to keep cost low
                if z_sample.shape[0] > 4096:
                    idx = torch.randperm(z_sample.shape[0])[:4096]
                    z_sample = z_sample[idx]
                n_res = model.vq.resurrect_dead(z_sample, min_count=RESURRECT_MIN_COUNT)
                if n_res > 0:
                    print(f"[epoch {epoch+1}] resurrected {n_res} dead embeddings")
            except Exception as e:
                print("resurrect failed:", e)

        # save checkpoint
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
        print(f"[SAVE] {OUT_DIR / f'vqvae_ep{epoch+1:03d}.pt'}")

    # final save
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'config': ckpt['config']
    }, OUT_DIR / "vqvae_final.pt")
    print("[SAVE] final model")


if __name__ == "__main__":
    main()
