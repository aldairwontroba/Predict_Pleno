# transformer_train_fixed.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import logging
from typing import List, Tuple
import time
from collections import Counter
import matplotlib.pyplot as plt
import math
import random

from src.config import PATHS

# Configuração
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ======= CONFIG MELHORADA =======

# VQVAE grande que você escolheu
VQVAE_CHECKPOINT = str(PATHS.models / "vq_out" / "vqvae_best_1311.pt")

DATA_DIR = PATHS.eventos_processados
PATTERNS = ["*_norm_norm.npy", "*_norm.npy"]
DL = np.r_[0:4, 6:11, 12:23, 28, 31, 34, 37:59]

# Hiperparâmetros mais modestos
SEQ_LEN = 32
BATCH_SIZE = 64
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4

EMBED_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 3
DROPOUT = 0.3

VAL_RATIO = 0.1
MIN_SEQUENCE_LENGTH = 100

EARLY_STOP_PATIENCE = 3

# ======= SEED PARA REPRODUTIBILIDADE =======
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# ======= ANÁLISE DE TOKENS =======
def analyze_tokens(token_sequences: List[torch.Tensor], vocab_size: int):
    """
    Analisa a distribuição dos tokens e calcula alguns baselines de perplexidade.

    - Distribuição global (unigram)
    - Perplexidade se eu SEMPRE prever o token mais frequente
    - Perplexidade baseada na entropia unigram (sem contexto)
    - Checa faixa [min, max] e se está dentro de [0, vocab_size-1]
    """
    all_tokens = torch.cat(token_sequences).cpu().numpy()
    token_counts = Counter(all_tokens.tolist())

    unique_tokens = len(token_counts)
    total_tokens = len(all_tokens)

    usage_rate = unique_tokens / float(vocab_size)
    tmin = int(all_tokens.min())
    tmax = int(all_tokens.max())

    logger.info(f"[TOKENS] Total tokens: {total_tokens:,}")
    logger.info(f"[TOKENS] Tokens únicos: {unique_tokens} / {vocab_size} (uso = {usage_rate:.1%})")
    logger.info(f"[TOKENS] Faixa de IDs: min={tmin}, max={tmax}")

    if tmin < 0 or tmax >= vocab_size:
        logger.error(
            f"[TOKENS] ERRO: existem tokens fora do intervalo [0, vocab_size-1]! "
            f"(min={tmin}, max={tmax}, vocab_size={vocab_size})"
        )

    # Top tokens mais frequentes
    top_tokens = token_counts.most_common(10)
    logger.info("[TOKENS] Top 10 tokens mais frequentes (freq relativa):")
    for token, count in top_tokens:
        logger.info(f"  Token {token:6d}: {count/total_tokens:.3%}")

    # 1) Sempre prever o token mais frequente
    most_freq_token, most_freq_count = top_tokens[0]
    p_max = most_freq_count / total_tokens
    const_model_perplex = 1.0 / p_max
    logger.info(
        f"[TOKENS] Baseline perplexidade (sempre token mais frequente={most_freq_token}): "
        f"{const_model_perplex:.2f}"
    )

    # 2) Perplexidade baseada na distribuição unigram (sem contexto)
    probs = np.array(list(token_counts.values()), dtype=np.float64) / float(total_tokens)
    entropy = -np.sum(probs * np.log(probs + 1e-12))
    unigram_perplex = float(np.exp(entropy))
    logger.info(
        f"[TOKENS] Perplexidade baseada na distribuição global (unigram, sem contexto): "
        f"{unigram_perplex:.2f}"
    )

    # Plot distribuição
    counts = list(token_counts.values())

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(counts, bins=50, log=True)
    plt.title('Distribuição de Frequência dos Tokens')
    plt.xlabel('Frequência')
    plt.ylabel('Número de Tokens (log)')

    plt.subplot(1, 2, 2)
    plt.plot(sorted(counts, reverse=True)[:1000])
    plt.title('Top 1000 Tokens (frequência)')
    plt.xlabel('Rank')
    plt.ylabel('Frequência')

    plt.tight_layout()
    plt.savefig('token_distribution.png')
    plt.close()

    return token_counts, const_model_perplex, unigram_perplex


# ======= DATASET COM VALIDAÇÃO =======
class TokenDataset(Dataset):
    """
    Cada item é uma janela de tamanho seq_len+1.
    x = seq[:-1], y = seq[1:] (next-token).
    """
    def __init__(self, token_sequences: List[torch.Tensor], seq_len: int):
        self.seq_len = seq_len
        self.sequences = []

        for tokens in token_sequences:
            if len(tokens) > seq_len:
                # stride 1: maximiza número de exemplos
                # garante que é long
                tokens = tokens.to(torch.long)
                for i in range(0, len(tokens) - seq_len):
                    self.sequences.append(tokens[i:i + seq_len + 1])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        x = seq[:-1]
        y = seq[1:]
        return x, y


# ======= MODELO TRANSFORMER (ENCODER-ONLY COM MÁSCARA CAUSAL) =======
class NextTokenTransformer(nn.Module):
    """
    Modelo decoder-only correto usando TransformerEncoder com máscara causal.

    Importante: não usa TransformerDecoder com memory=h para evitar
    vazamento de informação do futuro via cross-attention.
    """
    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int,
                 num_layers: int, seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.seq_len = seq_len

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(seq_len, embed_dim)

        self.embed_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.layer_norm = nn.LayerNorm(embed_dim)
        self.output_head = nn.Linear(embed_dim, vocab_size)

        # Máscara causal fixa (ajustada por seq_len no forward)
        causal = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        self.register_buffer("causal_mask", causal)

    def forward(self, x: torch.Tensor):
        """
        x: [B, T] de tokens (int64)
        retorna logits: [B, T, vocab_size]
        """
        batch_size, seq_len = x.shape

        tok_emb = self.token_embedding(x)               # [B, T, D]
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embedding(positions)         # [B, T, D]

        h = tok_emb + pos_emb
        h = self.embed_dropout(h)

        # Ajusta máscara para o T atual
        tgt_mask = self.causal_mask[:seq_len, :seq_len]

        # Encoder com máscara causal → cada posição só vê passado/presente
        h = self.transformer(
            h,
            mask=tgt_mask
        )  # [B, T, D]

        h = self.layer_norm(h)
        logits = self.output_head(h)                    # [B, T, V]

        return logits


# ======= GERAR TOKENS =======
def generate_tokens(
    vqvae_model,
    data_dir: Path,
    patterns: List[str],
    cols: np.ndarray,
    device: torch.device,
    max_files: int = None,
    file_list: List[Path] = None,
) -> List[torch.Tensor]:
    """
    Converte eventos em tokens mantendo a ordem temporal DENTRO de cada arquivo.
    Se file_list for fornecido, usa exatamente esses arquivos (cada um = 1 'dia').
    Caso contrário, usa list_first_nonempty + patterns.
    """
    from src.vqvae.train import list_first_nonempty

    if file_list is None:
        files = list_first_nonempty(data_dir, patterns)
        files.sort()
    else:
        files = [Path(f) for f in file_list]

    if max_files:
        files = files[:max_files]

    logger.info(f"Processando {len(files)} arquivos para gerar tokens...")

    all_tokens = []

    with torch.no_grad():
        for file_idx, file_path in enumerate(files):
            try:
                data = np.load(file_path, mmap_mode='r')
                if cols is not None:
                    data = data[:, cols]

                # Pula arquivos muito curtos (dia sem eventos suficientes)
                if len(data) < MIN_SEQUENCE_LENGTH:
                    logger.info(f"Arquivo {file_path.name} pulado (len={len(data)} < {MIN_SEQUENCE_LENGTH})")
                    continue

                tokens_file = []
                batch_size = 4096

                # Mantém ORDEM dos eventos dentro do arquivo
                for i in range(0, len(data), batch_size):
                    batch = data[i:i + batch_size]
                    batch_tensor = torch.from_numpy(batch.astype(np.float32)).to(device)

                    z_e = vqvae_model.encoder(batch_tensor)
                    z_e = vqvae_model.enc_ln(z_e)
                    z_e = torch.tanh(z_e) * vqvae_model.enc_scale

                    if hasattr(vqvae_model.vq, "levels"):
                        _, codes_list, _ = vqvae_model.vq(z_e)
                        codes = codes_list[0]   # nível 0
                    else:
                        _, codes, _ = vqvae_model.vq(z_e)

                    tokens_file.append(codes.cpu())

                if tokens_file:
                    tokens_concatenated = torch.cat(tokens_file, dim=0)
                    all_tokens.append(tokens_concatenated)

                    logger.info(
                        f"Arquivo {file_idx + 1}/{len(files)}: {file_path.name} "
                        f"-> {len(tokens_concatenated)} tokens"
                    )

            except Exception as e:
                logger.warning(f"Erro no arquivo {file_path}: {e}")
                continue

    return all_tokens


# ======= FUNÇÕES DE DEBUG / SANITY CHECK =======
def debug_check_dataset_alignment(dataset: TokenDataset, name: str, num_checks: int = 3):
    """
    Verifica se, para alguns exemplos aleatórios do dataset, vale:
        y[t] == x[t+1]
    e loga tipos e faixas de tokens.
    """
    logger.info(f"[DEBUG] Verificando alinhamento x->y no dataset {name}...")

    n = len(dataset)
    if n == 0:
        logger.warning(f"[DEBUG] Dataset {name} está vazio!")
        return

    idxs = random.sample(range(n), min(num_checks, n))

    for j, idx in enumerate(idxs):
        x, y = dataset[idx]
        if x.shape[0] != y.shape[0]:
            logger.error(
                f"[DEBUG] Dataset {name}: tamanhos diferentes em idx={idx}: "
                f"x.shape={x.shape}, y.shape={y.shape}"
            )
            return

        # checa shift
        if not torch.all(y[:-1] == x[1:]):
            logger.error(f"[DEBUG] Dataset {name}: falha no alinhamento x->y em idx={idx}")
            logger.error(f"  x[:10] = {x[:10].tolist()}")
            logger.error(f"  y[:10] = {y[:10].tolist()}")
            return

        if j == 0:
            # pega um exemplo para logar faixa/tipos
            logger.info(
                f"[DEBUG] {name} exemplo idx={idx}: "
                f"x.dtype={x.dtype}, y.dtype={y.dtype}, "
                f"x.min={int(x.min())}, x.max={int(x.max())}"
            )

    logger.info(f"[DEBUG] Dataset {name}: alinhamento x->y OK nos {len(idxs)} exemplos verificados.")


def debug_sample_batch(loader: DataLoader, name: str, device: torch.device, vocab_size: int):
    """
    Pega o primeiro batch do loader e loga formas, tipos, alguns tokens
    e checa faixa de IDs vs vocab_size.
    """
    try:
        x, y = next(iter(loader))
    except StopIteration:
        logger.warning(f"[DEBUG] Loader {name} vazio, não há batch para inspecionar.")
        return

    logger.info(f"[DEBUG] {name} batch inicial: x.shape={x.shape}, y.shape={y.shape}")
    logger.info(f"[DEBUG] {name} dtypes: x.dtype={x.dtype}, y.dtype={y.dtype}")

    x0 = x[0]
    y0 = y[0]
    logger.info(f"[DEBUG] {name} x[0][:16] = {x0[:16].tolist()}")
    logger.info(f"[DEBUG] {name} y[0][:16] = {y0[:16].tolist()}")

    # checa shift
    if x0.shape[0] > 1:
        shift_ok = torch.all(x0[1:] == y0[:-1])
        logger.info(f"[DEBUG] {name} shift x->y no primeiro exemplo OK? {bool(shift_ok)}")

    xmin = int(x.min())
    xmax = int(x.max())
    ymin = int(y.min())
    ymax = int(y.max())

    logger.info(
        f"[DEBUG] {name} faixa de tokens no batch: "
        f"x.min={xmin}, x.max={xmax}, y.min={ymin}, y.max={ymax}, vocab_size={vocab_size}"
    )

    if xmin < 0 or ymin < 0 or xmax >= vocab_size or ymax >= vocab_size:
        logger.error(
            f"[DEBUG] {name} ERRO: tokens fora do intervalo [0, vocab_size-1] "
            f"(x[{xmin}, {xmax}], y[{ymin}, {ymax}])"
        )


# ======= TREINO COM VALIDAÇÃO =======
def train_transformer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Usando dispositivo: {device}")

    # 1. Carrega VQVAE
    vqvae_model, vq_config, vocab_size = load_vqvae(VQVAE_CHECKPOINT, device)

    # 2. Escolher dias para treino e validação
    from src.vqvae.train import list_first_nonempty

    all_files = list_first_nonempty(DATA_DIR, PATTERNS)
    all_files = sorted(all_files)

    if not all_files:
        raise ValueError("Nenhum arquivo encontrado em DATA_DIR com os padrões especificados.")

    # Embaralha os arquivos (dias) de forma reprodutível
    random.seed(42)
    random.shuffle(all_files)

    n_files = len(all_files)
    n_val_files = max(1, int(n_files * VAL_RATIO))  # p.ex. 10% dos dias para validação

    val_files = all_files[:n_val_files]
    train_files = all_files[n_val_files:]

    logger.info(f"Total de arquivos (dias): {n_files}")
    logger.info(f"  → {len(train_files)} para TREINO")
    logger.info(f"  → {len(val_files)}   para VALIDAÇÃO")
    logger.info("Alguns arquivos de treino:")
    for f in train_files[:5]:
        logger.info(f"  [train] {Path(f).name}")
    logger.info("Alguns arquivos de validação:")
    for f in val_files[:5]:
        logger.info(f"  [val]   {Path(f).name}")

    # 3. Gera tokens separadamente: treino vs validação (por dia)
    tokens_train = generate_tokens(
        vqvae_model, DATA_DIR, PATTERNS, DL, device,
        max_files=None,
        file_list=train_files,
    )

    tokens_val = generate_tokens(
        vqvae_model, DATA_DIR, PATTERNS, DL, device,
        max_files=None,
        file_list=val_files,
    )

    if not tokens_train:
        raise ValueError("Nenhum token gerado para TREINO. Verifique MIN_SEQUENCE_LENGTH ou arquivos.")
    if not tokens_val:
        raise ValueError("Nenhum token gerado para VALIDAÇÃO. Verifique MIN_SEQUENCE_LENGTH ou divisão de arquivos.")

    # 4. Análise de tokens (treino+val)
    token_sequences_global = tokens_train + tokens_val
    token_counts, const_perplex, unigram_perplex = analyze_tokens(token_sequences_global, vocab_size)
    logger.info(f"[INFO] Baseline perplex (constante): {const_perplex:.2f}")
    logger.info(f"[INFO] Baseline perplex (unigram):  {unigram_perplex:.2f}")

    # 5. Cria datasets SEM random_split (já estão separados por dia)
    train_dataset = TokenDataset(tokens_train, SEQ_LEN)
    val_dataset   = TokenDataset(tokens_val,   SEQ_LEN)

    logger.info(f"Dataset: {len(train_dataset)} janelas TREINO, {len(val_dataset)} janelas VALIDAÇÃO")

    # Sanity check: alinhamento x->y
    debug_check_dataset_alignment(train_dataset, "TREINO")
    debug_check_dataset_alignment(val_dataset,   "VAL")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Debug do primeiro batch
    debug_sample_batch(train_loader, "TREINO", device, vocab_size)
    debug_sample_batch(val_loader,   "VAL",    device, vocab_size)

    # 6. Modelo
    model = NextTokenTransformer(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        seq_len=SEQ_LEN,
        dropout=DROPOUT
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Transformer: {n_params:,} parâmetros")

    # 7. Otimizador e loss
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.1)
    criterion = nn.CrossEntropyLoss()  # sem padding → não precisa ignore_index

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    best_val_loss = float("inf")
    epochs_no_improve = 0

    # 8. Loop de treino
    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()

        # === Treino ===
        model.train()
        train_loss_sum = 0.0
        train_batches = 0
        correct_tokens = 0
        total_tokens = 0

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)  # [B, T, V]
            loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))

            if not torch.isfinite(loss):
                logger.error(f"[TREINO] Loss não finita detectada no batch {batch_idx}: {loss}")
                return

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss_sum += loss.item()
            train_batches += 1

            # Accuracy por batch
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                correct_tokens += (preds == y).sum().item()
                total_tokens += y.numel()

            # Debug do primeiro batch da primeira época
            if epoch == 0 and batch_idx == 0:
                logger.info(f"[DEBUG] Epoch 1, Batch 0 - exemplo de preds vs targets:")
                logger.info(f"  y[0][:16]     = {y[0, :16].tolist()}")
                logger.info(f"  preds[0][:16] = {preds[0, :16].tolist()}")

            if batch_idx % 100 == 0:
                batch_perplex = math.exp(loss.item())
                batch_acc = correct_tokens / max(total_tokens, 1)
                logger.info(
                    f"[TREINO] Epoch {epoch+1}, Batch {batch_idx}, "
                    f"Loss: {loss.item():.4f}, Perplex: {batch_perplex:.2f}, "
                    f"Acc (acum): {batch_acc:.3f}"
                )

        avg_train_loss = train_loss_sum / max(train_batches, 1)
        train_losses.append(avg_train_loss)
        train_perplex = math.exp(avg_train_loss)
        train_acc = correct_tokens / max(total_tokens, 1)
        train_accuracies.append(train_acc)

        # === Validação ===
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        val_correct = 0
        val_total = 0

        val_pred_counter = Counter()
        val_target_counter = Counter()

        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(val_loader):
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))

                if not torch.isfinite(loss):
                    logger.error(f"[VAL] Loss não finita detectada no batch {batch_idx}: {loss}")
                    return

                val_loss_sum += loss.item()
                val_batches += 1

                preds = logits.argmax(dim=-1)
                val_correct += (preds == y).sum().item()
                val_total += y.numel()

                # Amostra para ver distribuição de preds vs targets
                if batch_idx < 5:  # só primeiros batches para não ficar pesado
                    flat_preds = preds.view(-1).cpu().numpy()
                    flat_targets = y.view(-1).cpu().numpy()

                    # subsample se muito grande
                    if flat_preds.size > 5000:
                        idx = np.random.choice(flat_preds.size, size=5000, replace=False)
                        flat_preds = flat_preds[idx]
                        flat_targets = flat_targets[idx]

                    val_pred_counter.update(flat_preds.tolist())
                    val_target_counter.update(flat_targets.tolist())

        avg_val_loss = val_loss_sum / max(val_batches, 1)
        val_losses.append(avg_val_loss)
        val_perplex = math.exp(avg_val_loss)
        val_acc = val_correct / max(val_total, 1)
        val_accuracies.append(val_acc)

        epoch_time = time.time() - epoch_start

        logger.info(f"Epoch {epoch+1}/{NUM_EPOCHS} (tempo: {epoch_time:.1f}s)")
        logger.info(
            f"  Treino    - Loss: {avg_train_loss:.4f}, Perplex: {train_perplex:.2f}, "
            f"Acc: {train_acc:.3f}"
        )
        logger.info(
            f"  Validação - Loss: {avg_val_loss:.4f}, Perplex: {val_perplex:.2f}, "
            f"Acc: {val_acc:.3f}"
        )

        # Comparar com baseline:
        logger.info(
            f"  Comparação baseline perplex (const={const_perplex:.1f}, unigram={unigram_perplex:.1f})"
        )

        # Debug da distribuição de preds vs targets na validação
        def _log_top(counter, label):
            if not counter:
                logger.info(f"[DEBUG] Epoch {epoch+1} {label}: contador vazio.")
                return
            total = sum(counter.values())
            top = counter.most_common(5)
            msg = ", ".join([f"{tok}:{cnt} ({cnt/total:.1%})" for tok, cnt in top])
            logger.info(f"[DEBUG] Epoch {epoch+1} {label} top5: {msg}")

        _log_top(val_target_counter, "VAL target")
        _log_top(val_pred_counter,   "VAL pred")

        # Early stopping + salvar melhor modelo
        if avg_val_loss < best_val_loss - 1e-4:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "transformer_best.pt")
            logger.info("  ↳ Novo melhor modelo salvo!")
        else:
            epochs_no_improve += 1
            logger.info(f"  ↳ Sem melhora no val_loss por {epochs_no_improve} epoch(s)")
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                logger.info("Early stopping acionado.")
                break

    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig('training_loss.png')
    plt.close()

    # Plot acurácia
    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label='Train Acc')
    plt.plot(val_accuracies, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.savefig('training_accuracy.png')
    plt.close()

    logger.info("Treino completo!")


# Função load_vqvae
def load_vqvae(checkpoint_path: str, device: torch.device):
    from src.vqvae.train import VQVAE

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']

    model = VQVAE(
        input_dim=config['input_dim'],
        hidden=config['hidden_dims'],
        emb_dim=config['embedding_dim'],
        num_emb=config['num_embeddings'],
        commit_beta=config['commit_beta'],
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # ajuste de vocab_size pra combinar com RVQ
    if hasattr(model.vq, "levels"):
        vocab_per_level = model.vq.num_embeddings if hasattr(model.vq, "num_embeddings") else config['num_embeddings']
        vocab_size = vocab_per_level
    else:
        vocab_size = config['num_embeddings']

    logger.info(f"VQVAE carregado: {checkpoint_path}")
    logger.info(f"[VQ] vocab_size derivado = {vocab_size}")
    return model, config, vocab_size


if __name__ == "__main__":
    train_transformer()
