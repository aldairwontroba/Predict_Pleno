import os
from dataclasses import dataclass
from typing import Optional, List

import torch
from torch.utils.data import DataLoader

# ============================================================
# IMPORTS – AJUSTE CONFORME SEU PROJETO
# ============================================================

from src.agent.model import AgentTransformer, TransformerConfig, TokenConfig
from src.agent.dataset import AgentSequenceDataset
from src.agent.build_sequences import AgentWindowConfig
from src.config import PATHS


# ============================================================
# CARREGAR MELHOR CHECKPOINT
# ============================================================

def load_best_agent(model: torch.nn.Module, ckpt_dir: str, device: torch.device):
    ckpt_path = os.path.join(ckpt_dir, "stf_epoch_001.pt")
    ckpt = torch.load(ckpt_path, map_location=device)

    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    print(
        f"[LOAD] Melhor modelo carregado de {ckpt_path} | "
        f"epoch={ckpt.get('epoch')} val_loss={ckpt.get('val_loss'):.4f}"
    )
    return model


# ============================================================
# BASELINES ESTATÍSTICOS (validação)
# ============================================================

@torch.no_grad()
def compute_baselines(
    ds_val,
    tok_cfg: TokenConfig,
    win_cfg: AgentWindowConfig,
    device: torch.device,
    batch_size: int = 256,
    max_batches: int = 500,
    num_workers: int = 0,
):
    """
    Calcula baseline de "sempre prever classe majoritária" para:
      - FUT: token-level
      - ANALYSIS: token-level
      - ACTION: token-level e decisão (primeiro token do bloco)
    """

    dl = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    layout = win_cfg.layout()
    fut_s, fut_e = layout["fut"]
    ana_s, ana_e = layout["analysis"]
    act_s, act_e = layout["action"]

    # Contadores globais
    # FUT
    fut_counts = {}
    fut_total = 0

    # ANALYSIS
    ana_counts = {}
    ana_total = 0

    # ACTION
    act_counts = {}
    act_total = 0

    # ACTION por janela (usa o primeiro token do bloco como "decisão")
    act_decision_counts = {}
    act_decision_total = 0

    for b_idx, x in enumerate(dl):
        if b_idx >= max_batches:
            break

        x = x.to(device)  # [B, T]
        B, T = x.shape

        # FUT
        fut_tokens = x[:, fut_s:fut_e]  # [B, n_fut]
        fut_flat = fut_tokens.reshape(-1)
        for t in fut_flat.tolist():
            fut_counts[t] = fut_counts.get(t, 0) + 1
        fut_total += fut_flat.numel()

        # ANALYSIS
        ana_tokens = x[:, ana_s:ana_e]  # [B, n_ana]
        ana_flat = ana_tokens.reshape(-1)
        for t in ana_flat.tolist():
            ana_counts[t] = ana_counts.get(t, 0) + 1
        ana_total += ana_flat.numel()

        # ACTION (token-level)
        act_tokens = x[:, act_s:act_e]  # [B, n_act]
        act_flat = act_tokens.reshape(-1)
        for t in act_flat.tolist():
            act_counts[t] = act_counts.get(t, 0) + 1
        act_total += act_flat.numel()

        # ACTION decisão (primeiro token do bloco)
        act_dec = act_tokens[:, 0]  # [B]
        for t in act_dec.tolist():
            act_decision_counts[t] = act_decision_counts.get(t, 0) + 1
        act_decision_total += act_dec.numel()

    def majority_stats(counts: dict, total: int):
        if total == 0 or not counts:
            return None, 0.0
        best_token, best_count = max(counts.items(), key=lambda kv: kv[1])
        acc = best_count / total
        return best_token, acc

    fut_maj, fut_acc = majority_stats(fut_counts, fut_total)
    ana_maj, ana_acc = majority_stats(ana_counts, ana_total)
    act_maj, act_acc = majority_stats(act_counts, act_total)
    act_dec_maj, act_dec_acc = majority_stats(act_decision_counts, act_decision_total)

    action_base = tok_cfg.RANGE_ACTION[0]

    print("============================================================")
    print("[BASELINE] Estatístico (classe majoritária na validação)")
    print("------------------------------------------------------------")
    print(f"FUT  : total_tokens={fut_total}")
    print(f"       maj_token={fut_maj}  baseline_acc={fut_acc*100:.4f}%")
    print("------------------------------------------------------------")
    print(f"ANALYSIS: total_tokens={ana_total}")
    print(f"          maj_token={ana_maj}  baseline_acc={ana_acc*100:.4f}%")
    print("------------------------------------------------------------")
    print(f"ACTION (token-level): total_tokens={act_total}")
    if act_maj is not None:
        print(f"          maj_token={act_maj}  offset={act_maj - action_base:+d}")
        print(f"          baseline_acc={act_acc*100:.4f}%")
    else:
        print("          (sem dados)")
    print("------------------------------------------------------------")
    print(f"ACTION (decisão por janela = primeiro token): total_decisions={act_decision_total}")
    if act_dec_maj is not None:
        print(f"          maj_decision_token={act_dec_maj}  offset={act_dec_maj - action_base:+d}")
        print(f"          baseline_decision_acc={act_dec_acc*100:.4f}%")
    else:
        print("          (sem dados)")
    print("============================================================")

    baselines = {
        "fut": {"maj_token": fut_maj, "acc": fut_acc, "total": fut_total},
        "analysis": {"maj_token": ana_maj, "acc": ana_acc, "total": ana_total},
        "action_token": {"maj_token": act_maj, "acc": act_acc, "total": act_total},
        "action_decision": {"maj_token": act_dec_maj, "acc": act_dec_acc, "total": act_decision_total},
    }
    return baselines


# ============================================================
# GERAÇÃO AUTORREGRESSIVA A PARTIR DE ctx+past+SEP_FUT
# ============================================================

@torch.no_grad()
def generate_autoregressive_from_prefix(
    model: torch.nn.Module,
    x_true: torch.Tensor,   # [T]
    win_cfg: AgentWindowConfig,
    device: torch.device,
):
    """
    Usa: ctx + past + SEP_FUT reais como prefixo, e gera o resto até T=512.
    Retorna sequência gerada completa [T].
    """
    layout = win_cfg.layout()
    fut_s, _ = layout["fut"]        # futuro começa logo após SEP_FUT
    T = x_true.size(0)

    # prefixo: tudo até o início de FUT (inclusive SEP_FUT na posição fut_s-1)
    prefix = x_true[:fut_s].to(device)  # [fut_s]
    x_gen = prefix.unsqueeze(0)         # [1, L0]

    model.eval()

    while x_gen.size(1) < T:
        logits = model(x_gen)                          # [1, L, V]
        next_id = logits[:, -1, :].argmax(dim=-1)      # [1]
        next_id = next_id.unsqueeze(1)                 # [1,1]
        x_gen = torch.cat([x_gen, next_id], dim=1)     # [1, L+1]

    return x_gen[0].cpu()              # [T]


# ============================================================
# AVALIAÇÃO AUTORREGRESSIVA EM N SAMPLES
# ============================================================
@torch.no_grad()
def evaluate_autoreg_on_val(
    ds_val,
    model: torch.nn.Module,
    tok_cfg: TokenConfig,
    win_cfg: AgentWindowConfig,
    device: torch.device,
    num_samples: int = 200,
    seed: int = 42,
):
    """
    Escolhe 'num_samples' aleatórios da validação, gera FUT+ANALYSIS+ACTION
    autoregressivo e compara com o ground truth.

    Além disso:
      - calcula matriz 3x3 de confusão para decisão de ACTION
        (BUY / SELL / HOLD) usando o primeiro token do bloco.
    """

    torch.manual_seed(seed)

    layout = win_cfg.layout()
    fut_s, fut_e = layout["fut"]
    ana_s, ana_e = layout["analysis"]
    act_s, act_e = layout["action"]

    action_base = tok_cfg.RANGE_ACTION[0]

    N = len(ds_val)
    num_samples = min(num_samples, N)

    # índices aleatórios (sem repetição)
    perm = torch.randperm(N)[:num_samples].tolist()

    # contadores globais
    fut_correct = 0
    fut_total = 0

    ana_correct = 0
    ana_total = 0

    act_correct = 0
    act_total = 0

    # decisão por janela (primeiro token do bloco ACTION)
    act_decision_correct = 0
    act_decision_total = 0

    # matriz de confusão 3x3: [true_class, pred_class]
    # 0 = BUY, 1 = SELL, 2 = HOLD
    conf_mat = torch.zeros(3, 3, dtype=torch.long)

    def decode_action_class(token_id: int) -> Optional[int]:
        """
        Converte um token_id em classe 0/1/2 com base no offset em relação
        a RANGE_ACTION[0]:

          off = 0 -> BUY
          off = 1 -> SELL
          off = 2 -> HOLD

        Se estiver fora desse range, retorna None.
        """
        off = token_id - action_base
        if off < 0 or off > 2:
            return None
        return int(off)

    print("============================================================")
    print(f"[AUTO-REG] Avaliando {num_samples} amostras aleatórias da validação")
    print("============================================================")

    for k, idx in enumerate(perm, start=1):
        x_true = ds_val[idx]
        if not isinstance(x_true, torch.Tensor):
            x_true = torch.tensor(x_true, dtype=torch.long)
        x_true = x_true.clone().cpu()

        # gera sequência completa a partir do prefixo
        x_pred = generate_autoregressive_from_prefix(
            model=model,
            x_true=x_true,
            win_cfg=win_cfg,
            device=device,
        )

        # FUT
        fut_true = x_true[fut_s:fut_e]
        fut_pred = x_pred[fut_s:fut_e]
        fut_total += fut_true.numel()
        fut_correct += (fut_true == fut_pred).sum().item()

        # ANALYSIS
        ana_true = x_true[ana_s:ana_e]
        ana_pred = x_pred[ana_s:ana_e]
        ana_total += ana_true.numel()
        ana_correct += (ana_true == ana_pred).sum().item()

        # ACTION (token-level)
        act_true = x_true[act_s:act_e]
        act_pred = x_pred[act_s:act_e]
        act_total += act_true.numel()
        act_correct += (act_true == act_pred).sum().item()

        # ACTION (decisão) – primeiro token
        if act_true.numel() > 0:
            true_dec = int(act_true[0].item())
            pred_dec = int(act_pred[0].item())

            true_cls = decode_action_class(true_dec)
            pred_cls = decode_action_class(pred_dec)

            if true_cls is not None and pred_cls is not None:
                conf_mat[true_cls, pred_cls] += 1

            if true_dec == pred_dec:
                act_decision_correct += 1
            act_decision_total += 1

        if (k % 20) == 0 or k == num_samples:
            print(
                f"[{k:4d}/{num_samples:4d}] "
                f"FUT_acc={fut_correct/max(1,fut_total):.4f} "
                f"ANA_acc={ana_correct/max(1,ana_total):.4f} "
                f"ACT_tok_acc={act_correct/max(1,act_total):.4f} "
                f"ACT_dec_acc={act_decision_correct/max(1,act_decision_total):.4f}"
            )

    fut_acc = fut_correct / max(1, fut_total)
    ana_acc = ana_correct / max(1, ana_total)
    act_tok_acc = act_correct / max(1, act_total)
    act_dec_acc = act_decision_correct / max(1, act_decision_total)

    print("============================================================")
    print("[AUTO-REG] RESULTADOS FINAIS (apenas validação, sem FUT real)")
    print("------------------------------------------------------------")
    print(f"FUT   : token_acc = {fut_acc*100:.4f}%  ({fut_correct}/{fut_total})")
    print(f"ANALYSIS: token_acc = {ana_acc*100:.4f}%  ({ana_correct}/{ana_total})")
    print(f"ACTION : token_acc = {act_tok_acc*100:.4f}%  ({act_correct}/{act_total})")
    print(f"ACTION : decision_acc = {act_dec_acc*100:.4f}%  "
          f"({act_decision_correct}/{act_decision_total})")
    print("============================================================")

    # imprime matriz de confusão
    labels = ["BUY(0)", "SELL(1)", "HOLD(2)"]
    print("[ACTION] Matriz de confusão (decisão, 0/1/2 = BUY/SELL/HOLD)")
    print("      pred:")
    print("          " + "  ".join(f"{lbl:>10}" for lbl in labels))
    for i in range(3):
        row_counts = "  ".join(f"{int(conf_mat[i,j]):10d}" for j in range(3))
        print(f"true {labels[i]:>5}:  {row_counts}")
    print("------------------------------------------------------------")

    # versão normalizada por linha (distribuição de erro por classe real)
    print("[ACTION] Matriz de confusão normalizada por linha (true)")
    for i in range(3):
        row_sum = conf_mat[i].sum().item()
        if row_sum == 0:
            probs = ["   -    " for _ in range(3)]
        else:
            probs = [f"{conf_mat[i,j].item()/row_sum:7.3f}" for j in range(3)]
        print(f"true {labels[i]:>5}:  " + "  ".join(probs))
    print("============================================================")

    results = {
        "fut_acc": fut_acc,
        "analysis_acc": ana_acc,
        "action_token_acc": act_tok_acc,
        "action_decision_acc": act_dec_acc,
        "conf_mat": conf_mat,
    }
    return results


# ============================================================
# MAIN
# ============================================================

def main():
    # --------------------------------------------------------
    # CONFIG – AJUSTE CAMINHOS CONFORME SEU PROJETO
    # --------------------------------------------------------
    AGENT_SEQ_PATH = str(PATHS.tokens_out / "agent_sequences.pt")
    CKPT_DIR       = str(PATHS.checkpoints / "checkpoints_agent_stf")
    VAL_RATIO      = 0.1
    SEED           = 42

    NUM_SAMPLES_AUTOREG = 500
    BATCH_SIZE_BASELINE = 256
    NUM_WORKERS         = 0
    # --------------------------------------------------------

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("============================================================")
    print(f"[EVAL] device = {device}")
    print("============================================================")

    # Config do modelo (mesma do treinamento tiny)
    transf_cfg = TransformerConfig()
    tok_cfg    = TokenConfig()
    win_cfg    = AgentWindowConfig()

    model = AgentTransformer(
        cfg=transf_cfg,
        vocab_size=tok_cfg.FINAL_VOCAB_SIZE,
    ).to(device)

    model = load_best_agent(model, CKPT_DIR, device)

    # Dataset de validação
    ds_val = AgentSequenceDataset(
        AGENT_SEQ_PATH,
        split="val",
        val_ratio=VAL_RATIO,
        seed=SEED,
    )
    print(f"[EVAL] sequences val: {len(ds_val)}")

    # --------------------------------------------------------
    # 1) Baseline estatístico
    # --------------------------------------------------------
    baselines = compute_baselines(
        ds_val=ds_val,
        tok_cfg=tok_cfg,
        win_cfg=win_cfg,
        device=device,
        batch_size=BATCH_SIZE_BASELINE,
        max_batches=500,
        num_workers=NUM_WORKERS,
    )

    # --------------------------------------------------------
    # 2) Avaliação autoregressiva em N amostras
    # --------------------------------------------------------
    results = evaluate_autoreg_on_val(
        ds_val=ds_val,
        model=model,
        tok_cfg=tok_cfg,
        win_cfg=win_cfg,
        device=device,
        num_samples=NUM_SAMPLES_AUTOREG,
        seed=SEED,
    )

    # --------------------------------------------------------
    # 3) Comparação rápida com baseline de ACTION (decisão)
    # --------------------------------------------------------
    act_base = baselines["action_decision"]["acc"]
    act_model = results["action_decision_acc"]
    print("============================================================")
    print("[COMPARAÇÃO] ACTION decisão vs baseline majoritária")
    print(f"Baseline (sempre maj): {act_base*100:.4f}%")
    print(f"Modelo autoregressivo: {act_model*100:.4f}%")
    print("============================================================")


if __name__ == "__main__":
    main()
