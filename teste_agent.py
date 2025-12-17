# diagnostics_context_sensitivity.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F


# =============================================================================
# 1) VOCÊ PRECISA ADAPTAR APENAS ISTO
# =============================================================================

def load_model(checkpoint_path: str, device: str = "cuda") -> torch.nn.Module:
    """
    TODO: adapte para seu loader real.
    Deve retornar o modelo em eval() e no device.

    Esperado: model(x) -> logits (B, T, V) ou (B, V) dependendo do seu forward.
    Aqui a gente assume que, dado input ids (1, L0), o modelo retorna logits (1, L0, V).
    """
    # EXEMPLO PLACEHOLDER:
    raise NotImplementedError("Implemente load_model() conforme seu projeto.")


def load_day_tokens(day: str) -> torch.Tensor:
    """
    TODO: adapte para seu formato.
    Deve retornar 1D LongTensor [L] com tokens do dia inteiro.
    Ex: day="20230606" -> tokens shape [L].

    Dica: se você tem algo tipo tokens_by_day[day] já em RAM, só retorna isso.
    """
    # EXEMPLO PLACEHOLDER:
    raise NotImplementedError("Implemente load_day_tokens(day) conforme seu projeto.")


def list_available_days() -> List[str]:
    """
    TODO: retorne lista de dias disponíveis.
    Se você já tem um arquivo/índice/dict, use isso.

    Ex: ["20180411", "20230606", "20250813", ...]
    """
    raise NotImplementedError("Implemente list_available_days() conforme seu projeto.")


# =============================================================================
# 2) CORE MÉTRICAS
# =============================================================================

@torch.no_grad()
def get_next_logits(model: torch.nn.Module, prefix_ids: torch.Tensor) -> torch.Tensor:
    """
    prefix_ids: (1, L0) long
    retorna logits do próximo token: (V,)
    Suporta forward que retorna (B, T, V) ou (B, V).
    """
    out = model(prefix_ids)
    if isinstance(out, (tuple, list)):
        out = out[0]
    if out.dim() == 3:
        # (1, L0, V) -> pega última posição
        logits = out[:, -1, :].squeeze(0)
    elif out.dim() == 2:
        # (1, V)
        logits = out.squeeze(0)
    else:
        raise RuntimeError(f"Formato inesperado de logits: {tuple(out.shape)}")
    return logits  # (V,)


def probs_from_logits(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    return F.softmax(logits / temperature, dim=-1)


def entropy_from_probs(p: torch.Tensor, eps: float = 1e-12) -> float:
    # H(p) = -sum p log p
    p = p.clamp_min(eps)
    return float(-(p * p.log()).sum().item())


def topk_mass(p: torch.Tensor, k: int) -> float:
    vals, _ = torch.topk(p, k=min(k, p.numel()))
    return float(vals.sum().item())


def top1_stats(p: torch.Tensor) -> Tuple[int, float]:
    v, idx = torch.max(p, dim=-1)
    return int(idx.item()), float(v.item())


def rank_of_id(logits: torch.Tensor, true_id: int) -> int:
    """
    rank 1 = maior logit.
    """
    # argsort desc:
    order = torch.argsort(logits, descending=True)
    # posição do true_id:
    pos = (order == int(true_id)).nonzero(as_tuple=False)
    if pos.numel() == 0:
        return int(logits.numel())
    return int(pos.item() + 1)


def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> float:
    """
    Jensen-Shannon divergence (base e).
    JS(p||q) = 0.5 KL(p||m) + 0.5 KL(q||m), m=0.5(p+q)
    """
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    m = 0.5 * (p + q)

    kl_pm = (p * (p.log() - m.log())).sum()
    kl_qm = (q * (q.log() - m.log())).sum()
    js = 0.5 * (kl_pm + kl_qm)
    return float(js.item())


# =============================================================================
# 3) TRANSFORMAÇÕES DE PREFIXO
# =============================================================================

def make_drop_last(prefix: torch.Tensor, drop_n: int = 1) -> torch.Tensor:
    if drop_n <= 0:
        return prefix
    if prefix.numel() <= drop_n:
        return prefix[:1].clone()  # evita vazio
    return prefix[:-drop_n].clone()


def make_drop_old(prefix: torch.Tensor, keep_recent_len: int) -> torch.Tensor:
    """
    mantém só o recent (últimos keep_recent_len tokens)
    """
    L0 = prefix.numel()
    keep_recent_len = max(1, min(keep_recent_len, L0))
    return prefix[-keep_recent_len:].clone()


def make_drop_recent(prefix: torch.Tensor, keep_old_len: int) -> torch.Tensor:
    """
    mantém só o old (primeiros keep_old_len tokens)
    """
    L0 = prefix.numel()
    keep_old_len = max(1, min(keep_old_len, L0))
    return prefix[:keep_old_len].clone()


def make_rand_old_keep_recent(
    prefix: torch.Tensor,
    split_idx: int,
    vocab_size: int,
    rng: torch.Generator,
) -> torch.Tensor:
    """
    prefix = [old | recent]
    randomiza old, mantém recent.
    """
    L0 = prefix.numel()
    split_idx = max(1, min(split_idx, L0 - 1))
    old = prefix[:split_idx]
    recent = prefix[split_idx:]

    rand_old = torch.randint(low=0, high=vocab_size, size=old.shape, generator=rng, device=prefix.device)
    return torch.cat([rand_old, recent], dim=0)


def pad_or_trim_to_len(x: torch.Tensor, L0: int, pad_id: int) -> torch.Tensor:
    """
    O modelo provavelmente espera sempre L0.
    Então quando drop_* encurta, a gente PAD à esquerda (old padding) com pad_id.
    E se por algum motivo ficar maior, corta pela direita.
    """
    if x.numel() == L0:
        return x
    if x.numel() > L0:
        return x[-L0:].clone()
    pad_len = L0 - x.numel()
    pad = torch.full((pad_len,), int(pad_id), dtype=torch.long, device=x.device)
    return torch.cat([pad, x], dim=0)


# =============================================================================
# 4) EXECUÇÃO DO DIAGNÓSTICO
# =============================================================================

@dataclass
class CondResult:
    top1_prob: float
    topk_mass: float
    top1_id: int
    entropy: float
    rank: Optional[int]  # rank do ground-truth (se disponível)


def eval_condition(
    model: torch.nn.Module,
    prefix_ids_1d: torch.Tensor,      # (L0,)
    true_next_id: int,
    L0: int,
    vocab_size: int,
    pad_id: int,
    temperature: float,
    topk: int,
) -> Tuple[torch.Tensor, CondResult]:
    """
    retorna probs e CondResult
    """
    x = pad_or_trim_to_len(prefix_ids_1d, L0=L0, pad_id=pad_id).unsqueeze(0)  # (1, L0)
    logits = get_next_logits(model, x)  # (V,)
    p = probs_from_logits(logits, temperature=temperature)

    t1_id, t1_prob = top1_stats(p)
    res = CondResult(
        top1_prob=t1_prob,
        topk_mass=topk_mass(p, topk),
        top1_id=t1_id,
        entropy=entropy_from_probs(p),
        rank=rank_of_id(logits, true_next_id) if true_next_id is not None else None,
    )
    return p, res


def sample_offsets(L: int, L0: int, n_offsets: int, rng_py: random.Random) -> List[int]:
    """
    offsets t escolhidos tal que prefix=t-L0:t e target=t existe:
    t in [L0, L-1]
    """
    if L <= L0 + 1:
        return []
    low = L0
    high = L - 1
    all_ok = list(range(low, high))
    if len(all_ok) <= n_offsets:
        return all_ok
    return rng_py.sample(all_ok, n_offsets)


def summarize(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": float("nan"), "p50": float("nan"), "p90": float("nan"), "min": float("nan"), "max": float("nan")}
    v = sorted(values)
    n = len(v)
    def pct(p):
        i = int(round(p * (n - 1)))
        return float(v[i])
    return {
        "mean": float(sum(v) / n),
        "p50": pct(0.50),
        "p90": pct(0.90),
        "min": float(v[0]),
        "max": float(v[-1]),
    }


def run_diagnostics(
    checkpoint_path: str,
    device: str = "cuda",
    seed: int = 123,
    n_days: int = 8,
    n_offsets_per_day: int = 64,
    L0_list: List[int] = (16, 64, 128, 256, 400),
    split_fracs: List[float] = (0.25, 0.50, 0.75),
    topk: int = 50,
    temperature: float = 1.0,
    pad_id: int = 0,
    vocab_size: Optional[int] = None,
) -> None:
    """
    Prints:
      - por dia
      - por L0
      - por split_frac
      - estatísticas de JS e rank
    """
    rng_py = random.Random(seed)
    rng_torch = torch.Generator(device=device)
    rng_torch.manual_seed(seed)

    model = load_model(checkpoint_path, device=device).eval()

    days_all = list_available_days()
    if not days_all:
        raise RuntimeError("Nenhum dia disponível em list_available_days().")

    if n_days >= len(days_all):
        days = days_all
    else:
        days = rng_py.sample(days_all, n_days)

    print(f"[MODEL] {checkpoint_path}")
    print(f"[DAYS] sampled {len(days)} days (seed={seed})")

    for day in days:
        tokens = load_day_tokens(day).to(device=device)
        if tokens.dtype != torch.long:
            tokens = tokens.long()
        L = int(tokens.numel())

        # se vocab_size não foi informado, tenta inferir (não perfeito, mas ajuda)
        vsize = vocab_size
        if vsize is None:
            vsize = int(tokens.max().item()) + 1

        print("\n" + "="*90)
        print(f"[DAY] {day} | L={L} | vocab_size~{vsize}")
        print("="*90)

        for L0 in L0_list:
            if L <= L0 + 2:
                print(f"\n[PREFIX] L0={L0} (SKIP: dia curto demais)")
                continue

            offsets = sample_offsets(L, L0=L0, n_offsets=n_offsets_per_day, rng_py=rng_py)
            if not offsets:
                print(f"\n[PREFIX] L0={L0} (SKIP: sem offsets válidos)")
                continue

            print("\n" + "#"*80)
            print(f"[PREFIX] L0={L0} | offsets={len(offsets)}")
            print("#"*80)

            for split_frac in split_fracs:
                split_idx = int(round(L0 * split_frac))
                split_idx = max(1, min(split_idx, L0 - 1))
                keep_recent_len = L0 - split_idx
                keep_old_len = split_idx

                # coletores
                js_drop_last = []
                js_drop_old = []
                js_drop_recent = []
                js_rand_old = []

                ranks_full = []
                ranks_drop_last = []
                ranks_drop_old = []
                ranks_drop_recent = []
                ranks_rand_old = []

                top1_full = []
                top1_drop_last = []
                top1_drop_old = []
                top1_drop_recent = []
                top1_rand_old = []

                ent_full = []
                ent_drop_last = []
                ent_drop_old = []
                ent_drop_recent = []
                ent_rand_old = []

                for t in offsets:
                    prefix = tokens[t - L0 : t]          # (L0,)
                    true_next = int(tokens[t].item())

                    # full
                    p_full, r_full = eval_condition(
                        model, prefix, true_next, L0, vsize, pad_id, temperature, topk
                    )

                    # drop_last: remove 1 token do final e pad left
                    prefix_drop_last = make_drop_last(prefix, drop_n=1)
                    p_dl, r_dl = eval_condition(
                        model, prefix_drop_last, true_next, L0, vsize, pad_id, temperature, topk
                    )

                    # drop_old: mantém só recent
                    prefix_drop_old = make_drop_old(prefix, keep_recent_len=keep_recent_len)
                    p_do, r_do = eval_condition(
                        model, prefix_drop_old, true_next, L0, vsize, pad_id, temperature, topk
                    )

                    # drop_recent: mantém só old
                    prefix_drop_recent = make_drop_recent(prefix, keep_old_len=keep_old_len)
                    p_dr, r_dr = eval_condition(
                        model, prefix_drop_recent, true_next, L0, vsize, pad_id, temperature, topk
                    )

                    # rand_old_keep_recent
                    prefix_rand_old = make_rand_old_keep_recent(prefix, split_idx=split_idx, vocab_size=vsize, rng=rng_torch)
                    p_ro, r_ro = eval_condition(
                        model, prefix_rand_old, true_next, L0, vsize, pad_id, temperature, topk
                    )

                    # JS divergences vs full
                    js_drop_last.append(js_divergence(p_full, p_dl))
                    js_drop_old.append(js_divergence(p_full, p_do))
                    js_drop_recent.append(js_divergence(p_full, p_dr))
                    js_rand_old.append(js_divergence(p_full, p_ro))

                    # ranks
                    ranks_full.append(float(r_full.rank))
                    ranks_drop_last.append(float(r_dl.rank))
                    ranks_drop_old.append(float(r_do.rank))
                    ranks_drop_recent.append(float(r_dr.rank))
                    ranks_rand_old.append(float(r_ro.rank))

                    # top1 probs
                    top1_full.append(r_full.top1_prob)
                    top1_drop_last.append(r_dl.top1_prob)
                    top1_drop_old.append(r_do.top1_prob)
                    top1_drop_recent.append(r_dr.top1_prob)
                    top1_rand_old.append(r_ro.top1_prob)

                    # entropy
                    ent_full.append(r_full.entropy)
                    ent_drop_last.append(r_dl.entropy)
                    ent_drop_old.append(r_do.entropy)
                    ent_drop_recent.append(r_dr.entropy)
                    ent_rand_old.append(r_ro.entropy)

                print("\n" + "-"*70)
                print(f"[SPLIT] old={int(split_frac*100)}% (split_idx={split_idx}) | recent={L0-split_idx}")

                # JS summary
                s_js_dl = summarize(js_drop_last)
                s_js_do = summarize(js_drop_old)
                s_js_dr = summarize(js_drop_recent)
                s_js_ro = summarize(js_rand_old)

                print("JS(full vs drop_last)          :", s_js_dl)
                print("JS(full vs drop_old)           :", s_js_do)
                print("JS(full vs drop_recent)        :", s_js_dr)
                print("JS(full vs rand_old_keep_recent):", s_js_ro)

                # Rank summary
                s_rf = summarize(ranks_full)
                s_rdl = summarize(ranks_drop_last)
                s_rdo = summarize(ranks_drop_old)
                s_rdr = summarize(ranks_drop_recent)
                s_rro = summarize(ranks_rand_old)

                print("\nRank of Ground Truth (1=best):")
                print("  full          :", s_rf)
                print("  drop_last     :", s_rdl)
                print("  drop_old      :", s_rdo)
                print("  drop_recent   :", s_rdr)
                print("  rand_old_keep :", s_rro)

                # Top1 prob summary
                print("\nTop1 prob:")
                print("  full          :", summarize(top1_full))
                print("  drop_last     :", summarize(top1_drop_last))
                print("  drop_old      :", summarize(top1_drop_old))
                print("  drop_recent   :", summarize(top1_drop_recent))
                print("  rand_old_keep :", summarize(top1_rand_old))

                # Entropy summary
                print("\nEntropy:")
                print("  full          :", summarize(ent_full))
                print("  drop_last     :", summarize(ent_drop_last))
                print("  drop_old      :", summarize(ent_drop_old))
                print("  drop_recent   :", summarize(ent_drop_recent))
                print("  rand_old_keep :", summarize(ent_rand_old))

        print("\n[DONE DAY]")


# =============================================================================
# 5) MAIN (ajuste parâmetros aqui)
# =============================================================================

if __name__ == "__main__":
    # Ajuste aqui
    CKPT = r"checkpoints_agent_tiny\agent_best.pt"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    run_diagnostics(
        checkpoint_path=CKPT,
        device=DEVICE,
        seed=123,
        n_days=10,                 # quantos dias aleatórios
        n_offsets_per_day=64,       # quantos pontos (horas/trechos) por dia
        L0_list=[16, 64, 128, 256, 400],
        split_fracs=[0.25, 0.50, 0.75],
        topk=50,
        temperature=1.0,
        pad_id=0,                   # IMPORTANTE: coloque o token de PAD/BOS correto do seu treino
        vocab_size=None,            # se souber, coloque fixo (ex: 4096). None tenta inferir do dia.
    )
