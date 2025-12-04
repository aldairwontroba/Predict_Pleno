# rt_agent_runner.py
from collections import deque
from pathlib import Path
from typing import Deque, Tuple

import datetime as dt
import numpy as np
import torch

from CreateVector.main import realtime_process
from vq_tokenizer_rt import RealTimeVQTokenizer
from agent_transformer import AgentTransformer
from build_agent_sequences import (
    AgentWindowConfig,
    TokenConfig,
    TransformerConfig,
    _is_last_business_day_of_month,
)
from validate_sft import generate_autoregressive_from_prefix  # reaproveitado

# ------------------------------------------------------------------
# Contexto em tempo real (usa dia/hora atual e mantém posição = FLAT)
# ------------------------------------------------------------------


def compute_rt_context_tokens(
    now: dt.datetime,
    tok_cfg: TokenConfig,
    n_ctx_tokens: int,
) -> torch.Tensor:
    """
    Versão "real-time" do compute_context_and_state_tokens:
      - não usa raw_events
      - usa hora/data correntes
      - posição sempre FLAT (0)
    """

    base = tok_cfg.RANGE_CONTEXT[0]
    ctx_neutral = base

    tokens = [ctx_neutral] * n_ctx_tokens

    # 1) dia da semana + último dia útil
    d = now.date()
    dow = d.weekday()  # 0=Seg ... 6=Dom
    dow_token = base + dow

    is_last_bd = 1 if _is_last_business_day_of_month(d) else 0
    last_bd_token = base + 20 + is_last_bd

    # 2) bucket de horário (mesma lógica do offline)
    hour_int = now.hour

    if hour_int < 10:
        tod_bucket = 0
    elif hour_int < 11:
        tod_bucket = 1
    elif hour_int < 12:
        tod_bucket = 2
    elif hour_int < 16:
        tod_bucket = 3
    else:
        tod_bucket = 4

    tod_token = base + 10 + tod_bucket

    # 3) estado de posição (sempre FLAT por enquanto)
    pos_state_id = 0  # 0=FLAT, 1=LONG, 2=SHORT (futuro)
    pos_state_token = base + 30 + pos_state_id

    # 4) preencher primeiros slots
    if n_ctx_tokens > 0:
        tokens[0] = dow_token
    if n_ctx_tokens > 1:
        tokens[1] = tod_token
    if n_ctx_tokens > 2:
        tokens[2] = last_bd_token
    if n_ctx_tokens > 3:
        tokens[3] = pos_state_token

    return torch.tensor(tokens, dtype=torch.long)


# ------------------------------------------------------------------
# Runner em tempo real
# ------------------------------------------------------------------


class RealTimeAgentRunner:
    def __init__(
        self,
        pair: Tuple[str, str],
        vq_ckpt: str | Path,
        agent_ckpt: str | Path,
        device: str = "cuda",
    ):
        self.pair = pair
        self.device = torch.device(device)

        # --- Configs reaproveitadas do treino ---
        self.transf_cfg = TransformerConfig()
        self.tok_cfg = TokenConfig()
        self.win_cfg = AgentWindowConfig()

        # guarda layout pronto
        self.layout = self.win_cfg.layout()

        # --- VQ-VAE para tokenizar eventos ---
        self.tokenizer = RealTimeVQTokenizer(
            vq_ckpt=vq_ckpt,
            scaler_path="scaler_norm2.pkl",
            device=device,
        )

        # --- Modelo de agente (world model + SFT) ---
        self.model = AgentTransformer(
            cfg=self.transf_cfg,
            vocab_size=self.tok_cfg.FINAL_VOCAB_SIZE,
        ).to(self.device)

        state = torch.load(agent_ckpt, map_location="cpu")
        self.model.load_state_dict(state["model_state"])
        self.model.to(self.device)
        self.model.eval()

        # buffer dos últimos eventos tokenizados (usa n_past_tokens oficial)
        self.past_tokens: Deque[int] = deque(
            maxlen=self.win_cfg.n_past_tokens
        )

    # ---------------------------------------------------------
    # Callback chamado pelo realtime_process
    # ---------------------------------------------------------
    def on_event(self, ev: dict):
        """
        Recebe 1 evento bruto do Segmenter.
        Converte em vetor, normaliza, tokeniza e roda o modelo se já tiver contexto suficiente.
        """
        # ev["vector"] é um dict de features
        vec_dict = ev.get("vector", {})
        tok = self.tokenizer.event_to_token(vec_dict)

        self.past_tokens.append(tok)

        if len(self.past_tokens) < self.win_cfg.n_past_tokens:
            # ainda montando histórico suficiente
            return

        # quando já temos PAST completo, rodamos uma predição:
        self.run_model_once()

    @torch.no_grad()
    def run_model_once(self):
        """
        Monta o prefixo [CTX + PAST + SEP_FUT] e gera:
        - FUT (64 tokens)
        - ANALYSIS (4 tokens)
        - ACTION (2 tokens)
        """

        # contexto em tempo real
        now = dt.datetime.now()
        ctx = compute_rt_context_tokens(
            now=now,
            tok_cfg=self.tok_cfg,
            n_ctx_tokens=self.win_cfg.n_ctx_tokens,
        ).to(self.device)

        # passado de eventos (já no deque)
        past = torch.tensor(
            list(self.past_tokens),
            dtype=torch.long,
            device=self.device,
        )

        # separador de futuro vindo do TokenConfig
        sep_fut = torch.tensor(
            [self.tok_cfg.SEP_FUT_TOKEN],
            dtype=torch.long,
            device=self.device,
        )

        # prefixo completo (1, prefix_len)
        x_prefix = torch.cat([ctx, past, sep_fut], dim=0).unsqueeze(0)

        # queremos gerar até o fim da seção ACTION
        _, action_end = self.layout["action"]
        T_target = action_end  # posição final (exclusive) da janela

        # gera autoregressivo usando a mesma função que você já validou
        x_gen = generate_autoregressive_from_prefix(
            model=self.model,
            x_prefix=x_prefix,
            device=self.device,
            T=T_target,
        )  # (1, T_target)

        x_gen = x_gen[0].tolist()

        # indexes oficiais via layout()
        fut_start, fut_end = self.layout["fut"]
        ana_start, ana_end = self.layout["analysis"]
        act_start, act_end = self.layout["action"]

        fut_tokens = x_gen[fut_start:fut_end]
        ana_tokens = x_gen[ana_start:ana_end]
        act_tokens = x_gen[act_start:act_end]

        print("================================================")
        print(f"[RT] now={now}")
        print("[RT] FUT tokens (primeiros 10):", fut_tokens[:10], "...")
        print("[RT] ANALYSIS tokens:", ana_tokens)
        print("[RT] ACTION tokens:", act_tokens)
        # aqui depois você pluga o decodificador BUY/SELL/HOLD
        print("================================================")


def main_rt():
    pair = ("wdo", "dol")
    runner = RealTimeAgentRunner(
        pair=pair,
        vq_ckpt="vq_out/vqvae_best_1311.pt",          # ajuste se mudar o caminho
        agent_ckpt="checkpoints_agent_stf/stf_best.pt",
        device="cuda",
    )

    # IMPORTANTE: seu realtime_process precisa aceitar o callback on_event
    # algo como: realtime_process(pair, on_event=runner.on_event)
    realtime_process(pair, on_event=runner.on_event)


if __name__ == "__main__":
    main_rt()
