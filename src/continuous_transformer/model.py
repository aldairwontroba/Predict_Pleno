"""
ContinuousEventTransformer (CET)

Diferença em relação ao AgentTransformer (src/agent/model.py):
  - Não usa nn.Embedding para tokens discretos.
  - Cada evento é um vetor float32 de INPUT_DIM features.
  - input_proj: Linear(input_dim, d_model) projeta cada evento.
  - next_event_head: prediz o próximo evento (pré-treino, Huber loss).
  - action_head: classifica ação no último token (SFT, CrossEntropy).

O trunk (blocos Transformer causais) é idêntico ao modelo existente.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.continuous_transformer.config import CTConfig


# ============================================================
# Blocos internos (mesmo padrão de src/agent/model.py)
# ============================================================

class _MHSA(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.qkv  = nn.Linear(d_model, d_model * 3)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        att = att.masked_fill(mask == 0, float("-inf"))
        att = self.drop(torch.softmax(att, dim=-1))
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class _Block(nn.Module):
    def __init__(self, cfg: CTConfig):
        super().__init__()
        self.ln1  = nn.LayerNorm(cfg.d_model)
        self.attn = _MHSA(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.ln2  = nn.LayerNorm(cfg.d_model)
        self.mlp  = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return x


# ============================================================
# Modelo principal
# ============================================================

class ContinuousEventTransformer(nn.Module):
    """
    Transformer causal para séries de eventos de mercado contínuos.

    Forward recebe:
      events : [B, T, input_dim]   float32, eventos normalizados
      meta   : [B, num_meta]       float32, opcional (dia da semana, horário, etc.)

    Retorna um dict com:
      'next_event' : [B, T, input_dim]   logits de pré-treino (predição do próximo evento)
      'action'     : [B, num_actions]    logits de ação no ÚLTIMO token (SFT)
      'hidden'     : [B, T, d_model]     representações internas (útil para análise)
    """

    def __init__(self, cfg: CTConfig | None = None):
        super().__init__()
        if cfg is None:
            cfg = CTConfig()
        self.cfg = cfg

        # Projeção de entrada: evento float → espaço d_model
        self.input_proj = nn.Linear(cfg.input_dim, cfg.d_model)
        self.input_ln   = nn.LayerNorm(cfg.d_model)

        # Embedding posicional learnable
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.d_model)

        # Projeção de metadados (opcional): dia, hora, etc.
        if cfg.num_meta > 0:
            self.meta_proj = nn.Linear(cfg.num_meta, cfg.d_model)
        else:
            self.meta_proj = None

        # Trunk: blocos causais
        self.blocks   = nn.ModuleList([_Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_final = nn.LayerNorm(cfg.d_model)

        # Cabeça de pré-treino: prevê o próximo evento (regressão)
        self.next_event_head = nn.Linear(cfg.d_model, cfg.input_dim)

        # Cabeça de ação: buy / sell / hold (SFT)
        self.action_head = nn.Linear(cfg.d_model, cfg.num_actions)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        events: torch.Tensor,
        meta: torch.Tensor | None = None,
    ) -> dict:
        """
        events : [B, T, input_dim]
        meta   : [B, num_meta] opcional
        """
        B, T, _ = events.shape
        assert T <= self.cfg.seq_len, (
            f"Sequência de {T} eventos excede seq_len={self.cfg.seq_len}"
        )

        # Projeção de entrada
        h = self.input_ln(self.input_proj(events))  # [B, T, d_model]

        # Embedding posicional
        pos = torch.arange(T, device=events.device).unsqueeze(0)  # [1, T]
        h = h + self.pos_emb(pos)

        # Condicionamento de metadados (soma a todos os tokens)
        if meta is not None and self.meta_proj is not None:
            meta_vec = self.meta_proj(meta.float())   # [B, d_model]
            h = h + meta_vec.unsqueeze(1)             # broadcast [B, 1, d_model]

        # Máscara causal [1, 1, T, T]
        mask = torch.tril(torch.ones(T, T, device=events.device)).unsqueeze(0).unsqueeze(0)

        # Blocos Transformer
        for blk in self.blocks:
            h = blk(h, mask)
        h = self.ln_final(h)   # [B, T, d_model]

        next_event_logits = self.next_event_head(h)       # [B, T, input_dim]
        action_logits     = self.action_head(h[:, -1, :]) # [B, num_actions]

        return {
            "next_event": next_event_logits,
            "action":     action_logits,
            "hidden":     h,
        }

    def encode(self, events: torch.Tensor, meta: torch.Tensor | None = None) -> torch.Tensor:
        """Retorna apenas as representações internas [B, T, d_model]. Útil para análise."""
        return self.forward(events, meta)["hidden"]

    def predict_action(self, events: torch.Tensor, meta: torch.Tensor | None = None) -> torch.Tensor:
        """
        Retorna probabilidades de ação [B, num_actions] para inferência.
        Usa softmax sobre os logits do último token.
        """
        with torch.no_grad():
            out = self.forward(events, meta)
        return torch.softmax(out["action"], dim=-1)

    @classmethod
    def from_checkpoint(cls, ckpt_path: str, map_location: str = "cpu") -> "ContinuousEventTransformer":
        """Carrega modelo de um checkpoint salvo pelo train_pretrain ou train_sft."""
        ckpt = torch.load(ckpt_path, map_location=map_location)
        cfg_dict = ckpt.get("cfg", {})
        cfg = CTConfig()
        for k, v in cfg_dict.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        model = cls(cfg)
        state = ckpt.get("model_state") or ckpt
        model.load_state_dict(state, strict=True)
        return model
