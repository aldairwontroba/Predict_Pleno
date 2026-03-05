# build_agent_sequences.py

from __future__ import annotations
import os
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import datetime as _dt
from src.config import PATHS

def _parse_date_from_day_key(day_key: str) -> _dt.date:
    """
    day_key no formato 'YYYYMMDD', ex: '20180411' -> date(2018, 4, 11)
    """
    return _dt.datetime.strptime(day_key, "%Y%m%d").date()


def _is_last_business_day_of_month(d: _dt.date) -> bool:
    """
    Considera dia útil como Monday..Friday (0..4).
    Retorna True se 'd' é o último dia útil do mês.
    (Ignora feriado, depois se quiser refinamos com calendário B3).
    """
    year, month = d.year, d.month

    if month == 12:
        next_month = _dt.date(year + 1, 1, 1)
    else:
        next_month = _dt.date(year, month + 1, 1)

    last_dom = next_month - _dt.timedelta(days=1)

    # volta até encontrar o último weekday (0..4)
    cur = last_dom
    while cur.weekday() >= 5:  # 5=Saturday, 6=Sunday
        cur -= _dt.timedelta(days=1)

    return d == cur


# ============================================================
#  CONFIGS BÁSICAS (reutiliza o que você já tem)
# ============================================================

class TransformerConfig:
    # MODELO MENOR (já treinado)
    dim = 128
    n_heads = 4
    n_layers = 4
    seq_len = 512
    dropout = 0.1


class TokenConfig:
    # tokens do VQ-VAE
    VQ_TOKENS = 4096

    # reservas futuras
    NUM_ACTION_TOKENS = 16
    NUM_CONTEXT_TOKENS = 64
    NUM_AUX_TOKENS = 32

    # vocabulário principal
    VOCAB_SIZE = (
        VQ_TOKENS +
        NUM_ACTION_TOKENS +
        NUM_CONTEXT_TOKENS +
        NUM_AUX_TOKENS
    )

    RANGE_VQ = (0, VQ_TOKENS - 1)
    RANGE_ACTION = (VQ_TOKENS, VQ_TOKENS + NUM_ACTION_TOKENS - 1)
    RANGE_CONTEXT = (RANGE_ACTION[1] + 1, RANGE_ACTION[1] + NUM_CONTEXT_TOKENS)
    RANGE_AUX = (RANGE_CONTEXT[1] + 1, VOCAB_SIZE - 1)

    # token extra só para padding
    PAD_TOKEN = VOCAB_SIZE
    FINAL_VOCAB_SIZE = VOCAB_SIZE + 1

    # alguns IDs reservados (você pode mudar depois)
    # marcadores de seção (pegando de RANGE_AUX)
    SEP_FUT_TOKEN = RANGE_AUX[0]       # início da seção de futuro
    SEP_ANALYSIS_TOKEN = RANGE_AUX[0] + 1
    SEP_ACTION_TOKEN = RANGE_AUX[0] + 2

    # base para tokens de análise e ação (faixas internas)
    ANALYSIS_BASE = RANGE_AUX[0] + 10  # onde começam os "analysis_*"
    ACTION_BASE = RANGE_ACTION[0]      # primeira action


# ============================================================
#  CONFIG DA JANELA DO AGENTE
# ============================================================

@dataclass
class AgentWindowConfig:
    """
    Define como a janela de 512 é quebrada em pedaços:
    [CTX/STATE][PAST][SEP_FUT][FUT][SEP_ANALYSIS][ANALYSIS][SEP_ACTION][ACTION][PAD...]
    """
    seq_len: int = TransformerConfig.seq_len

    n_ctx_tokens: int = 16       # contexto + estado
    n_past_tokens: int = 320     # tokens de mercado passado (VQ)
    n_future_tokens: int = 64    # tokens de mercado futuro (VQ)

    n_analysis_tokens: int = 4   # ex: trend, vol, volume, agressão
    n_action_tokens: int = 2     # ex: ação principal + força/tamanho

    stride: int = 16             # passo entre janelas dentro de um dia
    max_episodes_per_day: Optional[int] = None  # se quiser limitar

    def layout(self):
        """
        Retorna os índices de cada seção na janela.
        """
        pos = 0
        ctx_start = pos
        ctx_end = ctx_start + self.n_ctx_tokens
        pos = ctx_end

        past_start = pos
        past_end = past_start + self.n_past_tokens
        pos = past_end

        sep_fut_pos = pos
        pos += 1

        fut_start = pos
        fut_end = fut_start + self.n_future_tokens
        pos = fut_end

        sep_analysis_pos = pos
        pos += 1

        analysis_start = pos
        analysis_end = analysis_start + self.n_analysis_tokens
        pos = analysis_end

        sep_action_pos = pos
        pos += 1

        action_start = pos
        action_end = action_start + self.n_action_tokens
        pos = action_end

        if pos > self.seq_len:
            raise ValueError(
                f"Layout estourou seq_len={self.seq_len}: precisa de {pos} tokens."
            )

        return {
            "ctx": (ctx_start, ctx_end),
            "past": (past_start, past_end),
            "sep_fut": sep_fut_pos,
            "fut": (fut_start, fut_end),
            "sep_analysis": sep_analysis_pos,
            "analysis": (analysis_start, analysis_end),
            "sep_action": sep_action_pos,
            "action": (action_start, action_end),
        }


# ============================================================
#  CARREGAR tokens_by_day.pt
# ============================================================

def load_tokens_by_day(path: str | Path) -> Dict[str, object]:
    """
    Carrega o arquivo tokens_by_day.pt gerado pelo tokenize_all_events.
    Deve conter:
      - day_keys: List[str]
      - tokens_by_day: List[torch.Tensor]
      - meta: dict
    """
    path = Path(path)
    data = torch.load(path, map_location="cpu")
    expected_keys = {"day_keys", "tokens_by_day", "meta"}
    missing = expected_keys - set(data.keys())
    if missing:
        raise KeyError(f"Arquivo {path} não contém chaves: {missing}")
    return data


# ============================================================
#  MAPEAR day_key -> arquivo de eventos crus
# ============================================================

def default_extract_day_key_from_filename(fname: str) -> str:
    """
    *** IMPORTANTE ***
    Essa função PRECISA ser consistente com a extract_day_key() que você usou
    no tokenize_all_events.py.

    Aqui eu coloco uma default genérica (pega a parte antes do primeiro '_'):
       ex: '2019-01-05_WDO.npy' -> '2019-01-05'

    Ajuste isso se o seu day_key for diferente.
    """
    base = os.path.basename(fname)
    name, _ext = os.path.splitext(base)
    # exemplo: '2019-01-05_WDO' -> '2019-01-05'
    return name.split("_")[0]


def build_day_to_events_path_map(
    events_dir: str | Path,
    day_keys: List[str],
    extract_fn=default_extract_day_key_from_filename,
) -> Dict[str, Path]:
    """
    Vasculha o diretório com eventos crus e cria um mapeamento day_key -> Path.

    - events_dir: 'C:/Users/Aldair/Desktop/eventos_processados'
    - day_keys: vindo do tokens_by_day.pt
    - extract_fn: função para extrair day_key do nome do arquivo .npy

    Se tiver mais de um arquivo com o mesmo day_key, pega o primeiro (você pode
    ajustar depois).
    """
    events_dir = Path(events_dir)
    all_files = list(events_dir.glob("*_wdo_dol.npy"))

    key_to_path: Dict[str, Path] = {}

    for f in all_files:
        dk = extract_fn(f.name)
        if dk not in key_to_path:
            key_to_path[dk] = f

    missing = [dk for dk in day_keys if dk not in key_to_path]
    if missing:
        print("[WARN] Alguns day_keys não possuem arquivo de eventos cru:")
        for dk in missing[:20]:
            print(f"   - {dk}")
        if len(missing) > 20:
            print(f"   ... (+{len(missing)-20} outros)")

    return key_to_path


# ============================================================
#  HELPERS PARA DISCRETIZAR FEATURES FUTURAS E GERAR TOKENS
# ============================================================

def _bucketize(value: float, thresholds: List[float]) -> int:
    """
    thresholds: lista ordenada [t1, t2, t3, ...]
    retorna um índice de faixa 0 .. len(thresholds)
    """
    for i, th in enumerate(thresholds):
        if value < th:
            return i
    return len(thresholds)

def compute_context_and_state_tokens(
    day_key: str,
    raw_events: np.ndarray,
    center_idx: int,
    tok_cfg: TokenConfig,
    n_ctx_tokens: int,
) -> torch.Tensor:
    """
    Gera tokens de contexto + estado para o prefixo da janela.

    Definição atual:
      ctx[0] = dia da semana       (0=Seg, ..., 6=Dom)
      ctx[1] = faixa de horário    (5 buckets: 9,10,11,12-16,>16)
      ctx[2] = último dia útil?    (0 = não, 1 = sim)
      ctx[3] = estado posição      (0 = flat, no futuro: 1 long, 2 short)
      ctx[4..] = contexto neutro (reservado)

    Todos codificados na faixa RANGE_CONTEXT.
    """

    base = tok_cfg.RANGE_CONTEXT[0]
    ctx_neutral = base  # token neutro de contexto

    # inicializa tudo neutro
    tokens = [ctx_neutral] * n_ctx_tokens

    # --------------------------------------------------------
    # 1) Data, dia da semana e "último dia útil"
    # --------------------------------------------------------
    d = _parse_date_from_day_key(day_key)
    dow = d.weekday()  # 0=Seg, 1=Ter, ..., 4=Sex, 5=Sab, 6=Dom

    # token dia da semana: base + dow (0..6)
    dow_token = base + dow  # garante ficar dentro do bloco de contexto

    is_last_bd = 1 if _is_last_business_day_of_month(d) else 0
    # token last business day: base + 20 + (0 ou 1)
    last_bd_token = base + 20 + is_last_bd

    # --------------------------------------------------------
    # 2) Reconstruir tempo absoluto dentro do dia
    # --------------------------------------------------------
    # garante índice válido no raw_events
    center_idx = int(max(0, min(center_idx, raw_events.shape[0] - 1)))

    # soma dos deltas de tempo até o evento central
    # raw_events[:, 0] = delta_t em segundos
    dt_cumsum = float(raw_events[: center_idx + 1, 0].sum())

    # assumindo pregão começa às 09:00
    start_sec = 9 * 3600  # 09:00:00 em segundos
    t_sec = start_sec + dt_cumsum

    # volta pro range [0, 24h) só pra ter um sanity check
    # (não é super necessário, mas evita drift numérico absurdo)
    t_sec = max(0.0, min(t_sec, 24 * 3600 - 1))

    hour_float = t_sec / 3600.0
    hour_int = int(hour_float)

    # buckets que você pediu:
    #   0: [9,10)
    #   1: [10,11)
    #   2: [11,12)
    #   3: [12,16)
    #   4: [16, +inf)
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

    # token time-of-day: base + 10 + bucket (0..4)
    tod_token = base + 10 + tod_bucket

    # --------------------------------------------------------
    # 3) Estado da posição (por enquanto sempre FLAT)
    # --------------------------------------------------------
    # design:
    #   base + 30 + 0 = FLAT
    #   base + 30 + 1 = LONG  (futuro)
    #   base + 30 + 2 = SHORT (futuro)
    pos_state_id = 0  # sempre flat nessa fase
    pos_state_token = base + 30 + pos_state_id

    # --------------------------------------------------------
    # 4) Preenche os primeiros slots do contexto
    # --------------------------------------------------------
    if n_ctx_tokens > 0:
        tokens[0] = dow_token
    if n_ctx_tokens > 1:
        tokens[1] = tod_token
    if n_ctx_tokens > 2:
        tokens[2] = last_bd_token
    if n_ctx_tokens > 3:
        tokens[3] = pos_state_token
    # tokens[4:] já estão neutros

    return torch.tensor(tokens, dtype=torch.long)




def compute_future_features(
    raw_events: np.ndarray,
    start_idx: int,
    end_idx: int,
) -> Dict[str, float]:
    """
    Extrai features do FUTURO real entre [start_idx, end_idx).

    Convenções:
      - raw_events[:, 0] = delta_t (segundos)  [não usado aqui]
      - raw_events[:, 1] = delta_preço
      - VOL_COL          = coluna de volume (ajuste conforme seu layout)

    Retorna:
      - max_up         : máximo deslocamento positivo (pontos)
      - max_down       : máximo deslocamento negativo (pontos, negativo)
      - vol_range      : max_up - max_down
      - zero_crossings : nº de cruzamentos pela abertura (preço 0)
      - total_volume   : soma dos volumes na janela
      - mean_volume    : média de volume por evento
      - direction_label: +1 (compra), -1 (venda), 0 (flat)
    """

    n = raw_events.shape[0]
    if n == 0:
        return {
            "max_up": 0.0,
            "max_down": 0.0,
            "vol_range": 0.0,
            "zero_crossings": 0,
            "total_volume": 0.0,
            "mean_volume": 0.0,
            "direction_label": 0,
        }

    # Sanitiza índices
    start_idx = int(max(0, min(start_idx, n - 1)))
    end_idx = int(max(start_idx + 1, min(end_idx, n)))

    if end_idx <= start_idx + 1:
        return {
            "max_up": 0.0,
            "max_down": 0.0,
            "vol_range": 0.0,
            "zero_crossings": 0,
            "total_volume": 0.0,
            "mean_volume": 0.0,
            "direction_label": 0,
        }

    # --------------------------------------------------------
    # 1) Preço relativo: acumula delta_preço a partir do evento atual
    # --------------------------------------------------------
    deltas = raw_events[start_idx:end_idx, 1].astype(float)
    deltas = deltas[~np.isnan(deltas)]
    if deltas.size == 0:
        return {
            "max_up": 0.0,
            "max_down": 0.0,
            "vol_range": 0.0,
            "zero_crossings": 0,
            "total_volume": 0.0,
            "mean_volume": 0.0,
            "direction_label": 0,
        }

    # price_rel[0] = 0 (preço no início da janela = 0)
    # price_rel[k] = soma dos deltas até k-1
    price_rel = np.concatenate(([0.0], np.cumsum(deltas)))

    max_up = float(np.max(price_rel))
    max_down = float(np.min(price_rel))  # <= 0
    vol_range = float(max_up - max_down)

    # --------------------------------------------------------
    # 2) Cruzamentos pela abertura (0): quantas vezes troca de sinal
    # --------------------------------------------------------
    # Para evitar sign 0 bagunçando, propagamos último sinal não-nulo
    sign = np.sign(price_rel)
    # resolve zeros: copia o último sinal não-zero (bem simples)
    last = 0.0
    for i in range(sign.size):
        if sign[i] == 0 and last != 0:
            sign[i] = last
        elif sign[i] != 0:
            last = sign[i]

    zero_crossings = int(np.sum(sign[:-1] * sign[1:] < 0))

    # --------------------------------------------------------
    # 3) Direção forte (regra dos 5 pontos "sem voltar")
    # --------------------------------------------------------
    THRESH_STRONG = 5.0  # pelo menos 5 pontos na mesma direção
    dir_strong = 0  # +1 compra, -1 venda, 0 neutro

    for p in price_rel:
        if p >= THRESH_STRONG:
            dir_strong = +1
            break
        if p <= -THRESH_STRONG:
            dir_strong = -1
            break

    # --------------------------------------------------------
    # 4) Dominância de extremo (max/min > 2x outro, menor <= 10)
    # --------------------------------------------------------
    THRESH_DOM_RATIO = 2.0
    THRESH_OTHER_MAX = 10.0

    abs_up = max_up
    abs_down = abs(max_down)

    dominant_dir = 0
    # candidato compra: movimento pra cima bem maior que pra baixo
    if abs_up >= THRESH_STRONG and abs_up >= THRESH_DOM_RATIO * abs_down and abs_down <= THRESH_OTHER_MAX:
        dominant_dir = +1
    # candidato venda: movimento pra baixo bem maior que pra cima
    elif abs_down >= THRESH_STRONG and abs_down >= THRESH_DOM_RATIO * abs_up and abs_up <= THRESH_OTHER_MAX:
        dominant_dir = -1

    # --------------------------------------------------------
    # 5) Combina regras + filtro de ruído (zero crossings)
    # --------------------------------------------------------
    MAX_CROSSINGS = 4

    direction_label = 0  # default flat

    if zero_crossings <= MAX_CROSSINGS:
        if dir_strong != 0:
            # regra 1 tem prioridade: 5 pontos "limpos" primeiro
            direction_label = dir_strong
        elif dominant_dir != 0:
            # senão, usa dominância de extremo
            direction_label = dominant_dir
        else:
            direction_label = 0  # flat
    else:
        # muita ida-e-volta → flat, mesmo que tenha amplitude
        direction_label = 0

    # --------------------------------------------------------
    # 6) Volume: total e médio
    # --------------------------------------------------------
    total_volume = 0.0
    mean_volume = 0.0
    VOL_COL = 19
    if raw_events.shape[1] > VOL_COL:
        vol_slice = raw_events[start_idx:end_idx, VOL_COL].astype(float)
        vol_slice = vol_slice[~np.isnan(vol_slice)]
        if vol_slice.size > 0:
            total_volume = float(vol_slice.sum())
            mean_volume = float(vol_slice.mean())

    return {
        "max_up": float(max_up),
        "max_down": float(max_down),
        "vol_range": float(vol_range),
        "zero_crossings": int(zero_crossings),
        "total_volume": float(total_volume),
        "mean_volume": float(mean_volume),
        "direction_label": int(direction_label),
    }

def compute_analysis_tokens(
    fut_features: Dict[str, float],
    tok_cfg: TokenConfig,
    n_analysis_tokens: int,
) -> torch.Tensor:
    """
    Gera tokens de análise (tendência, vol, volume, choppiness) a partir
    das features do futuro.

    fut_features esperadas:
      - max_up         (float)
      - max_down       (float, negativo ou zero)
      - vol_range      (float = max_up - max_down)
      - total_volume   (float)
      - mean_volume    (float)   [ainda não usado, mas disponível]
      - zero_crossings (int)
      - direction_label (int: -1, 0, +1)
    """

    base = tok_cfg.ANALYSIS_BASE
    tokens = [base] * n_analysis_tokens

    max_up         = float(fut_features.get("max_up", 0.0))
    max_down       = float(fut_features.get("max_down", 0.0))   # <= 0
    vol_range      = float(fut_features.get("vol_range", 0.0))
    total_volume   = float(fut_features.get("total_volume", 0.0))
    zero_crossings = float(fut_features.get("zero_crossings", 0.0))
    direction_label = int(fut_features.get("direction_label", 0))

    # --------------------------------------------------------
    # 1) Tendência: 5 estados
    #    0 = strong_down, 1 = weak_down, 2 = flat,
    #    3 = weak_up,    4 = strong_up
    # --------------------------------------------------------
    trend_bin = 2  # default = flat

    if direction_label > 0:
        strength = max_up
        if strength >= 10.0:
            trend_bin = 4   # strong up
        elif strength >= 5.0:
            trend_bin = 3   # weak up
        else:
            trend_bin = 2   # flat
    elif direction_label < 0:
        strength = abs(max_down)
        if strength >= 10.0:
            trend_bin = 0   # strong down
        elif strength >= 5.0:
            trend_bin = 1   # weak down
        else:
            trend_bin = 2   # flat

    # trend: faixa [base + 0 .. base + 4]
    trend_token = base + trend_bin

    # --------------------------------------------------------
    # 2) Volatilidade (amplitude max_up - max_down)
    #    thresholds chutados, ajuste depois:
    #    [0..3), [3..6), [6..10), [10..20), >=20
    # --------------------------------------------------------
    vol_bin = _bucketize(
        vol_range,
        thresholds=[3.0, 6.0, 10.0, 20.0]
    )
    # vol: faixa [base + 5 .. base + 9]
    vol_token = base + 5 + vol_bin

    # --------------------------------------------------------
    # 3) Volume total
    # --------------------------------------------------------
    volu_bin = _bucketize(
        total_volume,
        thresholds=[100000.0, 200000.0, 300000.0, 500000.0]
    )
    # volume: faixa [base + 10 .. base + 14]
    volu_token = base + 10 + volu_bin

    # --------------------------------------------------------
    # 4) Choppiness (zero_crossings)
    # --------------------------------------------------------
    choppy_bin = _bucketize(
        zero_crossings,
        thresholds=[0.0, 2.0, 4.0, 6.0]
    )
    # choppy: faixa [base + 15 .. base + 19]
    choppy_token = base + 15 + choppy_bin

    # --------------------------------------------------------
    # Preenche os primeiros slots
    # --------------------------------------------------------
    if n_analysis_tokens > 0:
        tokens[0] = trend_token
    if n_analysis_tokens > 1:
        tokens[1] = vol_token
    if n_analysis_tokens > 2:
        tokens[2] = volu_token
    if n_analysis_tokens > 3:
        tokens[3] = choppy_token

    return torch.tensor(tokens, dtype=torch.long)



def compute_action_tokens(
    fut_features: Dict[str, float],
    tok_cfg: TokenConfig,
    n_action_tokens: int,
) -> torch.Tensor:
    """
    Gera tokens de ação com base em direction_label:

      direction_label:
        +1 -> BUY
        -1 -> SELL
         0 -> HOLD (flat / não entra)

    Usa o bloco começando em tok_cfg.ACTION_BASE.
    """

    direction_label = int(fut_features.get("direction_label", 0))

    if direction_label > 0:
        action_id = 0  # BUY
    elif direction_label < 0:
        action_id = 1  # SELL
    else:
        action_id = 2  # HOLD

    main_action_token = tok_cfg.ACTION_BASE + action_id

    tokens = [main_action_token] * n_action_tokens
    return torch.tensor(tokens, dtype=torch.long)



# ============================================================
#  CONSTRUIR JANELAS DO AGENTE PARA UM DIA
# ============================================================

def build_agent_sequences_for_day(
    day_key: str,
    day_tokens: torch.Tensor,
    events_path: Path,
    tok_cfg: TokenConfig,
    win_cfg: AgentWindowConfig,
    rng: random.Random,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Constrói várias janelas [seq_len] para um único dia.

    Retorna:
      - seqs: Tensor [N, seq_len]
      - centers: Tensor [N] com os índices centrais usados (p/ debug)
    """
    layout = win_cfg.layout()
    L = day_tokens.numel()

    # carrega eventos crus desse dia
    raw_events = np.load(events_path)  # TODO: ajuste se for .npz ou outro formato
    # assume que day_tokens e raw_events estão alinhados em índice, ou próximo disso.

    # limites para escolher o índice central
    min_center = win_cfg.n_past_tokens
    max_center = L - win_cfg.n_future_tokens - 1
    if max_center <= min_center:
        # dia muito curto
        return torch.empty(0, win_cfg.seq_len, dtype=torch.long), torch.empty(0, dtype=torch.long)

    possible_centers = list(range(min_center, max_center))

    # aplica stride ou amostragem
    indices: List[int] = []
    if win_cfg.max_episodes_per_day is not None:
        if len(possible_centers) <= win_cfg.max_episodes_per_day:
            indices = possible_centers
        else:
            indices = sorted(rng.sample(possible_centers, win_cfg.max_episodes_per_day))
    else:
        indices = possible_centers[::win_cfg.stride]

    seqs = []
    centers = []

    for c in indices:
        past_start = c - win_cfg.n_past_tokens
        past_end = c
        fut_start = c
        fut_end = c + win_cfg.n_future_tokens

        past_tokens = day_tokens[past_start:past_end]    # [n_past_tokens]
        fut_tokens = day_tokens[fut_start:fut_end]       # [n_future_tokens]

        # === contexto + estado ===
        ctx_tokens = compute_context_and_state_tokens(
            day_key=day_key,
            raw_events=raw_events,
            center_idx=c,
            tok_cfg=tok_cfg,
            n_ctx_tokens=win_cfg.n_ctx_tokens,
        )

        # === análise futura + ação ===
        fut_features = compute_future_features(raw_events, fut_start, fut_end)
        analysis_tokens = compute_analysis_tokens(
            fut_features, tok_cfg, win_cfg.n_analysis_tokens
        )
        action_tokens = compute_action_tokens(
            fut_features, tok_cfg, win_cfg.n_action_tokens
        )

        # === monta a sequência completa ===
        seq = torch.full(
            (win_cfg.seq_len,),
            fill_value=tok_cfg.PAD_TOKEN,
            dtype=torch.long,
        )

        # ctx/estado
        s, e = layout["ctx"]
        seq[s:e] = ctx_tokens

        # passado
        s, e = layout["past"]
        if (e - s) != past_tokens.shape[0]:
            raise RuntimeError("n_past_tokens inconsistente com layout.")
        seq[s:e] = past_tokens

        # separador de futuro
        seq[layout["sep_fut"]] = tok_cfg.SEP_FUT_TOKEN

        # futuro
        s, e = layout["fut"]
        if (e - s) != fut_tokens.shape[0]:
            raise RuntimeError("n_future_tokens inconsistente com layout.")
        seq[s:e] = fut_tokens

        # separador de análise
        seq[layout["sep_analysis"]] = tok_cfg.SEP_ANALYSIS_TOKEN

        # análise
        s, e = layout["analysis"]
        seq[s:e] = analysis_tokens

        # separador de ação
        seq[layout["sep_action"]] = tok_cfg.SEP_ACTION_TOKEN

        # ação
        s, e = layout["action"]
        seq[s:e] = action_tokens

        seqs.append(seq)
        centers.append(c)

    if not seqs:
        return torch.empty(0, win_cfg.seq_len, dtype=torch.long), torch.empty(0, dtype=torch.long)

    seqs_tensor = torch.stack(seqs, dim=0)
    centers_tensor = torch.tensor(centers, dtype=torch.long)
    return seqs_tensor, centers_tensor


# ============================================================
#  FUNÇÃO PRINCIPAL: build_agent_sequences
# ============================================================

def build_agent_sequences(
    tokens_by_day_pt: str | Path,
    events_dir: str | Path,
    win_cfg: Optional[AgentWindowConfig] = None,
    seed: int = 42,
):
    """
    Constrói o dataset de agente, a partir de:
      - tokens_by_day.pt
      - diretório com eventos crus (.npy)

    Retorna um dict:
      {
        "sequences": Tensor [N, seq_len],
        "day_indices": Tensor [N],
        "center_indices": Tensor [N],
        "day_keys": List[str],
        "win_cfg": AgentWindowConfig,
      }
    """
    if win_cfg is None:
        win_cfg = AgentWindowConfig()

    rng = random.Random(seed)

    tbd = load_tokens_by_day(tokens_by_day_pt)
    day_keys: List[str] = tbd["day_keys"]
    tokens_by_day: List[torch.Tensor] = tbd["tokens_by_day"]

    if len(day_keys) != len(tokens_by_day):
        raise ValueError("day_keys e tokens_by_day com tamanhos diferentes.")

    # mapa day_key -> arquivo cru
    day_to_path = build_day_to_events_path_map(events_dir, day_keys)

    all_seqs = []
    all_day_indices = []
    all_centers = []

    tok_cfg = TokenConfig()

    for i, dk in enumerate(day_keys):
        if dk not in day_to_path:
            # sem arquivo cru, pula
            continue

        events_path = day_to_path[dk]
        day_tokens = tokens_by_day[i]

        seqs, centers = build_agent_sequences_for_day(
            day_key=dk,
            day_tokens=day_tokens,
            events_path=events_path,
            tok_cfg=tok_cfg,
            win_cfg=win_cfg,
            rng=rng,
        )

        if seqs.numel() == 0:
            continue

        all_seqs.append(seqs)
        all_centers.append(centers)
        all_day_indices.append(torch.full_like(centers, fill_value=i))

    if not all_seqs:
        raise RuntimeError("Nenhuma sequência de agente foi gerada.")

    sequences = torch.cat(all_seqs, dim=0)
    day_indices = torch.cat(all_day_indices, dim=0)
    center_indices = torch.cat(all_centers, dim=0)

    print("============================================================")
    print(f"[AGENT DATASET] seq_len           : {win_cfg.seq_len}")
    print(f"[AGENT DATASET] total_sequences   : {sequences.size(0)}")
    print(f"[AGENT DATASET] num_days_covered  : {int(day_indices.unique().numel())}")
    print("============================================================")

    return {
        "sequences": sequences,          # [N, seq_len]
        "day_indices": day_indices,      # [N]
        "center_indices": center_indices,# [N]
        "day_keys": day_keys,
        "win_cfg": win_cfg,
    }


# ============================================================
#  EXEMPLO DE USO DIRETO (pode virar um script)
# ============================================================

if __name__ == "__main__":
    tokens_pt = str(PATHS.tokens_out / "tokens_by_day.pt")
    events_dir = str(PATHS.eventos_processados)

    win_cfg = AgentWindowConfig(
        seq_len=TransformerConfig.seq_len,
        n_ctx_tokens=16,
        n_past_tokens=320,
        n_future_tokens=64,
        n_analysis_tokens=4,
        n_action_tokens=2,
        stride=16,
        max_episodes_per_day=256,  # por ex., limita ~256 janelas por dia
    )

    agent_data = build_agent_sequences(
        tokens_by_day_pt=tokens_pt,
        events_dir=events_dir,
        win_cfg=win_cfg,
        seed=42,
    )

    # Exemplo de salvar pra usar no treino:
    out_path = Path(PATHS.tokens_out) / "agent_sequences.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(agent_data, out_path)
    print(f"[SAVE] agent_sequences salvos em: {out_path}")
