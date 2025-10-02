from collections import defaultdict, deque, Counter
import math
import numpy as np
from math import exp
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import ctypes
import mmap
import time
import win32event
import win32con
import pywintypes
from pathlib import Path
import re
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone
import matplotlib.dates as mdates
import pandas as pd

# ----------------- Config (knobs principais) -----------------
@dataclass
class SegParams:
    # ticks e pesos
    tick_a: float = 0.5      # WDO/WIN
    tick_b: float = 0.5      # DOL/IND
    lot_mult_a: float = 1.0  # WDO/WIN
    lot_mult_b: float = 5.0  # DOL/IND

    # EMAs (10–30 min p/ convergir)
    tau_vol_min: float = 5.0
    tau_var_min: float = 2.0
    var_win_sec: int = 120     # <- janela de cálculo do range: 2 minutos
    tau_order_min: float = 5.0   # ← NOVO: meia-vida para taxa de ordens (ordens/s)

    # COOLDOWN GLOBAL DE FRONTEIRA
    boundary_cooldown_s: int = 0   # tempo mínimo entre cortes (exceto time_pre)

    # --- VOL / PLAYER / RANGE ---
    start_vol_mult:  float = 5.0     # limiar relativo (EMA_vol) para volume de 1s
    vol_1s_hard:     float = 2000.0  # hard cut, independente da EMA
    player_mult:     float = 5.0     # limiar relativo (EMA_vol) para ordem agregada de player

    # --- RANGE (com decaimento) ---
    range_start_mult:  float = 1.0   # começa mais exigente vs var5m
    range_end_mult:    float = 0.4   # vai afrouxando até 1.0x

    # --- DECAIMENTO (vol e range) ---
    half_life_frac:    float = 1.0/3.0  # meia-vida ≈ 1/3 de max_dur_s

    # --- TEMPO ---
    max_dur_s: int = 60   # pode aumentar um pouco, já que não temos "calm"

    # === RENKO (inversão) ===
    renko_nticks: int = 2        # tamanho do tijolo em ticks
    renko_symbol: str = "a"      # "a" (WDO/WIN), "b" (DOL/IND)

    # === TAXA DE NEGÓCIOS (“ordens originais”/s) ===
    tau_order_rate_min: float = 8.0   # EMA de taxa (min)
    order_rate_mult: float = 1.5      # corte: rate >= mult * EMA por >= Ns
    order_rate_consec_sec: int = 2

    # === SPEED por preço (baseline global; sem decaimento no evento) ===
    speed_price_mult: float = 20.0     # corte: speed_now >= mult * base_rate_now
    speed_price_consec_sec: int = 3  # precisa manter por N segundos
    min_event_sec_speed: int = 3      # idade mínima do evento p/ poder cortar por speed

    # === PLAYER e VOL (sem decaimento global; vol tem decaimento por evento) ===
    player_mult_vs_vol_ema: float = 5.0

    # === RANGE (com decaimento por evento) ===
    range_mult_start: float = 1.5
    range_mult_end: float = 0.8

    # half-life em ~ 1/3 do tempo máximo (para vol e range)
    decay_half_life_frac: float = 1.0/3.0

    # --- Absorção (detecção por retrace do RENKO/close de WDO) ---
    absorb_ticks_thr: int = 2         # retrace mínimo (em ticks) para contar absorção
    absorb_max_wait_s: int = 10       # janelo máx (s) para completar o retrace após uma pernada

    # --- Referência de preço/participação (qual símbolo usar como "preço") ---
    price_ref_symbol: str = "a"       # "a" = WDO (no seu par WDO+DOL), "b" = DOL

    # --- Players top do dia (opcional) ---
    top_buyers_a: tuple = ()          # ids/agências compradoras "top" (WDO)
    top_sellers_a: tuple = ()
    top_buyers_b: tuple = ()          # ids/agências compradoras "top" (DOL)
    top_sellers_b: tuple = ()

# ----------------- Util -----------------
def linear_decay(t: int, start_mult: float, end_mult: float, t_max: int) -> float:
    if t <= 0: return start_mult
    if t >= t_max: return end_mult
    lam = t / float(t_max)
    return (1.0 - lam) * start_mult + lam * end_mult

def ticks_move(p_from: Optional[float], p_to: Optional[float], tick: float) -> float:
    if p_from is None or p_to is None or tick <= 0: return 0.0
    return (p_to - p_from) / tick

def summarize_second(trades: List[List[float]]) -> dict:
    if not trades:
        return {"n":0,"vol":0.0,"lot_max":0.0,"open":None,"close":None,"pmin":None,"pmax":None}
    prices = [float(tr[0]) for tr in trades]
    lots   = [float(tr[1]) if len(tr)>1 else 1.0 for tr in trades]
    return {
        "n": len(trades),
        "vol": float(sum(lots)),
        "lot_max": float(max(lots)),
        "open": prices[0],
        "close": prices[-1],
        "pmin": min(prices),
        "pmax": max(prices),
    }

def aggregate_orders_by_player(trades: List[List[float]]) -> List[dict]:
    # cada trade: [preço, lote, broker_comp, broker_vend, aggr, ...]
    out = []
    cur = None
    for tr in trades:
        preco = float(tr[0])
        lote  = float(tr[1]) if len(tr)>1 else 1.0
        aggr  = int(tr[4]) if len(tr)>4 else 0
        side  = 1 if aggr>0 else (-1 if aggr<0 else 0)
        b_agg = int(tr[2]) if side==1 else (int(tr[3]) if side==-1 else -1)
        if cur and side==cur["side"] and b_agg==cur["broker"]:
            cur["lot"] += lote; cur["p_close"]=preco
            cur["pmin"]=min(cur["pmin"],preco); cur["pmax"]=max(cur["pmax"],preco)
            cur["n"]+=1
        else:
            if cur: out.append(cur)
            cur={"side":side,"broker":b_agg,"lot":lote,"p_open":preco,"p_close":preco,"pmin":preco,"pmax":preco,"n":1}
    if cur: out.append(cur)
    return out

def safe_min(a, b):
    if a is None: return b
    if b is None: return a
    return min(a, b)

def safe_max(a, b):
    if a is None: return b
    if b is None: return a
    return max(a, b)

def sign(x: float) -> int:
    return 1 if x > 0 else (-1 if x < 0 else 0)

def ticks_delta(p0, p1, tick) -> float:
    if p0 is None or p1 is None or tick <= 0: return 0.0
    return (float(p1) - float(p0)) / float(tick)

def _vol_by_aggr(trades):
    """retorna (buy_vol, sell_vol, total_vol) para uma lista de trades [p, lot, bcomp, bvend, aggr]."""
    buy = sell = tot = 0.0
    for tr in trades:
        lot = float(tr[1]) if len(tr)>1 else 1.0
        ag  = int(tr[4]) if len(tr)>4 else 0
        tot += lot
        if ag > 0: buy += lot
        elif ag < 0: sell += lot
    return buy, sell, tot

def _group_stats_one_symbol(trades, tick):
    """
    Usa aggregate_orders_by_player (contíguo dentro do segundo) p/ achar:
      - maior lote de um grupo,
      - maior deslocamento (ticks) de um grupo,
      - contagem de grupos por lado (+1, -1, 0).
    """
    groups = aggregate_orders_by_player(trades)
    max_lot = 0.0
    max_ticks = 0.0
    g_buy = g_sell = g_neu = 0
    for g in groups:
        max_lot = max(max_lot, float(g["lot"]))
        dtk = abs(ticks_delta(g.get("p_open"), g.get("p_close"), tick))
        max_ticks = max(max_ticks, dtk)
        if g["side"] > 0:   g_buy += 1
        elif g["side"] < 0: g_sell += 1
        else:               g_neu += 1
    return max_lot, max_ticks, g_buy, g_sell, g_neu

def _bucket_price_stats_wdo(trades_a, close_a):
    """
    Para WDO (símbolo 'a'): acumula:
      - contagem de negócios por preço,
      - volume por preço,
      - segundos no preço (usando close do segundo).
    """
    price_count = defaultdict(int)
    price_vol   = defaultdict(float)
    for tr in trades_a:
        p = float(tr[0]); lot = float(tr[1]) if len(tr)>1 else 1.0
        price_count[p] += 1
        price_vol[p]   += lot
    sec_price = None if close_a is None else float(close_a)
    return price_count, price_vol, sec_price

def _to_float(x):
    try:
        return float(x)
    except Exception:
        return None

def _is_finite(x):
    return (x is not None) and math.isfinite(x)

def safe_mean(seq, fallback=None):
    vals = []
    for v in (seq or []):
        f = _to_float(v)
        if _is_finite(f):
            vals.append(f)
    if vals:
        return float(sum(vals) / len(vals))
    fb = _to_float(fallback)
    return float(fb) if _is_finite(fb) else 0.0

# ----------------- Estado das referências -----------------
@dataclass
class RefMetrics:
    alpha_vol: float
    alpha_var: float
    alpha_rate: float
    ema_vol_sec: float = 1.0
    ema_var5m: float = 1.0
    ema_order_rate: float = 1.0
    buf_a: List[float] = field(default_factory=list)
    buf_b: List[float] = field(default_factory=list)

    @classmethod
    def create(cls, tau_vol_min: float, tau_var_min: float, tau_order_min: float) -> "RefMetrics":
        # alphas por segundo (EMA)
        a_vol  = 1.0 - exp(-1.0 / max(1.0, tau_vol_min  * 60.0))
        a_var  = 1.0 - exp(-1.0 / max(1.0, tau_var_min  * 60.0))
        a_rate = 1.0 - exp(-1.0 / max(1.0, tau_order_min* 60.0))
        return cls(a_vol, a_var, a_rate)

    def update(self,
               close_a: Optional[float],
               close_b: Optional[float],
               vol_comb: float,
               orders_started_this_sec: float):
        # EMA do volume por segundo
        self.ema_vol_sec = (1.0 - self.alpha_vol) * self.ema_vol_sec + self.alpha_vol * float(vol_comb)

        # buffers de preço para var (range) de ~N minutos
        if close_a is not None:
            self.buf_a.append(float(close_a)); self.buf_a = self.buf_a[-300:]  # ~5min se 1Hz
        if close_b is not None:
            self.buf_b.append(float(close_b)); self.buf_b = self.buf_b[-300:]

        var5_a = (max(self.buf_a) - min(self.buf_a)) if len(self.buf_a) >= 2 else 0.0
        var5_b = (max(self.buf_b) - min(self.buf_b)) if len(self.buf_b) >= 2 else 0.0
        # EMA da “variação 5m” (usaremos em ticks no step)
        self.ema_var5m = (1.0 - self.alpha_var) * self.ema_var5m + self.alpha_var * max(var5_a, var5_b)

        # EMA da taxa de ordens (ordens/s)
        self.ema_order_rate = (1.0 - self.alpha_rate) * self.ema_order_rate + self.alpha_rate * float(orders_started_this_sec)

# ----------------- Detector -----------------

def exp_decay_mult(t: int, start_mult: float, end_mult: float, half_life_sec: float) -> float:
    """Multiplicador que vai de start_mult para end_mult com decaimento ~exponencial."""
    if t <= 0: return start_mult
    # fator que cai pela metade a cada half_life_sec
    k = 0.5 ** (t / max(half_life_sec, 1e-9))
    return end_mult + (start_mult - end_mult) * k

def coalesce_orders_one_sec(trades, last_open):
    """
    Agrupa ‘ordem original’ por broker+lado preservando ordem.
    Retorna:
      starts (qtd de novas ordens iniciadas neste segundo)
      last_open (ordem aberta para carry p/ próximo segundo)
    """
    starts = 0
    cur = last_open  # pode vir carregada do segundo anterior
    def _side(tr): 
        a = int(tr[4]) if len(tr)>4 else 0
        return 1 if a>0 else (-1 if a<0 else 0)
    def _broker(tr, side):
        return int(tr[2]) if side==1 else (int(tr[3]) if side==-1 else -1)

    for tr in trades:
        s = _side(tr)
        if s == 0:  # trade neutro: encerra a corrente
            if cur: cur = None
            continue
        b = _broker(tr, s)
        if cur and cur["side"]==s and cur["broker"]==b:
            # continua mesma “ordem”
            cur["lot"] += float(tr[1]) if len(tr)>1 else 1.0
        else:
            # inicia nova “ordem”
            starts += 1
            cur = {"side":s, "broker":b, "lot": float(tr[1]) if len(tr)>1 else 1.0}
    return starts, cur

def renko_reversal_this_second(anchor, direction, pmin, pmax, brick):
    """
    Detecta reversão Renko ‘clássica’: precisa andar 2 tijolos na direção oposta
    desde o último close. Usa high/low do segundo.
    anchor: último close do renko
    direction: +1/-1/0
    brick: tamanho do tijolo em preço
    Retorna (reversal_bool, new_anchor, new_dir)
    """
    if anchor is None or pmin is None or pmax is None:
        return False, anchor, direction

    rev = False
    # Regras
    if direction >= 0:
        # queda >= 2 tijolos a partir do anchor
        if pmin <= anchor - 2*brick:
            rev = True
            # novo close (primeiro tijolo oposto)
            anchor = anchor - brick
            direction = -1
    if direction <= 0 and not rev:
        # alta >= 2 tijolos
        if pmax >= anchor + 2*brick:
            rev = True
            anchor = anchor + brick
            direction = +1

    # Ajusta anchor com tijolos adicionais na MESMA direção, se passou mais que 1
    if direction == +1 and pmax is not None:
        n_up = int((pmax - anchor) // brick)
        if n_up > 0:
            anchor += n_up * brick
    if direction == -1 and pmin is not None:
        n_dn = int((anchor - pmin) // brick)
        if n_dn > 0:
            anchor -= n_dn * brick

    return rev, anchor, direction

def event_acc_init():
    return {
        # --- preço (WDO) ---
        "price_count": defaultdict(int),     # trades por preço (WDO)
        "price_vol":   defaultdict(float),   # volume por preço (WDO)
        "secs_at_price": defaultdict(int),   # quantos segundos o close==preço (WDO)

        "max_group_lot": 0.0,                # maior lote agrupado (A+B)
        "max_group_ticks": 0.0,              # maior deslocamento (ticks) num grupo (A/B)

        # --- volume ---
        "buy_vol_a": 0.0, "sell_vol_a": 0.0, "tot_vol_a": 0.0,
        "buy_vol_b": 0.0, "sell_vol_b": 0.0, "tot_vol_b": 0.0,

        # --- grupos (ordens originais aproximadas por segundo) ---
        "g_buy": 0, "g_sell": 0, "g_neu": 0,  # contagem de grupos
        "orders_started_sum": 0,              # soma de starts_a+starts_b
        "rate_max": 0.0,                      # máximo rate dentro do evento

        # --- streak por lado (domínio por segundo) ---
        "streak_buy": 0, "streak_sell": 0, "max_streak_buy": 0, "max_streak_sell": 0,

        # --- movimento por segundo (base WDO) ---
        "persec": [],                         # lista de dicts por segundo com deltas e vols

        # --- run p/ absorção (base WDO) ---
        "run_sign": 0, "run_ticks": 0.0, "run_secs": 0, "run_dir_vol": 0.0,
        "retrace_ticks": 0.0, "retrace_secs": 0,
        "absorb_buy": [],   # lista de {'ticks':x,'secs':y}
        "absorb_sell": [],

        # --- players top (acúmulo) ---
        "top_buy_vol_a": 0.0, "top_sell_vol_a": 0.0,
        "top_buy_vol_b": 0.0, "top_sell_vol_b": 0.0,
    }

def event_accumulate_second(ev_acc: dict,
                            trades_a, trades_b,
                            smry_a, smry_b,
                            close_a, close_b,
                            last_close_a, last_close_b,
                            tick_a, tick_b,
                            starts_a: int, starts_b: int,
                            p: SegParams):
    # ---- preço (WDO) ----
    price_count, price_vol, sec_price = _bucket_price_stats_wdo(trades_a, close_a)
    for k,v in price_count.items(): ev_acc["price_count"][k] += v
    for k,v in price_vol.items():   ev_acc["price_vol"][k]   += v
    if sec_price is not None:
        ev_acc["secs_at_price"][sec_price] += 1

    # ---- volume por agressão ----
    buy_a, sell_a, tot_a = _vol_by_aggr(trades_a)
    buy_b, sell_b, tot_b = _vol_by_aggr(trades_b)
    ev_acc["buy_vol_a"]  += buy_a;  ev_acc["sell_vol_a"] += sell_a;  ev_acc["tot_vol_a"] += tot_a
    ev_acc["buy_vol_b"]  += buy_b;  ev_acc["sell_vol_b"] += sell_b;  ev_acc["tot_vol_b"] += tot_b

    # ---- grupos (por símbolo dentro do segundo) ----
    mlot_a, mtick_a, g_buy_a, g_sell_a, g_neu_a = _group_stats_one_symbol(trades_a, tick_a)
    mlot_b, mtick_b, g_buy_b, g_sell_b, g_neu_b = _group_stats_one_symbol(trades_b, tick_b)
    ev_acc["max_group_lot"]   = max(ev_acc["max_group_lot"], mlot_a, mlot_b)
    ev_acc["max_group_ticks"] = max(ev_acc["max_group_ticks"], mtick_a, mtick_b)

    # contagem de grupos por lado (aproximação)
    ev_acc["g_buy"]  += (g_buy_a + g_buy_b)
    ev_acc["g_sell"] += (g_sell_a + g_sell_b)
    ev_acc["g_neu"]  += (g_neu_a + g_neu_b)

    # taxa de ordens originais
    rate_now = float(starts_a + starts_b)
    ev_acc["orders_started_sum"] += rate_now
    ev_acc["rate_max"] = max(ev_acc["rate_max"], rate_now)

    # streak por segundo (lado "dominante" por número de grupos)
    if (g_buy_a+g_buy_b) > (g_sell_a+g_sell_b):
        ev_acc["streak_buy"]  += 1
        ev_acc["streak_sell"]  = 0
    elif (g_sell_a+g_sell_b) > (g_buy_a+g_buy_b):
        ev_acc["streak_sell"] += 1
        ev_acc["streak_buy"]   = 0
    else:
        ev_acc["streak_buy"] = 0
        ev_acc["streak_sell"] = 0
    ev_acc["max_streak_buy"]  = max(ev_acc["max_streak_buy"],  ev_acc["streak_buy"])
    ev_acc["max_streak_sell"] = max(ev_acc["max_streak_sell"], ev_acc["streak_sell"])

    # ---- movimento por segundo (base no símbolo de preço de referência) ----
    use_a = (p.price_ref_symbol.lower() == "a")
    dtk_a = ticks_delta(last_close_a, close_a, tick_a)
    dtk_b = ticks_delta(last_close_b, close_b, tick_b)
    dtk_ref = dtk_a if use_a else dtk_b

    # volume agressor da direção no ref-symbol (para V/tick)
    # usamos WDO (a) e DOL (b) para rateios e métricas
    dir_buy_vol  = (buy_a + buy_b)
    dir_sell_vol = (sell_a + sell_b)

    ev_acc["persec"].append({
        "dtk_a": dtk_a, "dtk_b": dtk_b,
        "buy_a": buy_a, "sell_a": sell_a,
        "buy_b": buy_b, "sell_b": sell_b,
        "dtk_ref": dtk_ref
    })

    # ---- absorção (com base na pernada do ref e retrace >= absorb_ticks_thr) ----
    sgn = sign(dtk_ref)
    if sgn == 0:
        # nada muda
        pass
    elif ev_acc["run_sign"] == 0:
        # inicia uma pernada
        ev_acc["run_sign"]  = sgn
        ev_acc["run_ticks"] = abs(dtk_ref)
        ev_acc["run_secs"]  = 1
        ev_acc["retrace_ticks"] = 0.0
        ev_acc["retrace_secs"]  = 0
        ev_acc["run_dir_vol"]   = dir_buy_vol if sgn>0 else dir_sell_vol
    elif sgn == ev_acc["run_sign"]:
        # continua a pernada
        ev_acc["run_ticks"] += abs(dtk_ref)
        ev_acc["run_secs"]  += 1
        ev_acc["run_dir_vol"] += (dir_buy_vol if sgn>0 else dir_sell_vol)
        # reset retrace
        ev_acc["retrace_ticks"] = 0.0
        ev_acc["retrace_secs"]  = 0
    else:
        # retrace em andamento
        ev_acc["retrace_ticks"] += abs(dtk_ref)
        ev_acc["retrace_secs"]  += 1
        if ev_acc["retrace_ticks"] >= p.absorb_ticks_thr and ev_acc["run_ticks"] >= p.absorb_ticks_thr:
            # conta absorção: se subiu e retraiu -> absorção na venda; se caiu e retraiu -> absorção na compra
            rec = {"ticks": ev_acc["retrace_ticks"], "secs": ev_acc["retrace_secs"]}
            if ev_acc["run_sign"] > 0:  # pernada foi de compra
                ev_acc["absorb_sell"].append(rec)
            else:
                ev_acc["absorb_buy"].append(rec)
            # zera e começa nova pernada na direção atual
            ev_acc["run_sign"]  = sgn
            ev_acc["run_ticks"] = abs(dtk_ref)
            ev_acc["run_secs"]  = 1
            ev_acc["run_dir_vol"] = (dir_buy_vol if sgn>0 else dir_sell_vol)
            ev_acc["retrace_ticks"] = 0.0
            ev_acc["retrace_secs"]  = 0

    # ---- players top do dia (se fornecido) ----
    if p.top_buyers_a or p.top_sellers_a:
        for tr in trades_a:
            lot = float(tr[1]) if len(tr)>1 else 1.0
            ag  = int(tr[4]) if len(tr)>4 else 0
            b_comp = int(tr[2]) if len(tr)>2 else -1
            b_vend = int(tr[3]) if len(tr)>3 else -1
            if ag > 0 and b_comp in p.top_buyers_a:  ev_acc["top_buy_vol_a"]  += lot
            if ag < 0 and b_vend in p.top_sellers_a: ev_acc["top_sell_vol_a"] += lot
    if p.top_buyers_b or p.top_sellers_b:
        for tr in trades_b:
            lot = float(tr[1]) if len(tr)>1 else 1.0
            ag  = int(tr[4]) if len(tr)>4 else 0
            b_comp = int(tr[2]) if len(tr)>2 else -1
            b_vend = int(tr[3]) if len(tr)>3 else -1
            if ag > 0 and b_comp in p.top_buyers_b:  ev_acc["top_buy_vol_b"]  += lot
            if ag < 0 and b_vend in p.top_sellers_b: ev_acc["top_sell_vol_b"] += lot

def extrair_preco(x):
    """
    Retorna um vetor de 5 características periódicas:
    [suave_5_0(x % 10), suave_5_0(x % 100),
     onda_quebrada(x % 10), onda_quebrada(x % 100),
     funcao_alvo(x % 100)]
    """

    def suave_5_0(x):
        resto = x % 10
        return np.cos(np.pi * resto / 5)

    def quebra_no_bola(x):
        frac = x % 10
        return -np.cos(np.pi * frac / 10)

    def cosseno_localizado(x, centro, largura=15):
        x_mod = x % 100
        dist = np.abs(x_mod - centro)
        y = np.full_like(x_mod, -1.0, dtype=float)
        dentro = dist <= largura
        y[dentro] = np.cos(np.pi * (x_mod[dentro] - centro) / largura)
        return y

    def funcao_357(x):
        picos = [0, 30, 50, 70, 100]
        largura = 15
        x_arr = np.asarray(x, dtype=float)
        y = np.full_like(x_arr, -1.0)
        for c in picos:
            y = np.maximum(y, cosseno_localizado(x_arr, c, largura))
        return y

    # Extrai características
    s10 = suave_5_0(x)
    s100 = suave_5_0(x / 10)
    o10 = quebra_no_bola(x)
    o100 = quebra_no_bola(x / 10)
    f100 = funcao_357(x)

    return np.stack([s10, s100, o10, o100, f100], axis=-1).tolist()

def finalize_event_enrichment(ev: dict,
                              p: SegParams,
                              metrics: "RefMetrics",
                              tick_a: float, tick_b: float,
                              extrair_preco_fn):
    acc = ev.pop("acc", None)
    if not acc: 
        return ev  # nada a fazer

    # ====== PRICE INFO (WDO) ======
    def _argmax_in_dict(dct):
        if not dct: return None, 0.0
        k = max(dct.items(), key=lambda kv: kv[1])[0]
        return k, dct[k]

    price_most_traded, _ = _argmax_in_dict(acc["price_count"])
    price_most_volume,  _ = _argmax_in_dict(acc["price_vol"])
    price_most_time,    _ = _argmax_in_dict(acc["secs_at_price"])

    ev["price_info"] = {
        "open": ev.get("p0_a"),
        "close": ev.get("plast_a"),
        "high": ev.get("pmax_a"),
        "low":  ev.get("pmin_a"),
        "price_most_traded": price_most_traded,
        "price_most_volume": price_most_volume,
        "price_most_time":   price_most_time,
        "max_group_ticks":   acc["max_group_ticks"],  # máx variação (ticks) num grupo
    }

    # ====== VOLUME INFO ======
    buy_a, sell_a, tot_a = acc["buy_vol_a"], acc["sell_vol_a"], acc["tot_vol_a"]
    buy_b, sell_b, tot_b = acc["buy_vol_b"], acc["sell_vol_b"], acc["tot_vol_b"]
    tot_all = tot_a + tot_b
    n_orders = max(1.0, float(acc["orders_started_sum"]))  # aprox. somatório por segundo
    ev["volume_info"] = {
        "buy_vol": buy_a + buy_b,
        "sell_vol": sell_a + sell_b,
        "total_vol_incl_cross": tot_all,
        "avg_vol_per_order": tot_all / n_orders,
        "max_group_vol": acc["max_group_lot"],
    }

    # ====== TRADE INFO ======
    ev["trade_info"] = {
        "groups_buy": acc["g_buy"],
        "groups_sell": acc["g_sell"],
        "groups_neutral": acc["g_neu"],
        "rate_avg": acc["orders_started_sum"] / max(1, int(ev["dur"])),
        "rate_max": acc["rate_max"],
        "max_streak_buy": acc["max_streak_buy"],
        "max_streak_sell": acc["max_streak_sell"],
    }

    # ====== MOVE INFO ======
    # variações em ticks open->close com guardas
    def _abs_dtk(p0, p1, tk):
        if p0 is None or p1 is None or not tk or tk <= 0:
            return 0.0
        return abs((float(p1) - float(p0)) / float(tk))

    dtk_a = _abs_dtk(ev.get("p0_a"), ev.get("plast_a"), tick_a)
    dtk_b = _abs_dtk(ev.get("p0_b"), ev.get("plast_b"), tick_b)

    # volume por tick (por segundo, lado e símbolo) -> média/mín/máx
    def _vol_per_tick_stats(persec, symbol="a"):
        vals_up, vals_dn = [], []
        for r in persec:
            if symbol == "a":
                dtk = r["dtk_a"]; buy = r["buy_a"]; sell = r["sell_a"]
            else:
                dtk = r["dtk_b"]; buy = r["buy_b"]; sell = r["sell_b"]
            adtk = abs(dtk)
            if adtk <= 0:
                continue
            if dtk > 0 and buy > 0:  vals_up.append(buy/adtk)
            if dtk < 0 and sell > 0: vals_dn.append(sell/adtk)
        def _stats(lst):
            if not lst: return {"mean":0.0,"min":0.0,"max":0.0,"n":0}
            import numpy as _np
            return {"mean": float(_np.mean(lst)), "min": float(_np.min(lst)), "max": float(_np.max(lst)), "n": int(len(lst))}
        return _stats(vals_up), _stats(vals_dn)

    up_a, dn_a = _vol_per_tick_stats(acc["persec"], "a")
    up_b, dn_b = _vol_per_tick_stats(acc["persec"], "b")

    # participação do WDO no deslocamento (por lado)
    buy_a, sell_a, tot_a = acc["buy_vol_a"], acc["sell_vol_a"], acc["tot_vol_a"]
    buy_b, sell_b, tot_b = acc["buy_vol_b"], acc["sell_vol_b"], acc["tot_vol_b"]
    tot_buy  = buy_a  + buy_b
    tot_sell = sell_a + sell_b
    buy_share_a  = (buy_a  / tot_buy)  if tot_buy  > 0 else 0.0
    sell_share_a = (sell_a / tot_sell) if tot_sell > 0 else 0.0

    # absorções (já calculado acima)
    def _abs_stats(lst):
        if not lst: return {"count":0,"ticks_mean":0.0,"ticks_max":0.0,"vel_mean":0.0}
        import numpy as _np
        ticks = [x["ticks"] for x in lst]
        vels  = [x["ticks"]/max(1,x["secs"]) for x in lst]
        return {
            "count": len(lst),
            "ticks_mean": float(_np.mean(ticks)),
            "ticks_max":  float(_np.max(ticks)),
            "vel_mean":   float(_np.mean(vels)),
        }

    abs_buy  = _abs_stats(acc["absorb_buy"])
    abs_sell = _abs_stats(acc["absorb_sell"])

    # facilidade de movimento (usa símbolo de referência definido em params)
    use_a = (p.price_ref_symbol.lower() == "a")
    ref_tick  = tick_a if use_a else tick_b
    ref_open  = ev.get("p0_a")   if use_a else ev.get("p0_b")
    ref_close = ev.get("plast_a") if use_a else ev.get("plast_b")

    # net_ticks e direção segura
    net_ticks = _abs_dtk(ref_open, ref_close, ref_tick)
    if ref_open is None or ref_close is None:
        dir_sign = 0
    else:
        diff = float(ref_close) - float(ref_open)
        dir_sign = 1 if diff > 0 else (-1 if diff < 0 else 0)

    # se subiu, resistência é sell_vol; se caiu, resistência é buy_vol; se flat, 0
    opp_vol = (sell_a + sell_b) if dir_sign > 0 else ((buy_a + buy_b) if dir_sign < 0 else 0.0)
    ease_of_move = (net_ticks / max(1.0, opp_vol)) if dir_sign != 0 else 0.0

    ev["move_info"] = {
        "range_ticks": ev.get("range_ticks", 0.0),
        "d_ticks_max": ev.get("d_ticks_max", 0.0),
        "dtk_open_close_a": dtk_a,
        "dtk_open_close_b": dtk_b,
        "vol_per_tick_up_a":   up_a,
        "vol_per_tick_dn_a":   dn_a,
        "vol_per_tick_up_b":   up_b,
        "vol_per_tick_dn_b":   dn_b,
        "buy_share_wdo":  buy_share_a,
        "sell_share_wdo": sell_share_a,
        "absorptions_buy":  abs_buy,
        "absorptions_sell": abs_sell,
        "ease_of_move": ease_of_move,
    }


    # ====== PLAYERS INFO (se top lists fornecidas) ======
    ev["players_info"] = {
        "top_buy_vol_wdo":  acc["top_buy_vol_a"],
        "top_sell_vol_wdo": acc["top_sell_vol_a"],
        "top_buy_vol_dol":  acc["top_buy_vol_b"],
        "top_sell_vol_dol": acc["top_sell_vol_b"],
    }

    # ====== CONTEXT INFO (preço de referência = WDO por padrão) ======
    use_a = (p.price_ref_symbol.lower() == "a")  # "a"=WDO, "b"=DOL
    ctx_price = ev.get("plast_a") if use_a else ev.get("plast_b")
    # fallback cruzado e, por fim, 0.0
    if ctx_price is None:
        ctx_price = ev.get("plast_b") if use_a else ev.get("plast_a")
    if ctx_price is None:
        ctx_price = 0.0

    # médias robustas (filtram None/NaN/inf e usam ctx_price como fallback)
    if use_a:
        sma15m = safe_mean(getattr(metrics, "buf15_a", []), ctx_price)
    else:
        sma15m = safe_mean(getattr(metrics, "buf15_b", []), ctx_price)

    # features periódicas do preço (vetor 5D) – garante entrada float
    try:
        import numpy as _np
        ctx_feats = extrair_preco(_np.array([float(ctx_price)], dtype=float))[0]
    except Exception:
        ctx_feats = [0.0, 0.0, 0.0, 0.0, 0.0]

    ev["context_info"] = {
        "ctx_price": float(ctx_price),
        "ema_vol_sec": float(getattr(metrics, "ema_vol_sec", 0.0) or 0.0),
        "ema_order_rate": float(getattr(metrics, "ema_order_rate", 0.0) or 0.0),
        "sma15m_price_wdo": float(sma15m),
        "price_feats": ctx_feats,            # 5 features periódicas
        "active_flags": list(ev.get("start_flags", {}).keys()),
        "end_flags": ev.get("end_reason", ""),
    }

    return ev

@dataclass
class EventSegmenter:
    p: SegParams
    # estado dinâmico
    metrics: RefMetrics = field(init=False)
    evt: Optional[dict] = None
    calm_streak: int = 0
    last_event_end: Optional[int] = None
    last_close_a: Optional[float] = None
    last_close_b: Optional[float] = None
    # runs p/ inversão
    run_sign_a: int = 0
    run_ticks_a: int = 0
    run_secs_a: int = 0
    run_sign_b: int = 0
    run_ticks_b: int = 0
    run_secs_b: int = 0
    speed_hi_streak: int = 0               # contador de segundos com speed acima do limiar
    last_boundary_at: Optional[int] = None # último segundo em que cortou
    # renko
    renko_anchor: Optional[float] = None
    renko_dir: int = 0   # +1 up, -1 down, 0 neutro

    # coalescer de “ordem original” (carrega entre segundos)
    _last_order_a: Optional[dict] = None
    _last_order_b: Optional[dict] = None

    # streaks
    rate_hi_streak: int = 0
    speed_price_streak: int = 0

    def __post_init__(self):
        self.metrics = RefMetrics.create(self.p.tau_vol_min, self.p.tau_var_min, self.p.tau_order_min)


    def reset(self):
        self.__post_init__()
        self.evt = None; self.calm_streak = 0
        self.last_event_end = None
        self.last_close_a = None; self.last_close_b = None
        self.run_sign_a = self.run_ticks_a = self.run_secs_a = 0
        self.run_sign_b = self.run_ticks_b = self.run_secs_b = 0
        self.speed_hi_streak = 0
        self.last_boundary_at = None

    # ---- passo por segundo: retorna eventos FINALIZADOS neste segundo (0 ou 1) ----
    def step(self, s: int, trades_a_raw: List[List[float]], trades_b_raw: List[List[float]]) -> List[dict]:
        # aplica multiplicadores de lote
        trades_a = [ [t[0], (t[1]*self.p.lot_mult_a)] + t[2:] for t in trades_a_raw ]
        trades_b = [ [t[0], (t[1]*self.p.lot_mult_b)] + t[2:] for t in trades_b_raw ]

        smry_a = summarize_second(trades_a)
        smry_b = summarize_second(trades_b)

        # closes atuais
        close_a = smry_a["close"] if smry_a["close"] is not None else self.last_close_a
        close_b = smry_b["close"] if smry_b["close"] is not None else self.last_close_b
        vol_comb = smry_a["vol"] + smry_b["vol"]

        # ===== taxa de "ordens originais" (A e B) =====
        starts_a, self._last_order_a = coalesce_orders_one_sec(trades_a, self._last_order_a)
        starts_b, self._last_order_b = coalesce_orders_one_sec(trades_b, self._last_order_b)
        orders_started = starts_a + starts_b

        # ===== atualiza referências globais =====
        # (sua RefMetrics.update aceita 4 args: close_a, close_b, vol_comb, orders_started)
        self.metrics.update(close_a, close_b, vol_comb, orders_started)

        # baseline de variação convertida para ticks (janela ~ tau_var_min)
        ema_var5m_ticks = max(
            self.metrics.ema_var5m / max(self.p.tick_a, 1e-9),
            self.metrics.ema_var5m / max(self.p.tick_b, 1e-9)
        )

        # ===== RENKO (inversão) =====
        use_a = (self.p.renko_symbol.lower() == "a")
        pmin = smry_a["pmin"] if use_a else smry_b["pmin"]
        pmax = smry_a["pmax"] if use_a else smry_b["pmax"]
        tick = self.p.tick_a if use_a else self.p.tick_b
        brick = self.p.renko_nticks * max(tick, 1e-9)

        if self.renko_anchor is None:
            base = close_a if use_a else close_b
            if base is not None:
                self.renko_anchor = round(base / brick) * brick
                self.renko_dir = 0

        inv_hit = False
        if self.renko_anchor is not None:
            inv_hit, self.renko_anchor, self.renko_dir = renko_reversal_this_second(
                self.renko_anchor, self.renko_dir, pmin, pmax, brick
            )

        # ===== SPEED por preço (instantânea por segundo) =====
        # usa |close - last_close|/tick de cada símbolo e pega o maior
        def _ticks_this_sec(last_close, close, tk):
            if last_close is None or close is None or tk <= 0: return 0.0
            return abs((float(close) - float(last_close)) / float(tk))

        ticks_a_1s = _ticks_this_sec(self.last_close_a, close_a, self.p.tick_a)
        ticks_b_1s = _ticks_this_sec(self.last_close_b, close_b, self.p.tick_b)
        ticks_per_s_this_sec = max(ticks_a_1s, ticks_b_1s)  # “speed” instantânea (ticks/s)

        # baseline global de ticks/s vindo da var 2–5min (range/300s)
        base_rate_now = max(ema_var5m_ticks / 120.0, 1e-9)
        speed_hi = (ticks_per_s_this_sec >= abs(self.p.speed_price_mult * base_rate_now))
        self.speed_price_streak = (self.speed_price_streak + 1) if speed_hi else 0
        speed_price_cut = (self.evt is not None) and \
                        (self.evt["dur"] >= self.p.min_event_sec_speed) and \
                        (self.speed_price_streak >= self.p.speed_price_consec_sec)

        # ===== RATE de ordens originais (global) =====
        rate_now = float(orders_started)  # ordens/s neste segundo
        rate_hi = (rate_now >= self.p.order_rate_mult * max(self.metrics.ema_order_rate, 1e-9))
        self.rate_hi_streak = (self.rate_hi_streak + 1) if rate_hi else 0
        order_rate_cut = (self.rate_hi_streak >= self.p.order_rate_consec_sec)

        # ===== PLAYER e VOLUME (em s atual) =====
        max_player_lot = (
            max([o["lot"] for o in aggregate_orders_by_player(trades_a)] + [0.0]) +
            max([o["lot"] for o in aggregate_orders_by_player(trades_b)] + [0.0])
        )
        player_big_hit = (max_player_lot >= self.p.player_mult_vs_vol_ema * self.metrics.ema_vol_sec)

        # vol com hard e com decaimento exponencial do threshold por idade do evento
        t_evt = (self.evt["dur"] if self.evt else 0)
        hl = self.p.decay_half_life_frac * self.p.max_dur_s
        ema_vol_ref = (self.evt["ema_vol_start"] if self.evt else self.metrics.ema_vol_sec)
        vol_thr_now = exp_decay_mult(t_evt, self.p.start_vol_mult * ema_vol_ref, 1.0 * ema_vol_ref, hl)
        vol_spike_hit = (vol_comb >= vol_thr_now) or (vol_comb >= self.p.vol_1s_hard)

        # ===== RANGE (se anexar s) com decaimento exponencial do threshold =====
        def _safe_span(ev_min, ev_max, snap_min, snap_max):
            vals = []
            if ev_min  is not None: vals.append(ev_min)
            if ev_max  is not None: vals.append(ev_max)
            if snap_min is not None: vals.append(snap_min)
            if snap_max is not None: vals.append(snap_max)
            return (max(vals) - min(vals)) if len(vals) >= 2 else 0.0

        range_a_if = _safe_span(self.evt["pmin_a"] if self.evt else None,
                                self.evt["pmax_a"] if self.evt else None,
                                smry_a["pmin"], smry_a["pmax"])
        range_b_if = _safe_span(self.evt["pmin_b"] if self.evt else None,
                                self.evt["pmax_b"] if self.evt else None,
                                smry_b["pmin"], smry_b["pmax"])
        range_t_if = max(
            range_a_if / max(self.p.tick_a, 1e-9),
            range_b_if / max(self.p.tick_b, 1e-9)
        )

        range_baseline_ticks = max(ema_var5m_ticks, 1e-9)
        range_thr = exp_decay_mult(
            t_evt,
            self.p.range_mult_start * range_baseline_ticks,
            self.p.range_mult_end   * range_baseline_ticks,
            hl
        )
        range_cut = (range_t_if >= range_thr)

        # ===== TEMPO MÁXIMO =====
        end_by_time_pre = (self.evt is not None) and ((self.evt["dur"] + 1) > self.p.max_dur_s)

        # ===== ABRIR PRIMEIRO EVENTO? =====
        events_out = []
        if self.evt is None and (smry_a["n"] + smry_b["n"]) > 0:
            def _fallback(primary, *alts):
                for v in (primary, *alts):
                    if v is not None: return float(v)
                return None
            open_a = _fallback(smry_a["open"], close_a, self.last_close_a)
            open_b = _fallback(smry_b["open"], close_b, self.last_close_b)

            self.evt = {
                "start": s, "end": s, "dur": 1,
                "p0_a": open_a, "p0_b": open_b,
                "pmin_a": smry_a["pmin"] if smry_a["pmin"] is not None else open_a,
                "pmax_a": smry_a["pmax"] if smry_a["pmax"] is not None else open_a,
                "pmin_b": smry_b["pmin"] if smry_b["pmin"] is not None else open_b,
                "pmax_b": smry_b["pmax"] if smry_b["pmax"] is not None else open_b,
                "plast_a": close_a if close_a is not None else open_a,
                "plast_b": close_b if close_b is not None else open_b,
                "vol": vol_comb, "n_trades": smry_a["n"] + smry_b["n"],
                "ema_vol_start": self.metrics.ema_vol_sec,
                "ema_var5m_start_ticks": ema_var5m_ticks,
                "start_reason": "bootstrap",
            }
            self.evt["acc"] = event_acc_init()

            self.last_close_a, self.last_close_b = close_a, close_b
            self.speed_price_streak = 0
            self.rate_hi_streak = 0
            return events_out

        if self.evt is None:
            self.last_close_a, self.last_close_b = close_a, close_b
            return events_out

        # ===== DECISÃO DE FRONTEIRA =====
        cooldown_ok = (self.last_boundary_at is None) or ((s - self.last_boundary_at) >= self.p.boundary_cooldown_s)
        cause_flags = {
            "inversion": inv_hit,
            "player":    player_big_hit,
            "vol":       vol_spike_hit,
            "range":     range_cut,
            "speed":     speed_price_cut,
            "rate":      order_rate_cut,
            "time":      end_by_time_pre,
        }
        cause_order = ["inversion","player","vol","range","speed","rate","time"]
        cause = next((c for c in cause_order if cause_flags[c]), None)
        boundary_now = (cooldown_ok or end_by_time_pre) and (cause is not None)

        if boundary_now:
            # fecha anterior em s-1
            self.evt["end"] = s - 1
            self.evt["dur"] = self.evt["end"] - self.evt["start"] + 1
            self.evt["end_reason"] = "time_pre" if cause == "time" else cause

            # métricas finais do evento
            def _mv(p0, p1, tk):
                if p0 is None or p1 is None or tk <= 0: return 0.0
                return abs((p1 - p0)/tk)
            self.evt["d_ticks_max"] = max(
                _mv(self.evt["p0_a"], self.evt["plast_a"], self.p.tick_a),
                _mv(self.evt["p0_b"], self.evt["plast_b"], self.p.tick_b)
            )
            self.evt["range_ticks"] = max(
                (self.evt["pmax_a"] - self.evt["pmin_a"]) / max(self.p.tick_a,1e-9) if self.evt["pmin_a"] is not None else 0.0,
                (self.evt["pmax_b"] - self.evt["pmin_b"]) / max(self.p.tick_b,1e-9) if self.evt["pmin_b"] is not None else 0.0
            )
            self.evt["ticks_per_s"] = self.evt["d_ticks_max"] / max(self.evt["dur"], 1)

            # === enriquecer com blocos price/volume/trade/move/players/context ===
            self.evt = finalize_event_enrichment(
                self.evt, self.p, self.metrics, self.p.tick_a, self.p.tick_b, extrair_preco
            )
            events_out.append(self.evt)
            self.last_event_end = self.evt["end"]
            self.last_boundary_at = s

            # abre novo em s com flags do corte
            self.evt = {
                "start": s, "end": s, "dur": 1,
                "p0_a": smry_a["open"] if smry_a["open"] is not None else close_a,
                "p0_b": smry_b["open"] if smry_b["open"] is not None else close_b,
                "pmin_a": smry_a["pmin"] if smry_a["pmin"] is not None else close_a,
                "pmax_a": smry_a["pmax"] if smry_a["pmax"] is not None else close_a,
                "pmin_b": smry_b["pmin"] if smry_b["pmin"] is not None else close_b,
                "pmax_b": smry_b["pmax"] if smry_b["pmax"] is not None else close_b,
                "plast_a": close_a, "plast_b": close_b,
                "vol": vol_comb, "n_trades": smry_a["n"] + smry_b["n"],
                "ema_vol_start": self.metrics.ema_vol_sec,
                "ema_var5m_start_ticks": ema_var5m_ticks,
                "start_reason": cause or "boundary",
                "start_flags": cause_flags,
            }
            self.evt["acc"] = event_acc_init()
            # reseta streaks dependentes
            self.speed_price_streak = 0
            self.rate_hi_streak = 0
        else:
            # agrega s no evento atual
            self.evt["end"] = s
            self.evt["dur"] = self.evt["end"] - self.evt["start"] + 1
            if smry_a["pmin"] is not None:
                self.evt["pmin_a"] = min(self.evt["pmin_a"], smry_a["pmin"])
                self.evt["pmax_a"] = max(self.evt["pmax_a"], smry_a["pmax"])
                if close_a is not None: self.evt["plast_a"] = close_a
            if smry_b["pmin"] is not None:
                self.evt["pmin_b"] = min(self.evt["pmin_b"], smry_b["pmin"])
                self.evt["pmax_b"] = max(self.evt["pmax_b"], smry_b["pmax"])
                if close_b is not None: self.evt["plast_b"] = close_b
            self.evt["vol"]      += vol_comb
            self.evt["n_trades"] += (smry_a["n"] + smry_b["n"])

            # === acumula estatísticas detalhadas deste segundo ===
            # usa os mesmos 'starts_a' e 'starts_b' que você já calcula para o RATE
            event_accumulate_second(
                self.evt["acc"],
                trades_a, trades_b,
                smry_a, smry_b,
                close_a, close_b,
                self.last_close_a, self.last_close_b,
                self.p.tick_a, self.p.tick_b,
                starts_a, starts_b,
                self.p
            )

        # atualiza closes
        self.last_close_a, self.last_close_b = close_a, close_b
        return events_out

def print_event(ev, tick_a=0.5, tick_b=0.5, label_a="A", label_b="B"):
    import datetime as dt

    def ts(x): 
        return dt.datetime.fromtimestamp(int(x)).strftime("%H:%M:%S")

    def nz(x, d=None):
        return d if (x is None) else x

    # campos básicos
    flags = ev.get("start_flags", {})
    on = [k for k, v in flags.items() if v]
    mv  = float(ev.get("d_ticks_max", 0.0))
    rng = float(ev.get("range_ticks", 0.0))
    dur = int(ev.get("dur", 0))
    spd = float(ev.get("ticks_per_s", mv / max(1, dur)))  # fallback

    # preços por símbolo
    p0a = ev.get("p0_a"); pla = ev.get("plast_a")
    p0b = ev.get("p0_b"); plb = ev.get("plast_b")
    pmina = ev.get("pmin_a"); pmaxa = ev.get("pmax_a")
    pminb = ev.get("pmin_b"); pmaxb = ev.get("pmax_b")

    # deslocamento e range em ticks por símbolo (com fallback seguro)
    def ticks_move(p0, p1, tk):
        if p0 is None or p1 is None or tk <= 0: 
            return 0.0
        return (float(p1) - float(p0)) / float(tk)

    def ticks_range(pmin, pmax, tk):
        if pmin is None or pmax is None or tk <= 0:
            return 0.0
        return (float(pmax) - float(pmin)) / float(tk)

    mv_a_t = ticks_move(p0a, pla, tick_a)
    mv_b_t = ticks_move(p0b, plb, tick_b)
    rg_a_t = ticks_range(pmina, pmaxa, tick_a)
    rg_b_t = ticks_range(pminb, pmaxb, tick_b)

    # prints
    print(
        f"[EVENT] {ts(ev['start'])}→{ts(ev['end'])} dur={dur:2d}s "
        f"start={ev.get('start_reason','?'):>9} end={ev.get('end_reason','?'):>7} "
        f"flags={on} vol={int(ev.get('vol',0))} n={int(ev.get('n_trades',0))} "
        f"mv={mv:+.1f}t range={rng:.1f}t speed={spd:.2f}t/s | "
        f"{label_a}: p0={nz(p0a,'-')} pf={nz(pla,'-')} d={mv_a_t:+.1f}t R={rg_a_t:.1f}t | "
        f"{label_b}: p0={nz(p0b,'-')} pf={nz(plb,'-')} d={mv_b_t:+.1f}t R={rg_b_t:.1f}t"
    )


DATA_DIR  = Path(r"E:\Mercado BMF&BOVESPA\tryd\consolidados_npz")
DAY       = "20201013"
PAIR      = ("wdo", "dol")   # ou 
# PAIR = ("win","ind")

TICK_SIZE = {"wdo":0.5, "dol":0.5, "win":5.0, "ind":5.0}
def lot_multiplier(sym: str) -> float:
    return 5.0 if sym.lower() in ("dol","ind") else 1.0

def find_files_for_day(data_dir: Path, day: str, pair: tuple[str,str]) -> dict[str, Path]:
    pat = re.compile(rf"^({day})_({pair[0]}|{pair[1]})\.npz$", re.IGNORECASE)
    out = {}
    for p in data_dir.glob("*.npz"):
        m = pat.match(p.name)
        if m:
            out[m.group(2).lower()] = p
    return out

def load_t_tt(path: Path):
    d = np.load(path, allow_pickle=True)
    t = None; TT = None
    for k in d.files:
        kl = k.lower()
        if kl in ("t","time","datetime","timestamps","date_time"):
            t = d[k]
        if kl in ("tt","x","data"):
            TT = d[k]
    if t is None or TT is None:
        raise ValueError(f"Chaves não encontradas em {path.name}: {d.files}")
    return t, TT

def to_epoch_seconds(t_arr: np.ndarray) -> np.ndarray:
    if np.issubdtype(t_arr.dtype, np.datetime64):
        return t_arr.astype("datetime64[s]").astype(np.int64)
    t = t_arr.astype(np.int64)
    mx = int(t.max()) if t.size else 0
    if   mx > 10**17: return (t // 1_000_000_000).astype(np.int64)
    elif mx > 10**14: return (t // 1_000_000).astype(np.int64)
    elif mx > 10**11: return (t // 1_000).astype(np.int64)
    else:             return t

def group_by_second_preserving_order(t_sec: np.ndarray, TT: np.ndarray, sym: str) -> dict[int, list]:
    mult = lot_multiplier(sym)
    by_sec = {}
    for ts, row in zip(t_sec, TT):
        r = list(row)
        if len(r) > 1:
            r[1] = float(r[1]) * mult  # aplica peso de lote
        by_sec.setdefault(int(ts), []).append(r)
    return by_sec
    
realtime = False
if realtime:
    import numpy as np

    sym_a, sym_b = PAIR[0].lower(), PAIR[1].lower()
    tick_a, tick_b = TICK_SIZE[sym_a], TICK_SIZE[sym_b]

    # === 3) INSTANCIA O SEGMENTER COM PARAMS DO PAR ===
    seg = EventSegmenter(SegParams(
        tick_a=tick_a, tick_b=tick_b,
        lot_mult_a=lot_multiplier(sym_a),
        lot_mult_b=lot_multiplier(sym_b),
    ))
    seg.reset()

    # === Definição da struct compartilhada ===
    class MyPyStruct(ctypes.Structure):
        _fields_ = [
            ("bookdol", ctypes.c_float * (10 * 4)),
            ("bookwdo", ctypes.c_float * (10 * 4)),

            ("n_trades_dol", ctypes.c_int),
            ("trades_dol", ctypes.c_float * (2000 * 5)),

            ("n_trades_wdo", ctypes.c_int),
            ("trades_wdo", ctypes.c_float * (2000 * 5)),

            ("cot", ctypes.c_float * 6),
        ]

    sizeof_struct = ctypes.sizeof(MyPyStruct)

    # === Conecta ao mapeamento de memória e ao evento ===
    MAP_NAME = "MyPythonConection"
    EVENT_NAME = "WorkEvent"

    mm = mmap.mmap(-1, sizeof_struct, tagname=MAP_NAME)
    data_ptr = MyPyStruct.from_buffer(mm)

    # Abre o handle do evento criado pelo C++
    max_wait_sec = 10
    event_handle = None
    for _ in range(max_wait_sec * 10):  # tenta por até 10 segundos
        try:
            event_handle = win32event.OpenEvent(win32con.EVENT_MODIFY_STATE | win32con.SYNCHRONIZE,
                                                False, EVENT_NAME)
            break
        except pywintypes.error:
            time.sleep(0.1)

    if event_handle is None:
        print("Não foi possível abrir o evento compartilhado (WorkEvent) após aguardar.")
        exit(1)

    print("✅ Aguardando sinal de novo pacote...")

    bookdol_hist = []
    bookwdo_hist = []
    cot_hist = []
    trades_dol_hist = []
    trades_wdo_hist = []

    while True:
        # Aguarda o evento ser sinalizado (timeout opcional: INFINITE = espera indefinida)
        wait_result = win32event.WaitForSingleObject(event_handle, win32event.INFINITE)

        if wait_result == win32con.WAIT_OBJECT_0:
            # lê snapshot
            n_dol = int(data_ptr.n_trades_dol)
            n_wdo = int(data_ptr.n_trades_wdo)

            if n_dol > 0:
                arr_d = np.ctypeslib.as_array(data_ptr.trades_dol)[: n_dol * 5].reshape(-1, 5).astype(float)
            else:
                arr_d = np.empty((0,5), dtype=float)

            if n_wdo > 0:
                arr_w = np.ctypeslib.as_array(data_ptr.trades_wdo)[: n_wdo * 5].reshape(-1, 5).astype(float)
            else:
                arr_w = np.empty((0,5), dtype=float)

            # print(f"\n📦 Novo pacote: {n_dol} trades DOL, {n_wdo} trades WDO")
            # print(f"    Últimos trades DOL: {arr_d}")
            # print(f"    Últimos trades WDO: {arr_w}")

            # depois de ler n_dol/n_wdo e montar arr_d, arr_w (shape N×5)
            sec_now = int(time.time())  # se tiver timestamp do produtor, use ele aqui!

            # passa os trades CRUS; o segmenter aplica os multiplicadores de lote
            events_done = seg.step(sec_now,
                                arr_w.tolist(),   # WDO
                                arr_d.tolist())   # DOL

            # imprime qualquer evento finalizado neste passo (normalmente 0 ou 1)
            for ev in events_done:
                print_event(ev)
        
        win32event.ResetEvent(event_handle)

else:
    # === 2) PREPARA O REPLAY DO DIA USANDO DATA_DIR ===
    files = find_files_for_day(DATA_DIR, DAY, PAIR)
    sym_a, sym_b = PAIR[0].lower(), PAIR[1].lower()
    t_a, TT_a = load_t_tt(files[sym_a]); sec_a = group_by_second_preserving_order(to_epoch_seconds(t_a), TT_a, sym_a)
    t_b, TT_b = load_t_tt(files[sym_b]); sec_b = group_by_second_preserving_order(to_epoch_seconds(t_b), TT_b, sym_b)
    secs = sorted(set(sec_a.keys()) | set(sec_b.keys()))
    tick_a, tick_b = TICK_SIZE[sym_a], TICK_SIZE[sym_b]

    # === 3) INSTANCIA O SEGMENTER COM PARAMS DO PAR ===
    seg = EventSegmenter(SegParams(
        tick_a=tick_a, tick_b=tick_b,
        lot_mult_a=lot_multiplier(sym_a),
        lot_mult_b=lot_multiplier(sym_b),
    ))
    seg.reset()

    eventos = []
    for s in secs:
        trades_a = sec_a.get(s, [])
        trades_b = sec_b.get(s, [])
        eventos.extend(seg.step(s, trades_a, trades_b))

    # fecha o último evento no fim do dia (opcional)
    if seg.evt is not None:
        seg.evt["end_reason"] = "eod"
        eventos.append(seg.evt)

    print(f"Dia {DAY} {PAIR[0].upper()}+{PAIR[1].upper()} → {len(eventos)} eventos (segmentação contínua)")


    # ================== ESTATÍSTICAS DOS EVENTOS ==================

    if not eventos:
        print("Sem eventos.")
    else:
        # 1) Motivo principal (start_reason)
        reason_cnt = Counter(ev.get("start_reason","") for ev in eventos)
        total = len(eventos)
        print("\n== Motivo principal (start_reason) ==")
        for k, c in reason_cnt.most_common():
            pct = 100.0 * c / total
            print(f"{k or 'unknown':>9}: {c:5d}  ({pct:5.1f}%)")

        # 2) Flags individuais (vol/player/inversion) — contam sobreposições
        flag_cnt = Counter()
        for ev in eventos:
            flags = ev.get("start_flags", {})
            for k, v in flags.items():
                if v: flag_cnt[k] += 1

        print("\n== Flags ON (contam sobreposições) ==")
        for k, c in flag_cnt.most_common():
            pct = 100.0 * c / total
            print(f"{k:>9}: {c:5d}  ({pct:5.1f}%)")

        # 3) Sobreposições (vol+player, vol+inv, player+inv, os três, etc.)
        overlap = Counter()
        for ev in eventos:
            on = tuple(sorted([k for k,v in ev.get("start_flags", {}).items() if v]))
            overlap[on] += 1

        print("\n== Combinações de gatilhos (overlap) ==")
        for combo, c in overlap.most_common():
            name = "+".join(combo) if combo else "none"
            pct = 100.0 * c / total
            print(f"{name:>20}: {c:5d}  ({pct:5.1f}%)")

        # 4) Eventos por hora (pra ver onde está “excesso”)
        per_hour = Counter(datetime.fromtimestamp(int(ev["start"])).hour for ev in eventos)
        print("\n== Eventos por hora ==")
        for h in sorted(per_hour):
            c = per_hour[h]; pct = 100.0 * c / total
            print(f"{h:02d}h: {c:5d}  ({pct:5.1f}%)")

    # --- helpers para extrair vetores ---
    def _event_stats_arrays(eventos_list):
        if not eventos_list:
            return [], [], [], [], []
        dur    = [int(ev.get("dur", 0)) for ev in eventos_list]
        vol    = [float(ev.get("vol", 0.0)) for ev in eventos_list]
        rng_t  = [float(ev.get("range_ticks", 0.0)) for ev in eventos_list]
        # velocidade média em ticks/s usando d_ticks_max se houver; fallback para range/dur
        spd    = []
        for ev in eventos_list:
            dmax = ev.get("d_ticks_max", None)
            if dmax is None:
                # fallback leve: range/dur
                r = float(ev.get("range_ticks", 0.0))
                d = max(1, int(ev.get("dur", 1)))
                spd.append(r / d)
            else:
                d = max(1, int(ev.get("dur", 1)))
                spd.append(float(dmax)/d)
        reason = []
        for ev in eventos_list:
            r = ev.get("end_reason", "")
            # normaliza "time_pre" -> "time"
            if r == "time_pre":
                r = "time"
            reason.append(r)
        return dur, vol, rng_t, spd, reason

    dur, vol, rng_t, spd, reason = _event_stats_arrays(eventos)

    if len(dur) == 0:
        print("Sem eventos para estatística.")
    else:
        # --- mapa de cores por motivo (dinâmico com defaults) ---
        base_color_map = {
            "vol": "C0",
            "player": "C1",
            "inversion": "C2",
            "speed": "C3",
            "range": "C4",
            "calm": "C5",
            "time": "C6",
            "unknown": "C7",
            "": "C7",
        }
        uniq = list(Counter(reason).keys())
        color_map = {r: base_color_map.get(r, f"C{(i % 8)}") for i, r in enumerate(uniq)}
        colors = [color_map.get(r, "C7") for r in reason]

        # --- hist de duração (s) ---
        max_dur_param = None
        try:
            # se você tiver 'seg' ou 'seg.p.max_dur_s' no escopo:
            max_dur_param = int(seg.p.max_dur_s)  # opcional
        except Exception:
            pass
        max_dur_val = max_dur_param if max_dur_param and max_dur_param > 1 else max(dur) if dur else 30
        bins_dur = range(1, int(max_dur_val) + 1)

        plt.figure(figsize=(10,4))
        plt.hist(dur, bins=bins_dur, edgecolor="none")
        plt.title("Eventos — Duração (s)")
        plt.xlabel("segundos"); plt.ylabel("contagem")
        plt.tight_layout(); plt.show()

        # --- hist de volume (log-bins) ---
        vmin = max(1.0, min(vol))
        vmax = max(vol)
        if vmin >= vmax:  # evita logspace inválido
            vmax = vmin + 1.0
        nb = 40
        bins = np.logspace(math.log10(vmin), math.log10(vmax + 1e-9), nb)
        plt.figure(figsize=(10,4))
        plt.hist(vol, bins=bins)
        plt.xscale("log")
        plt.title("Eventos — Volume ponderado (log)")
        plt.xlabel("volume"); plt.ylabel("contagem")
        plt.tight_layout(); plt.show()

        # --- scatter duração x range (ticks) ---
        plt.figure(figsize=(10,4))
        plt.scatter(dur, rng_t, s=8, c=colors, alpha=0.7)
        plt.title("Eventos — Duração vs Range (ticks)")
        plt.xlabel("duração (s)"); plt.ylabel("range (ticks)")
        # legenda por motivo com contagens
        counts = Counter(reason)
        handles = []
        labels  = []
        for r, c in counts.most_common():
            handles.append(plt.Line2D([0],[0], marker='o', linestyle='', color=color_map.get(r, "C7")))
            labels.append(f"{r or 'unknown'}: {c}")
        if handles:
            plt.legend(handles, labels, loc="upper right", fontsize=8, frameon=False, ncol=2)
        plt.tight_layout(); plt.show()

        # --- scatter duração x velocidade média (ticks/s) ---
        plt.figure(figsize=(10,4))
        plt.scatter(dur, spd, s=8, c=colors, alpha=0.7)
        plt.title("Eventos — Duração vs Velocidade média (ticks/s)")
        plt.xlabel("duração (s)"); plt.ylabel("ticks/s")
        if handles:
            plt.legend(handles, labels, loc="upper right", fontsize=8, frameon=False, ncol=2)
        plt.tight_layout(); plt.show()

    # ================== PLOT TICK-A-TICK (tempo real no eixo X) ==================

    def _build_event_sec_map(eventos_list):
        """sec -> id_evento; fora de evento = -1."""
        evmap = {}
        for k, ev in enumerate(eventos_list):
            for s in range(int(ev["start"]), int(ev["end"]) + 1):
                evmap[s] = k
        return evmap

    def _tick_series_time(sec_map, evmap):
        """
        Constrói série tick-a-tick:
        xs_time: datetimes (espalhando trades dentro do segundo)
        ys: preços
        ev_ids: id do evento (ou -1)
        """
        xs_time, ys, ev_ids = [], [], []
        for s in sorted(sec_map.keys()):
            trades = sec_map[s]
            n = len(trades)
            if n == 0: 
                continue
            # espalha no intervalo [s, s+1) para preservar ordem dentro do segundo
            for k, tr in enumerate(trades):
                frac = (k + 1) / (n + 1)    # 0<frac<1
                t_real = s + frac
                base_utc = datetime.fromtimestamp(t_real, tz=timezone.utc)
                xs_time.append(base_utc)  # UTC -> UTC-3 (São Paulo)
                ys.append(float(tr[0]))
                ev_ids.append(evmap.get(s, -1))
        return xs_time, ys, ev_ids

    def _plot_segments_time(xs, ys, ev_ids, title="Tick-a-tick — cores por evento (tempo real)"):
        if not xs:
            print("Sem dados para plot.")
            return
        # quebra em segmentos de mesmo evento
        segments = []
        start = 0
        last = ev_ids[0]
        for i in range(1, len(xs)):
            if ev_ids[i] != last:
                segments.append((xs[start:i], ys[start:i], last))
                start = i
                last = ev_ids[i]
        segments.append((xs[start:], ys[start:], last))

        fig, ax = plt.subplots(figsize=(14,5))
        for sx, sy, _eid in segments:
            ax.plot(sx, sy, linewidth=0.8)  # cores automáticas

        ax.set_title(title)
        ax.set_xlabel("tempo")
        ax.set_ylabel("preço")
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.show()

    # escolher qual símbolo usar como “preço” (WDO se existir, senão primeiro do par)
    if sym_a == "wdo":
        sec_price = sec_a
    elif sym_b == "wdo":
        sec_price = sec_b
    else:
        sec_price = sec_a  # fallback

    evmap = _build_event_sec_map(eventos)
    xs_time, ys_price, ev_ids = _tick_series_time(sec_price, evmap)
    _plot_segments_time(xs_time, ys_price, ev_ids,
                        title=f"Tick-a-tick {('WDO' if (sym_a=='wdo' or sym_b=='wdo') else sym_a.upper())} — cores por evento (tempo real)")

    breakpoint()

    # ---- helpers robustos
    def _as_float(x):
        try:
            return float(x)
        except Exception:
            return np.nan

    def _flatten_dict(d, prefix):
        out = {}
        if not isinstance(d, dict): 
            return out
        for k, v in d.items():
            key = f"{prefix}_{k}"
            if isinstance(v, dict):
                out.update(_flatten_dict(v, key))
            else:
                out[key] = v
        return out

    def events_to_df(eventos):
        rows = []
        for ev in (eventos or []):
            base = {
                "start_ts": ev.get("start"),
                "end_ts": ev.get("end"),
                "dur_s": ev.get("dur"),
                "start_reason": ev.get("start_reason"),
                "end_reason": ev.get("end_reason"),
                "vol": _as_float(ev.get("vol")),
                "n_trades": int(ev.get("n_trades", 0) or 0),
                "range_ticks": _as_float(ev.get("range_ticks")),
                "d_ticks_max": _as_float(ev.get("d_ticks_max")),
                "ticks_per_s": _as_float(ev.get("ticks_per_s")),
                # preços chaves (A=WDO, B=DOL pelo seu setup)
                "p0_a": _as_float(ev.get("p0_a")),
                "p0_b": _as_float(ev.get("p0_b")),
                "plast_a": _as_float(ev.get("plast_a")),
                "plast_b": _as_float(ev.get("plast_b")),
                "pmin_a": _as_float(ev.get("pmin_a")),
                "pmax_a": _as_float(ev.get("pmax_a")),
                "pmin_b": _as_float(ev.get("pmin_b")),
                "pmax_b": _as_float(ev.get("pmax_b")),
            }
            # flags (começam com 0/1)
            flags = ev.get("start_flags", {}) or {}
            for k, v in flags.items():
                base[f"flag_{k}"] = 1 if v else 0

            # blocos enriquecidos (se existirem)
            for blk in ("price_info","volume_info","trade_info","movement_info","player_info","context_info"):
                base.update(_flatten_dict(ev.get(blk, {}), blk))

            # tempo legível + hora
            try:
                base["start_time"] = datetime.fromtimestamp(int(base["start_ts"])) if base["start_ts"] else None
                base["end_time"]   = datetime.fromtimestamp(int(base["end_ts"]))   if base["end_ts"] else None
                base["hour"]       = base["start_time"].hour if base.get("start_time") else None
            except Exception:
                base["start_time"] = base["end_time"] = None
                base["hour"] = None

            rows.append(base)

        df = pd.DataFrame(rows)
        # tipos numéricos
        for c in ["dur_s","vol","n_trades","range_ticks","d_ticks_max","ticks_per_s",
                "p0_a","plast_a","pmin_a","pmax_a","p0_b","plast_b","pmin_b","pmax_b"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        if "hour" in df.columns:
            df["hour"] = pd.to_numeric(df["hour"], errors="coerce")
        return df
    
    def plot_event_distributions(df):
        if df.empty:
            print("Sem eventos no DataFrame.")
            return

        # --- Duração
        plt.figure(figsize=(10,4))
        vals = df["dur_s"].dropna()
        bins = np.arange(1, max(60, int(vals.max())+2))
        plt.hist(vals, bins=bins, edgecolor="none")
        plt.title("Eventos — Duração (s)")
        plt.xlabel("segundos"); plt.ylabel("contagem")
        plt.tight_layout(); plt.show()

        # --- Volume (log)
        if "vol" in df and df["vol"].notna().any():
            v = df["vol"].dropna()
            v = v[v > 0]
            if len(v) > 0:
                nb = 40
                vmin, vmax = max(1.0, v.min()), v.max()
                bins = np.logspace(math.log10(vmin), math.log10(vmax+1e-9), nb)
                plt.figure(figsize=(10,4))
                plt.hist(v, bins=bins)
                plt.xscale("log")
                plt.title("Eventos — Volume ponderado (log)")
                plt.xlabel("volume"); plt.ylabel("contagem")
                plt.tight_layout(); plt.show()

        # --- Range em ticks
        if "range_ticks" in df:
            plt.figure(figsize=(10,4))
            plt.hist(df["range_ticks"].dropna(), bins=50)
            plt.title("Eventos — Range (ticks)")
            plt.xlabel("ticks"); plt.ylabel("contagem")
            plt.tight_layout(); plt.show()

        # --- Speed (ticks/s)
        if "ticks_per_s" in df:
            plt.figure(figsize=(10,4))
            plt.hist(df["ticks_per_s"].dropna(), bins=50)
            plt.title("Eventos — Velocidade média (ticks/s)")
            plt.xlabel("ticks/s"); plt.ylabel("contagem")
            plt.tight_layout(); plt.show()

        # --- ECDFs úteis (dur, vol, range)
        def plot_ecdf(series, title, xlabel):
            x = np.sort(series.dropna().values)
            if len(x)==0: return
            y = np.arange(1, len(x)+1) / len(x)
            plt.figure(figsize=(8,4))
            plt.plot(x, y, drawstyle="steps-post")
            plt.title(title); plt.xlabel(xlabel); plt.ylabel("ECDF")
            plt.tight_layout(); plt.show()

        plot_ecdf(df["dur_s"], "ECDF — Duração", "segundos")
        if "vol" in df: plot_ecdf(np.log10(df["vol"].replace(0,np.nan)), "ECDF — log10(Volume)", "log10(volume)")
        if "range_ticks" in df: plot_ecdf(df["range_ticks"], "ECDF — Range (ticks)", "ticks")

    def plot_relationships(df):
        if df.empty:
            return

        # --- Hexbin: Volume vs Range
        if {"vol","range_ticks"} <= set(df.columns):
            plt.figure(figsize=(6,5))
            plt.hexbin(df["vol"], df["range_ticks"], gridsize=40, mincnt=1)
            plt.xscale("log")
            plt.title("Hexbin — Volume vs Range (ticks)")
            plt.xlabel("volume (log)"); plt.ylabel("range (ticks)")
            plt.tight_layout(); plt.show()

        # --- Hexbin: Volume vs ticks_per_s
        if {"vol","ticks_per_s"} <= set(df.columns):
            plt.figure(figsize=(6,5))
            plt.hexbin(df["vol"], df["ticks_per_s"], gridsize=40, mincnt=1)
            plt.xscale("log")
            plt.title("Hexbin — Volume vs Velocidade (ticks/s)")
            plt.xlabel("volume (log)"); plt.ylabel("ticks/s")
            plt.tight_layout(); plt.show()

        # --- Boxplot de métricas por motivo de corte
        if "end_reason" in df.columns:
            for metric in ["dur_s","vol","range_ticks","ticks_per_s","n_trades"]:
                if metric in df.columns:
                    sub = df[["end_reason", metric]].dropna()
                    if sub.empty: continue
                    groups = [g[metric].values for _, g in sub.groupby("end_reason")]
                    labels = [str(k) for k, _ in sub.groupby("end_reason")]
                    plt.figure(figsize=(10,4))
                    plt.boxplot(groups, labels=labels, showfliers=False)
                    plt.title(f"Boxplot — {metric} por end_reason")
                    plt.xlabel("end_reason"); plt.ylabel(metric)
                    plt.tight_layout(); plt.show()

    def plot_counts_by_hour_and_reason(df):
        if df.empty or "hour" not in df.columns: 
            return
        # barras por hora (total)
        per_hour = df.groupby("hour").size().reindex(range(0,24), fill_value=0)
        plt.figure(figsize=(10,3))
        plt.bar(per_hour.index, per_hour.values)
        plt.title("Eventos por hora"); plt.xlabel("hora"); plt.ylabel("contagem")
        plt.tight_layout(); plt.show()

        # stacked por end_reason
        if "end_reason" in df.columns:
            pivot = df.pivot_table(index="hour", columns="end_reason", values="start_ts",
                                aggfunc="count").fillna(0).reindex(range(0,24), fill_value=0)
            pivot.plot(kind="bar", stacked=True, figsize=(12,4))
            plt.title("Eventos por hora (stacked por end_reason)")
            plt.xlabel("hora"); plt.ylabel("contagem")
            plt.tight_layout(); plt.show()

        # barras simples por end_reason (frequência total)
        if "end_reason" in df.columns:
            cnt = df["end_reason"].value_counts()
            plt.figure(figsize=(8,3))
            plt.bar(cnt.index.astype(str), cnt.values)
            plt.title("Eventos por end_reason")
            plt.xlabel("end_reason"); plt.ylabel("contagem")
            plt.tight_layout(); plt.show()

    def print_top_outliers(df, top=10):
        def _show(title, ser):
            print(f"\n== Top {top} por {title} ==")
            if ser.empty:
                print("(vazio)")
                return
            idx = ser.sort_values(ascending=False).head(top).index
            cols = ["start_time","end_time","dur_s","end_reason","vol","range_ticks","ticks_per_s","n_trades",
                    "p0_a","plast_a","pmin_a","pmax_a"]
            cols = [c for c in cols if c in df.columns]
            print(df.loc[idx, cols].to_string(index=False))

        if "range_ticks" in df: _show("range_ticks", df["range_ticks"].dropna())
        if "vol" in df: _show("volume", df["vol"].dropna())
        if "ticks_per_s" in df: _show("ticks_per_s", df["ticks_per_s"].dropna())
        if "n_trades" in df: _show("n_trades", df["n_trades"].dropna())

    # df = events_to_df(eventos)

    # plot_event_distributions(df)
    # plot_relationships(df)
    # plot_counts_by_hour_and_reason(df)
    # print_top_outliers(df, top=15)

    # ---- util: expande colunas que são listas (ex.: context_info_ctx_features)
    def expand_list_columns(df, candidate_prefixes=("context_info_ctx_features",)):
        out = df.copy()
        for col in out.columns:
            if any(col.startswith(p) for p in candidate_prefixes):
                # se tiver pelo menos um list/tuple, expande
                is_listy = out[col].apply(lambda v: isinstance(v, (list, tuple, np.ndarray)))
                if is_listy.any():
                    max_len = int(out[col][is_listy].map(len).max())
                    for i in range(max_len):
                        out[f"{col}_{i}"] = out[col].apply(lambda v: (v[i] if isinstance(v,(list,tuple,np.ndarray)) and len(v)>i else np.nan))
                    out.drop(columns=[col], inplace=True)
        return out

    # ---- 1) Seleção de features relevantes (exclui preços absolutos)
    def build_feature_df(df):
        df2 = expand_list_columns(df)

        # Prefixos válidos (mantemos métricas derivadas, sem preço bruto)
        keep_prefixes = [
            "volume_info_", "trade_info_", "movement_info_", "player_info_", "context_info_"
        ]
        # features “base” úteis para semântica (sem preço absoluto)
        base_feats = [
            "dur_s","vol","n_trades","range_ticks","d_ticks_max","ticks_per_s",
            "flag_inversion","flag_player","flag_vol","flag_range","flag_speed","flag_rate"
        ]

        cols = []
        for c in base_feats:
            if c in df2.columns: cols.append(c)
        for c in df2.columns:
            if any(c.startswith(p) for p in keep_prefixes):
                # evita trazer preço absoluto se existir algo do tipo (por segurança)
                if not any(x in c for x in ["price_raw","price_abs","p0","plast","pmin","pmax"]):
                    cols.append(c)
        # remove duplicatas preservando ordem
        cols = list(dict.fromkeys(cols))
        feat_df = df2[cols].copy()

        # garante numérico
        for c in feat_df.columns:
            feat_df[c] = pd.to_numeric(feat_df[c], errors="coerce")
        return feat_df

    # ---- 2) Heurística de normalização robusta por coluna
    def _minmax_clip_norm(x, p_lo=1.0, p_hi=99.0, log1p=False):
        s = x.astype(float).copy()
        if log1p:
            s = np.log1p(np.clip(s, a_min=0, a_max=None))
        lo, hi = np.nanpercentile(s.dropna(), [p_lo, p_hi]) if s.notna().any() else (0.0, 1.0)
        if hi <= lo: hi = lo + 1e-9
        s = s.clip(lo, hi)
        s = (s - lo) / (hi - lo)  # 0..1
        return s

    def normalize_events_to_vectors(df, range_mode="0-1"):
        """
        range_mode: "0-1" ou "-1-1"
        Retorna: X (np.ndarray), feature_names (list), scaler_meta (dict com percentis/flags)
        """
        feat_df = build_feature_df(df)

        # Binárias (flags) — mantemos como estão
        bin_feats = [c for c in feat_df.columns if c.startswith("flag_")]

        # Candidatas a log (caudas longas)
        log_like = [c for c in feat_df.columns if any(x in c.lower() for x in [
            "vol","volume","lot","rate","count","n_trades","absorption","intensity"
        ])]

        # Para ticks/variações, mantemos linear mas com clipping robusto
        linear_like = [c for c in feat_df.columns
                    if c not in bin_feats and c not in log_like]

        scaler_meta = {"log_like": log_like, "linear_like": linear_like, "bin_feats": bin_feats}

        Z = pd.DataFrame(index=feat_df.index)
        for c in linear_like:
            Z[c] = _minmax_clip_norm(feat_df[c], p_lo=1, p_hi=99, log1p=False)
        for c in log_like:
            Z[c] = _minmax_clip_norm(feat_df[c], p_lo=1, p_hi=99, log1p=True)
        for c in bin_feats:
            Z[c] = feat_df[c].fillna(0.0).clip(0,1)

        # ordem consistente
        feature_names = list(Z.columns)
        X = Z[feature_names].values.astype("float32")

        if range_mode == "-1-1":
            X = X * 2.0 - 1.0

        return X, feature_names, scaler_meta

    def _tod_bucket(h):
        # buckets de horário (ajuste se quiser outra grade)
        if h is None: return "UNK"
        if 9 <= h < 10:  return "OPEN"
        if 10 <= h < 13: return "MID"
        if 13 <= h < 16: return "PM"
        return "OFF"

    def _regime(x, lo, hi, labels=("LOW","MID","HIGH")):
        if x is None or (isinstance(x,float) and math.isnan(x)): return "UNK"
        if x < lo:  return labels[0]
        if x > hi:  return labels[2]
        return labels[1]

    def _sign_label(delta):
        if delta is None or (isinstance(delta,float) and math.isnan(delta)): return "FLAT"
        return "UP" if delta > 0 else ("DOWN" if delta < 0 else "FLAT")

    def make_event_window_line(ev, asset="WDO", alt_asset="DOL"):
        """Retorna SOMENTE a linha compacta no padrão 'E | key:val | ...'."""
        def ts(x):
            try: return datetime.fromtimestamp(int(x)).strftime("%H:%M:%S")
            except: return "00:00:00"
        def num(x, nd=1):
            try: 
                v = float(x)
                return f"{v:.{nd}f}"
            except:
                return "NA"
        def geti(k, d=0): 
            v = ev.get(k, d); 
            try: return int(v)
            except: return d

        # básicos
        start_hhmmss = ts(ev.get("start"))
        dur  = geti("dur", 0)
        rngt = num(ev.get("range_ticks", 0.0), 1)
        dmax = num(ev.get("d_ticks_max", 0.0), 1)
        spd  = num(ev.get("ticks_per_s", 0.0), 2)
        vol  = geti("vol", 0)
        ntr  = geti("n_trades", 0)
        endr = (ev.get("end_reason") or "?").upper()

        # volumes lado (se não existirem, volta 0)
        vol_buy_a  = ev.get("volume_info_buy_a", 0.0) or 0.0
        vol_sell_a = ev.get("volume_info_sell_a", 0.0) or 0.0
        vol_buy_b  = ev.get("volume_info_buy_b", 0.0) or 0.0
        vol_sell_b = ev.get("volume_info_sell_b", 0.0) or 0.0
        vol_buy    = int(vol_buy_a + vol_buy_b)
        vol_sell   = int(vol_sell_a + vol_sell_b)

        # taxa de ordens/seg (se não existir, usa n_trades/dur)
        evt_rate = (ntr / max(dur, 1.0))
        rate_ctx = ev.get("context_info_ema_order_rate", None)
        rate_txt = num(evt_rate, 2)
        rate_rel = num(evt_rate / rate_ctx, 2) if rate_ctx and rate_ctx > 0 else "NA"

        # maior lote de player (se tiver salvo na sua etapa de enrich; se não, NA)
        big_lot = ev.get("volume_info_max_group_lot", None)
        big_lot_txt = str(int(big_lot)) if isinstance(big_lot,(int,float)) else "NA"

        # preços (podem vir None; só texto)
        p0a = ev.get("p0_a", None)
        p1a = ev.get("plast_a", None)
        p0t = num(p0a, 1) if p0a is not None else "NA"
        p1t = num(p1a, 1) if p1a is not None else "NA"

        # flags
        f_inv   = 1 if ev.get("flag_inversion",   ev.get("start_flags",{}).get("inversion", False)) else 0
        f_play  = 1 if ev.get("flag_player",      ev.get("start_flags",{}).get("player",    False)) else 0
        f_vol   = 1 if ev.get("flag_vol",         ev.get("start_flags",{}).get("vol",       False)) else 0
        f_range = 1 if ev.get("flag_range",       ev.get("start_flags",{}).get("range",     False)) else 0
        f_speed = 1 if ev.get("flag_speed",       ev.get("start_flags",{}).get("speed",     False)) else 0
        f_rate  = 1 if ev.get("flag_rate",        ev.get("start_flags",{}).get("rate",      False)) else 0

        # compacta
        line = (
            f"E | T:{start_hhmmss} | DUR:{dur}s | RNG:{rngt}t | DMAX:{dmax}t | SPD:{spd}t/s | "
            f"VOL:{vol} | NTR:{ntr} | BUY:{vol_buy} | SELL:{vol_sell} | BIGLOT:{big_lot_txt} | "
            f"RATE:{rate_txt}/s | RATE_REL:{rate_rel} | "
            f"INV:{f_inv} | PLAYER:{f_play} | VOLTR:{f_vol} | RANGETR:{f_range} | SPEEDTR:{f_speed} | RATETR:{f_rate} | "
            f"END:{endr} | P0:{p0t} | P1:{p1t}"
        )
        return line

    def make_all_event_window_lines(eventos, asset="WDO", alt_asset="DOL"):
        return [make_event_window_line(ev, asset, alt_asset) for ev in eventos]

    # 1) Constrói DataFrame dos eventos (se ainda não tiver)
    df = events_to_df(eventos)

    # 2) Vetores normalizados
    X01, feat_names, scaler = normalize_events_to_vectors(df, range_mode="0-1")
    Xm1, _, _ = normalize_events_to_vectors(df, range_mode="-1-1")

    print(f"Feature vector shape: {X01.shape}")
    print(f"{len(feat_names)} features:", feat_names[:12], "..." if len(feat_names)>12 else "")

    # 3) Textos
    texts = make_all_event_window_lines(eventos, asset="WDO", alt_asset="DOL")

    # 4) Amostras aleatórias (mesmos eventos para vetor e texto)
    # rng = np.random.default_rng(42)
    # k = min(5, len(df))
    # idx = rng.choice(len(df), size=k, replace=False) if len(df)>0 else []

    # for i in idx:
    #     print("\n=== SAMPLE EVENT IDX", int(i), "===")
    #     # vetor (mostra primeiros 12 dims pra caber)
    #     vec01 = X01[i]
    #     vecm1 = Xm1[i]
    #     print("Vector 0..1 (first 12):", np.round(vec01[:12], 3).tolist(), "...")
    #     print("-1..1 (first 12):", np.round(vecm1[:12], 3).tolist(), "...")
    #     print("Text:\n", texts[i])


    # --- (reaproveita expand/build do seu passo anterior, mas ajusto levemente os nomes) ---
    def expand_list_columns(df, candidate_prefixes=("context_info_ctx_features",)):
        out = df.copy()
        for col in out.columns:
            if any(col.startswith(p) for p in candidate_prefixes):
                is_listy = out[col].apply(lambda v: isinstance(v, (list, tuple, np.ndarray)))
                if is_listy.any():
                    max_len = int(out[col][is_listy].map(len).max())
                    for i in range(max_len):
                        out[f"{col}_{i}"] = out[col].apply(lambda v: (v[i] if isinstance(v,(list,tuple,np.ndarray)) and len(v)>i else np.nan))
                    out.drop(columns=[col], inplace=True)
        return out

    def build_feature_df(df):
        df2 = expand_list_columns(df)

        keep_prefixes = ["volume_info_", "trade_info_", "movement_info_", "player_info_", "context_info_"]
        base_feats = ["dur_s","vol","n_trades","range_ticks","d_ticks_max","ticks_per_s",
                    "flag_inversion","flag_player","flag_vol","flag_range","flag_speed","flag_rate"]

        cols = []
        for c in base_feats:
            if c in df2.columns: cols.append(c)
        for c in df2.columns:
            if any(c.startswith(p) for p in keep_prefixes):
                if not any(x in c for x in ["price_raw","price_abs","p0","plast","pmin","pmax"]):
                    cols.append(c)
        cols = list(dict.fromkeys(cols))
        feat_df = df2[cols].copy()
        for c in feat_df.columns:
            feat_df[c] = pd.to_numeric(feat_df[c], errors="coerce")
        return feat_df

    def guess_column_roles(feat_df):
        bin_feats = [c for c in feat_df.columns if c.startswith("flag_")]
        # volumetria/contagens/“cauda longa”
        log_like = [c for c in feat_df.columns if any(x in c.lower() for x in
                        ["vol","volume","lot","rate","count","n_trades","absorption","intensity"])]
        linear_like = [c for c in feat_df.columns if c not in bin_feats and c not in log_like]
        return bin_feats, log_like, linear_like

    def fit_event_vector_scaler(df, p_lo=1.0, p_hi=99.0):
        feat_df = build_feature_df(df)
        bin_feats, log_like, linear_like = guess_column_roles(feat_df)

        params = {}
        order = list(feat_df.columns)
        for c in order:
            s = feat_df[c].astype(float)
            if c in bin_feats:
                params[c] = {"type":"bin"}
                continue
            if c in log_like:
                s = np.log1p(np.clip(s, a_min=0, a_max=None))
                params[c] = {"type":"cont", "log1p":True}
            else:
                params[c] = {"type":"cont", "log1p":False}
            s_ = s.dropna()
            if s_.empty:
                lo, hi = 0.0, 1.0
            else:
                lo, hi = np.nanpercentile(s_, [p_lo, p_hi])
                if hi <= lo: hi = lo + 1e-9
            params[c]["lo"] = float(lo)
            params[c]["hi"] = float(hi)

        scaler = {"features": order, "params": params, "p_lo": p_lo, "p_hi": p_hi}
        return scaler

    def transform_event_vectors(df, scaler, range_mode="0-1"):
        feat_df = build_feature_df(df)
        # garante colunas e ordem
        for c in scaler["features"]:
            if c not in feat_df.columns:
                feat_df[c] = np.nan

        Z = []
        for c in scaler["features"]:
            p = scaler["params"][c]
            x = feat_df[c].astype(float)
            if p["type"] == "bin":
                z = x.fillna(0.0).clip(0,1)
            else:
                if p.get("log1p", False):
                    x = np.log1p(np.clip(x, a_min=0, a_max=None))
                lo, hi = p["lo"], p["hi"]
                x = x.clip(lo, hi)
                z = (x - lo) / (hi - lo)
            Z.append(z.values)
        X = np.vstack(Z).T.astype("float32")

        if range_mode == "-1-1":
            X = X * 2.0 - 1.0

        return X, scaler["features"]


    def plot_vector_hists(X, names, per_page=24, bins=40, title="Feature dists (normalized)"):
        n = X.shape[1]
        pages = int(np.ceil(n / per_page))
        for pg in range(pages):
            L = pg*per_page
            R = min((pg+1)*per_page, n)
            k = R - L
            cols = 4
            rows = int(np.ceil(k/cols))
            plt.figure(figsize=(4*cols, 2.6*rows))
            for i, j in enumerate(range(L, R)):
                ax = plt.subplot(rows, cols, i+1)
                ax.hist(X[:, j], bins=bins, edgecolor="none")
                ax.set_title(names[j], fontsize=9)
                ax.set_xlim(0, 1)  # pois já está 0..1 (ou mapeado p/ 0..1 antes de -1..1)
            plt.suptitle(title + f" — page {pg+1}/{pages}")
            plt.tight_layout()
            plt.show()

    def plot_corr_heatmap(X, names, title="Correlation (Pearson)"):
        # corr simples
        Xc = np.nan_to_num(X, nan=0.0)
        C = np.corrcoef(Xc, rowvar=False)
        plt.figure(figsize=(8,6))
        im = plt.imshow(C, vmin=-1, vmax=1, aspect='auto')
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.title(title)
        # rótulos enxutos (até 40 para caber)
        if len(names) <= 40:
            plt.xticks(range(len(names)), names, rotation=90, fontsize=7)
            plt.yticks(range(len(names)), names, fontsize=7)
        else:
            plt.xticks([]); plt.yticks([])
        plt.tight_layout()
        plt.show()

    def plot_pca_scatter(X, title="PCA(2D) of normalized vectors"):
        # PCA 2D “na unha”
        Xc = np.nan_to_num(X, nan=0.0)
        Xc = Xc - Xc.mean(axis=0, keepdims=True)
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        Z = Xc @ Vt[:2].T
        plt.figure(figsize=(6,5))
        plt.scatter(Z[:,0], Z[:,1], s=6, alpha=0.5)
        plt.title(title)
        plt.xlabel("PC1"); plt.ylabel("PC2")
        plt.tight_layout()
        plt.show()

    # df = events_to_df(eventos)
    scaler = fit_event_vector_scaler(df, p_lo=1, p_hi=99)
    X01, feat_names = transform_event_vectors(df, scaler, range_mode="0-1")

    # Ver distribuições
    plot_vector_hists(X01, feat_names, per_page=24, bins=40, title="Dists 0..1")
    plot_corr_heatmap(X01, feat_names)
    plot_pca_scatter(X01, "PCA 0..1")

    # Textos compactos (<WINDOW> only)
    lines = make_all_event_window_lines(eventos, asset="WDO", alt_asset="DOL")
    for i in range(min(5, len(lines))):
        print(lines[i])


    breakpoint()