from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Iterable
from collections import defaultdict, deque
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from datetime import datetime, timezone
import matplotlib.dates as mdates
from zoneinfo import ZoneInfo

LOCAL_TZ = ZoneInfo("America/Sao_Paulo")

# ===== helpers mínimos (sem utils) =====
def summarize_second(trades: List[List[float]]) -> Dict[str, Optional[float]]:
    """Summarize a list of trades for a single second.

    The trades list is expected to contain rows where the first element is
    price and the second element (if present) is the lot size. Additional
    columns are ignored for the purpose of summarisation.

    Returns a dictionary with the following keys:

    ``n``: number of trades
    ``vol``: total volume (sum of lots)
    ``lot_max``: maximum lot in this second
    ``open``: opening price (first trade)
    ``close``: closing price (last trade)
    ``pmin``: minimum price
    ``pmax``: maximum price

    If there are no trades the returned values are set to safe defaults.
    """
    if not trades:
        return {
            "n": 0,
            "vol": 0.0,
            "lot_max": 0.0,
            "open": None,
            "close": None,
            "pmin": None,
            "pmax": None,
        }
    prices = [float(tr[0]) for tr in trades]
    lots = [float(tr[1]) if len(tr) > 1 else 1.0 for tr in trades]
    return {
        "n": len(trades),
        "vol": float(sum(lots)),
        "lot_max": float(max(lots)),
        "open": prices[0],
        "close": prices[-1],
        "pmin": min(prices),
        "pmax": max(prices),
    }

def extrair_preco(x: np.ndarray) -> np.ndarray:
    """Extract periodic price features used for context information.

    For each price in ``x`` this function computes a vector of five features
    corresponding to different periodic transforms of the price modulo 10 and
    100. These features were introduced in the original script to capture
    meaningful price patterns relative to common tick thresholds.

    Parameters
    ----------
    x : np.ndarray
        Array of price values.

    Returns
    -------
    np.ndarray
        A two‑dimensional array where each row corresponds to the five
        features for the associated input price.
    """
    def suave_5_0(x_val: np.ndarray) -> np.ndarray:
        resto = x_val % 10
        return np.cos(np.pi * resto / 5)

    def quebra_no_bola(x_val: np.ndarray) -> np.ndarray:
        frac = x_val % 10
        return -np.cos(np.pi * frac / 10)

    def cosseno_localizado(x_val: np.ndarray, centro: float, largura: float = 15.0) -> np.ndarray:
        x_mod = x_val % 100
        dist = np.abs(x_mod - centro)
        y = np.full_like(x_mod, -1.0, dtype=float)
        dentro = dist <= largura
        y[dentro] = np.cos(np.pi * (x_mod[dentro] - centro) / largura)
        return y

    def funcao_357(x_val: np.ndarray) -> np.ndarray:
        picos = [0, 30, 50, 70, 100]
        largura = 15.0
        x_arr = np.asarray(x_val, dtype=float)
        y = np.full_like(x_arr, -1.0, dtype=float)
        for c in picos:
            y = np.maximum(y, cosseno_localizado(x_arr, c, largura))
        return y

    # Vectorised computations on numpy arrays
    s10 = suave_5_0(x)
    s100 = suave_5_0(x / 10)
    o10 = quebra_no_bola(x)
    o100 = quebra_no_bola(x / 10)
    f100 = funcao_357(x)

    return np.stack([s10, s100, o10, o100, f100], axis=-1)

def ticks_delta(p0: Optional[float], p1: Optional[float], tick: float) -> float:
    if p0 is None or p1 is None or not tick or tick <= 0:
        return 0.0
    return (float(p1) - float(p0)) / float(tick)

def _bucket_price_stats_ab(
    trades_a: List[List[float]],
    trades_b: List[List[float]],
    close_a: Optional[float],
    close_b: Optional[float],
    tick_a: float,
    tick_b: float,
):
    """
    Buckets de preço por segundo:
      - Se houver trades em A (WDO), bucketiza por preço de A (tick_a).
      - Caso contrário, bucketiza por preço de B (tick_b).
    Volume:
      - Soma o volume do símbolo de referência normalmente por preço.
      - Soma o volume do OUTRO símbolo no bucket do 'preço de fechamento' do símbolo de referência
        (não temos mapeamento de preço cruzado, então alocamos no close do ref).
    Retorna: price_count (dict), price_vol (dict), sec_price (float ou None)
    """
    def bucket(px: float, tk: float) -> float:
        return (round(px / tk) * tk) if (tk and tk > 0) else float(px)

    use_a = len(trades_a) > 0  # A é referência se tiver trade neste segundo
    if use_a:
        ref_trades, oth_trades = trades_a, trades_b
        tk_ref, tk_oth = tick_a, tick_b
        sec_price = bucket(float(close_a), tk_ref) if close_a is not None else (
            bucket(float(trades_a[-1][0]), tk_ref) if trades_a else None
        )
    else:
        ref_trades, oth_trades = trades_b, trades_a
        tk_ref, tk_oth = tick_b, tick_a
        sec_price = bucket(float(close_b), tk_ref) if close_b is not None else (
            bucket(float(trades_b[-1][0]), tk_ref) if trades_b else None
        )

    price_count: Dict[float, int] = defaultdict(int)
    price_vol:   Dict[float, float] = defaultdict(float)

    # Volume e contagem do símbolo de referência, por preço
    for tr in ref_trades:
        if not tr: 
            continue
        px  = float(tr[0])
        lot = float(tr[1]) if len(tr) > 1 else 0.0
        k = bucket(px, tk_ref)
        price_count[k] += 1
        price_vol[k]   += lot

    # Volume do outro símbolo: aloca no bucket do preço de FECHAMENTO do ref
    if sec_price is not None and oth_trades:
        vol_oth = 0.0
        for tr in oth_trades:
            if not tr: 
                continue
            vol_oth += float(tr[1]) if len(tr) > 1 else 0.0
        price_vol[sec_price] += vol_oth

    return price_count, price_vol, sec_price

def _vol_by_aggr(trades: Iterable[Iterable[float]]) -> Tuple[float,float,float]:
    buy = sell = tot = 0.0
    for tr in trades:
        if not tr: continue
        lot = float(tr[1]) if len(tr) > 1 else 0.0
        ag  = int(tr[4]) if len(tr) > 4 else 0
        tot += lot
        if ag == 1:   buy  += lot
        elif ag == 2: sell += lot
    return buy, sell, tot

def _group_stats_one_symbol_local(trades: List[List[float]], tick: float) -> Tuple[float,float,int,int,int]:
    """
    'Grupo' = (aggressor, broker_agressor) dentro do segundo.
    Retorna:
      max_lot, max_ticks (entre grupos), g_buy, g_sell, g_neu
    """
    if not trades:
        return 0.0, 0.0, 0, 0, 0
    groups: Dict[Tuple[int,int], Dict[str, float]] = {}
    for tr in trades:
        px  = float(tr[0]) if len(tr) > 0 else 0.0
        lot = float(tr[1]) if len(tr) > 1 else 0.0
        b_comp = int(tr[2]) if len(tr) > 2 else -1
        b_vend = int(tr[3]) if len(tr) > 3 else -1
        ag  = int(tr[4]) if len(tr) > 4 else 0
        if ag == 1:
            key = (1, b_comp)
        elif ag == 2:
            key = (-1, b_vend)
        else:
            key = (0, -1)
        g = groups.get(key)
        if g is None:
            groups[key] = {"lot": lot, "p_open": px, "p_close": px}
        else:
            g["lot"] += lot
            g["p_close"] = px

    max_lot = 0.0
    max_ticks = 0.0
    g_buy = g_sell = g_neu = 0
    for (side, _), g in groups.items():
        max_lot = max(max_lot, float(g["lot"]))
        dtk = abs(ticks_delta(g["p_open"], g["p_close"], tick))
        max_ticks = max(max_ticks, dtk)
        if side > 0:   g_buy  += 1
        elif side < 0: g_sell += 1
        else:          g_neu  += 1
    return max_lot, max_ticks, g_buy, g_sell, g_neu

def _stats_list(lst: List[float]) -> Dict[str, float | int]:
    if not lst:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "n": 0}
    arr = np.asarray(lst, dtype=float)
    return {"mean": float(arr.mean()), "max": float(arr.max()), "sum": float(arr.sum())}

def _sma_last(buf: List[float], window: int) -> float:
    if not buf: return 0.0
    if window <= 0 or len(buf) <= window:
        return float(np.mean(buf))
    return float(np.mean(buf[-window:]))

# ===== 1) acumulador simplificado =====
def event_acc_init() -> Dict[str, object]:
    return {
        # Histograma de preço (WDO)
        "price_count": defaultdict(int),
        "price_vol": defaultdict(float),
        "secs_at_price": defaultdict(int),

        # Volume por lado e total
        "buy_vol_a": 0.0, "sell_vol_a": 0.0, "tot_vol_a": 0.0,
        "buy_vol_b": 0.0, "sell_vol_b": 0.0, "tot_vol_b": 0.0,

        # Estatísticas de grupos (por segundo)
        "max_group_lot": 0.0,
        "max_group_ticks": 0.0,
        "g_buy": 0, "g_sell": 0, "g_neu": 0,

        # Taxa de ordens
        "orders_started_sum": 0.0,   # soma por segundo
        "rate_max": 0.0,             # máximo por segundo

        # Streaks (predominância por segundo)
        "streak_buy": 0,
        "streak_sell": 0,
        "max_streak_buy": 0,
        "max_streak_sell": 0,

        # Série por segundo (movimento e volumes)
        "persec": [],  # cada item: {dtk_a, dtk_b, buy_a, sell_a, buy_b, sell_b}

        # Absorção (pernas e retrações) — ref WDO
        "run_sign": 0,
        "run_ticks": 0.0,
        "run_secs": 0,
        "run_dir_vol": 0.0,
        "retrace_ticks": 0.0,
        "retrace_secs": 0,
        "absorb_buy": [],
        "absorb_sell": [],

        # Players (tops do dia)
        "top_buy_vol_a": 0.0, "top_sell_vol_a": 0.0,
        "top_buy_vol_b": 0.0, "top_sell_vol_b": 0.0,
    }

# ===== 2) acumular um segundo no evento (sem utils.*) =====
def event_accumulate_second(
    ev_acc: Dict[str, object],
    trades_a: List[List[float]],
    trades_b: List[List[float]],
    smry_a: Dict[str, Optional[float]],
    smry_b: Dict[str, Optional[float]],
    close_a: Optional[float],
    close_b: Optional[float],
    last_close_a: Optional[float],
    last_close_b: Optional[float],
    tick_a: float,
    tick_b: float,
    starts_a: int,
    starts_b: int,
    p,  # SegParams
    player_state_a: Optional[Dict[int, Dict[str, float]]] = None,
    player_state_b: Optional[Dict[int, Dict[str, float]]] = None,
) -> None:

    # ---- helper: top-K sets (compradores/vendedores) a partir do estado ----
    def _top_sets_by_state(state: Optional[Dict[int, Dict[str, float]]], k: int = 5):
        if not state:
            return set(), set()

        # se houver cumulativos, priorize-os; senão, use posição líquida
        has_cum = any(("cum_buy" in v or "cum_sell" in v) for v in state.values())

        if has_cum:
            buyers = sorted(((aid, float(v.get("cum_buy", 0.0))) for aid, v in state.items()),
                            key=lambda kv: kv[1], reverse=True)
            sellers = sorted(((aid, float(v.get("cum_sell", 0.0))) for aid, v in state.items()),
                             key=lambda kv: kv[1], reverse=True)
            buyers = [aid for aid, vol in buyers if vol > 0][:k]
            sellers = [aid for aid, vol in sellers if vol > 0][:k]
        else:
            # por posição líquida: >0 são compradores do dia; <0 vendedores
            buyers = sorted(((aid, float(v.get("pos", 0.0))) for aid, v in state.items() if float(v.get("pos", 0.0)) > 0),
                            key=lambda kv: kv[1], reverse=True)
            sellers = sorted(((aid, -float(v.get("pos", 0.0))) for aid, v in state.items() if float(v.get("pos", 0.0)) < 0),
                             key=lambda kv: kv[1], reverse=True)
            buyers = [aid for aid, _ in buyers][:k]
            sellers = [aid for aid, _ in sellers][:k]

        return set(buyers), set(sellers)

    # K configurável, default=5
    TOP_K = int(getattr(p, "top_k_players", 5))

    # locks/keys extras (se não existirem ainda)
    ev_acc.setdefault("top_buy_vol_a", 0.0)
    ev_acc.setdefault("top_sell_vol_a", 0.0)
    ev_acc.setdefault("top_buy_vol_b", 0.0)
    ev_acc.setdefault("top_sell_vol_b", 0.0)
    ev_acc.setdefault("top_dir_part_a", 0.0)  # participação direcional (A)
    ev_acc.setdefault("top_dir_part_b", 0.0)  # participação direcional (B)
    ev_acc.setdefault("top_dir_part",   0.0)  # soma A+B

    # --- histogramas de preço (WDO) ---
    price_count, price_vol, sec_price = _bucket_price_stats_ab(
        trades_a, trades_b, close_a, close_b, tick_a, tick_b
    )
    for k,v in price_count.items():
        ev_acc["price_count"][k] += v
    for k,v in price_vol.items():
        ev_acc["price_vol"][k] += v
    if sec_price is not None:
        ev_acc["secs_at_price"][sec_price] += 1

    # --- volumes por agressor ---
    buy_a, sell_a, tot_a = _vol_by_aggr(trades_a)
    buy_b, sell_b, tot_b = _vol_by_aggr(trades_b)
    ev_acc["buy_vol_a"] += buy_a; ev_acc["sell_vol_a"] += sell_a; ev_acc["tot_vol_a"] += tot_a
    ev_acc["buy_vol_b"] += buy_b; ev_acc["sell_vol_b"] += sell_b; ev_acc["tot_vol_b"] += tot_b

    # --- grupos por símbolo (lado+broker agressor) ---
    mlot_a, mtick_a, g_buy_a, g_sell_a, g_neu_a = _group_stats_one_symbol_local(trades_a, tick_a)
    mlot_b, mtick_b, g_buy_b, g_sell_b, g_neu_b = _group_stats_one_symbol_local(trades_b, tick_b)
    ev_acc["max_group_lot"]   = max(ev_acc["max_group_lot"], mlot_a, mlot_b)
    ev_acc["max_group_ticks"] = max(ev_acc["max_group_ticks"], mtick_a, mtick_b)
    ev_acc["g_buy"]  += (g_buy_a  + g_buy_b)
    ev_acc["g_sell"] += (g_sell_a + g_sell_b)
    ev_acc["g_neu"]  += (g_neu_a  + g_neu_b)

    # --- taxa de ordens (por segundo) ---
    rate_now = float(starts_a + starts_b)
    ev_acc["orders_started_sum"] += rate_now
    ev_acc["rate_max"] = max(ev_acc["rate_max"], rate_now)

    # --- streaks por segundo (lado dominante por #grupos) ---
    if (g_buy_a + g_buy_b) > (g_sell_a + g_sell_b):
        ev_acc["streak_buy"]  += 1
        ev_acc["streak_sell"]  = 0
    elif (g_sell_a + g_sell_b) > (g_buy_a + g_buy_b):
        ev_acc["streak_sell"] += 1
        ev_acc["streak_buy"]   = 0
    else:
        ev_acc["streak_buy"] = 0
        ev_acc["streak_sell"] = 0
    ev_acc["max_streak_buy"]  = max(ev_acc["max_streak_buy"],  ev_acc["streak_buy"])
    ev_acc["max_streak_sell"] = max(ev_acc["max_streak_sell"], ev_acc["streak_sell"])

    # --- movimento por segundo (dtk) + registro de volumes no segundo ---
    dtk_a = ticks_delta(last_close_a, close_a, tick_a)
    dtk_b = ticks_delta(last_close_b, close_b, tick_b)
    ev_acc["persec"].append({
        "dtk_a": dtk_a, "dtk_b": dtk_b,
        "buy_a": buy_a, "sell_a": sell_a,
        "buy_b": buy_b, "sell_b": sell_b,
    })

    # --- absorção (ref = símbolo A/WDO) ---
    use_a = (str(getattr(p, "price_ref_symbol", "a")).lower() == "a")
    dtk_ref = dtk_a if use_a else dtk_b
    dir_buy_vol  = (buy_a + buy_b)
    dir_sell_vol = (sell_a + sell_b)

    sgn = 1 if dtk_ref > 0 else (-1 if dtk_ref < 0 else 0)
    if sgn == 0:
        pass
    elif ev_acc["run_sign"] == 0:
        ev_acc["run_sign"] = sgn
        ev_acc["run_ticks"] = abs(dtk_ref)
        ev_acc["run_secs"]  = 1
        ev_acc["retrace_ticks"] = 0.0
        ev_acc["retrace_secs"]  = 0
        ev_acc["run_dir_vol"]   = (dir_buy_vol if sgn > 0 else dir_sell_vol)
    elif sgn == ev_acc["run_sign"]:
        ev_acc["run_ticks"] += abs(dtk_ref)
        ev_acc["run_secs"]  += 1
        ev_acc["run_dir_vol"] += (dir_buy_vol if sgn > 0 else dir_sell_vol)
        ev_acc["retrace_ticks"] = 0.0
        ev_acc["retrace_secs"]  = 0
    else:
        ev_acc["retrace_ticks"] += abs(dtk_ref)
        ev_acc["retrace_secs"]  += 1
        if ev_acc["retrace_ticks"] >= getattr(p, "absorb_ticks_thr", 2) and ev_acc["run_ticks"] >= getattr(p, "absorb_ticks_thr", 2):
            rec = {"ticks": ev_acc["retrace_ticks"], "secs": ev_acc["retrace_secs"]}
            if ev_acc["run_sign"] > 0:
                ev_acc["absorb_sell"].append(rec)
            else:
                ev_acc["absorb_buy"].append(rec)
            # start nova perna
            ev_acc["run_sign"]  = sgn
            ev_acc["run_ticks"] = abs(dtk_ref)
            ev_acc["run_secs"]  = 1
            ev_acc["run_dir_vol"] = (dir_buy_vol if sgn > 0 else dir_sell_vol)
            ev_acc["retrace_ticks"] = 0.0
            ev_acc["retrace_secs"]  = 0

    # --- tops (derivados do estado do dia) ---
    topB_a, topS_a = _top_sets_by_state(player_state_a, TOP_K)
    topB_b, topS_b = _top_sets_by_state(player_state_b, TOP_K)

    # função interna: acumula participação dado trades e conjuntos top
    def _accumulate_top_participation(trades, topB: set, topS: set, sgn_dir: int, which: str):
        """
        sgn_dir: +1 para alta, -1 para baixa, 0 = neutro (ignora)
        which: "a" ou "b"
        Regras:
          - sgn>0 (alta): +lote se (ag=1 e buyer ∈ topB), -lote se (ag=2 e seller ∈ topS)
          - sgn<0 (baixa): +lote se (ag=2 e seller ∈ topS), -lote se (ag=1 e buyer ∈ topB)
        Também mantém contadores crus top_buy_vol_* e top_sell_vol_* (independente de direção).
        """
        if sgn_dir == 0:
            return 0.0

        dir_acc = 0.0
        for tr in trades:
            lote = float(tr[1]) if len(tr) > 1 else 0.0
            ag   = int(tr[4]) if len(tr) > 4 else 0    # 1=buy, 2=sell
            b_comp = int(tr[2]) if len(tr) > 2 else -1
            b_vend = int(tr[3]) if len(tr) > 3 else -1

            # contadores crus (mantém compatibilidade com o que você já tinha)
            if which == "a":
                if ag == 1 and b_comp in topB: ev_acc["top_buy_vol_a"]  += lote
                if ag == 2 and b_vend in topS: ev_acc["top_sell_vol_a"] += lote
            else:
                if ag == 1 and b_comp in topB: ev_acc["top_buy_vol_b"]  += lote
                if ag == 2 and b_vend in topS: ev_acc["top_sell_vol_b"] += lote

            # participação direcional (soma se alinha, subtrai se contrária)
            if sgn_dir > 0:
                if ag == 1 and b_comp in topB:
                    dir_acc += lote       # buy em alta -> a favor
                if ag == 2 and b_vend in topS:
                    dir_acc -= lote       # sell em alta -> contra
            else:  # sgn_dir < 0
                if ag == 2 and b_vend in topS:
                    dir_acc += lote       # sell em baixa -> a favor
                if ag == 1 and b_comp in topB:
                    dir_acc -= lote       # buy em baixa -> contra

        if which == "a":
            ev_acc["top_dir_part_a"] += dir_acc
        else:
            ev_acc["top_dir_part_b"] += dir_acc

        return dir_acc

    # acumula por símbolo e totaliza
    part_a = _accumulate_top_participation(trades_a, topB_a, topS_a, sgn, "a")
    part_b = _accumulate_top_participation(trades_b, topB_b, topS_b, sgn, "b")
    ev_acc["top_dir_part"] += (part_a + part_b)

    # opcional: registrar por segundo no vetor persec, junto do que você já salva
    if ev_acc["persec"]:
        ev_acc["persec"][-1]["top_part_a"] = part_a
        ev_acc["persec"][-1]["top_part_b"] = part_b
        ev_acc["persec"][-1]["top_part"]   = part_a + part_b


# ===== 3) enriquecimento final simplificado =====
def finalize_event_enrichment(
    ev: Dict[str, object],
    p,                # SegParams
    metrics,          # RefMetrics (usa ema_vol_sec, ema_order_rate, buf_30m)
    tick_a: float,
    tick_b: float,
    extrair_preco_fn = lambda arr: np.zeros((1,5), dtype=float),
) -> Dict[str, object]:

    acc = ev.pop("acc", None)
    if not acc:
        return ev

    def _safe_div(a, b, default=0.0):
        b = float(b)
        return float(a) / b if abs(b) > 1e-12 else float(default)
    
    # --- preço mais negociado / maior volume / mais tempo (WDO) ---
    def _argmax(d: Dict[float, float]) -> Tuple[Optional[float], float]:
        if not d: return None, 0.0
        k, v = max(d.items(), key=lambda kv: kv[1])
        return k, float(v)

    price_most_traded, _ = _argmax(acc["price_count"])
    price_most_volume, _ = _argmax(acc["price_vol"])
    price_most_time,   _ = _argmax(acc["secs_at_price"])

    # Chaves básicas do evento (seguindo seu step simplificado)
    p_open  = float(ev.get("open", 0.0))
    p_close = float(ev.get("close", 0.0))
    p_high  = float(ev.get("max", 0.0))
    p_low   = float(ev.get("min", 0.0))
    dur     = int(ev.get("dur", 0))
    rng     = ev.get("range")
    dp = p_open - p_close
    dmx = p_open - p_high
    dmm = p_open - p_low
    dmt = p_open - price_most_traded
    dmv = p_open - price_most_volume

    buy_a, sell_a, tot_a = acc["buy_vol_a"], acc["sell_vol_a"], acc["tot_vol_a"]
    buy_b, sell_b, tot_b = acc["buy_vol_b"], acc["sell_vol_b"], acc["tot_vol_b"]
    tot_all = float(tot_a + tot_b)
    n_orders_avg_base = max(1.0, float(acc["orders_started_sum"]))  # soma das taxas por segundo


    # 4) Movimento
    # 4.1 variações por tick/volume separados (WDO/DOL; up usa buy, down usa sell)
    up_a: List[float] = []; dn_a: List[float] = []
    up_b: List[float] = []; dn_b: List[float] = []
    for r in acc["persec"]:
        dtk_a = float(r["dtk_a"]); dtk_b = float(r["dtk_b"])
        if   dtk_a > 0 and dtk_a != 0: up_a.append(float(r["buy_a"])  / abs(dtk_a))
        elif dtk_a < 0 and dtk_a != 0: dn_a.append(float(r["sell_a"]) / abs(dtk_a))
        if   dtk_b > 0 and dtk_b != 0: up_b.append(float(r["buy_b"])  / abs(dtk_b))
        elif dtk_b < 0 and dtk_b != 0: dn_b.append(float(r["sell_b"]) / abs(dtk_b))

    # 4.2 shares de participação do WDO no deslocamento (por lado)
    buy_share_a  = (buy_a / (buy_a + buy_b)) if (buy_a + buy_b) > 0 else 0.0
    sell_share_a = (sell_a / (sell_a + sell_b)) if (sell_a + sell_b) > 0 else 0.0

    # 4.3 absorção
    def _abs_stats(lst: List[Dict[str, float]]) -> Dict[str, float]:
        if not lst:
            return {"count": 0, "ticks_mean": 0.0, "ticks_max": 0.0, "vel_mean": 0.0}
        ticks = np.asarray([x["ticks"] for x in lst], dtype=float)
        vels  = np.asarray([x["ticks"]/max(1, x["secs"]) for x in lst], dtype=float)
        return {"count": len(lst), "ticks_mean": float(ticks.mean()), "ticks_max": float(ticks.max()), "vel_mean": float(vels.mean())}

    abs_buy = _abs_stats(acc["absorb_buy"])
    abs_sell= _abs_stats(acc["absorb_sell"])

    # 4.4 facilidade do deslocamento (ref WDO)
    net_ticks = abs(ticks_delta(p_open, p_close, tick_a))
    dir_sign = 1 if (p_close - p_open) > 0 else (-1 if (p_close - p_open) < 0 else 0)
    opp_vol = (sell_a + sell_b) if dir_sign > 0 else ((buy_a + buy_b) if dir_sign < 0 else 0.0)
    ease_of_move = (net_ticks / max(1.0, opp_vol)) if dir_sign != 0 else 0.0

    # SMA 15m do preço do WDO usando buffer de 30m do metrics (1Hz). Janela = 900s.
    d5m = p_close - metrics.mean_price_5m
    d30 = p_close - metrics.mean_price_30m
        
    dvwap = p_close - metrics.vwap
    ddayo = p_close - metrics.abertura

    ctx_price = p_close  # último preço WDO do evento
    ctx_feats = extrair_preco_fn(np.array([float(ctx_price)], dtype=float))[0].tolist()


    ev["vector"] = {
        "dt": dur,
        "dp": dp,
        "dmx": dmx,
        "dmm": dmm,
        "dmt": dmt,
        "dmv": dmv,
        "dvwap": dvwap,
        "ddayo": ddayo,
        "d5m": d5m,
        "d30m": d30,
        "rng": rng,
        "efrang": abs(p_close-p_open) / max(0.1, rng), #eficiencia de range
        "pctx0": ctx_feats[0],
        "pctx1": ctx_feats[1],
        "pctx2": ctx_feats[2],
        "pctx3": ctx_feats[3],
        "pctx4": ctx_feats[4],
        "b_vol": buy_a + buy_b,
        "s_vol": sell_a + sell_b,
        "t_vol": tot_all,
        "d_vol": buy_a+buy_b-sell_a-sell_b,
        "avg_vol": (tot_all / n_orders_avg_base),
        "g_b":   int(acc["g_buy"]), #grupos buy
        "g_s":  int(acc["g_sell"]), #grupos sell
        "g_n": int(acc["g_neu"]),   #grupos neutro
        "rate_avg": (float(acc["orders_started_sum"]) / max(1, dur)),
        "max_streak_buy":  int(acc["max_streak_buy"]),
        "max_streak_sell": int(acc["max_streak_sell"]),
        "ema_vol_total": float(getattr(metrics, "ema_vol_total", 0.0) or 0.0),
        "ema_order_rate": float(getattr(metrics, "ema_order_rate", 0.0) or 0.0),
        "d_ticks_max": float(acc["max_group_ticks"]),
        "vol_per_tick_up_a":  _stats_list(up_a),
        "vol_per_tick_dn_a":  _stats_list(dn_a),
        "vol_per_tick_up_b":  _stats_list(up_b),
        "vol_per_tick_dn_b":  _stats_list(dn_b),
        "buy_share_wdo":  float(buy_share_a),
        "sell_share_wdo": float(sell_share_a),
        "absorptions_buy":  abs_buy,
        "absorptions_sell": abs_sell,
        "ease_of_move": float(ease_of_move),
        "top_buy_vol_wdo": float(acc["top_buy_vol_a"]),
        "top_sell_vol_wdo": float(acc["top_sell_vol_a"]),
        "top_buy_vol_dol": float(acc["top_buy_vol_b"]),
        "top_sell_vol_dol": float(acc["top_sell_vol_b"]),
    }

    return ev

# ===== dataclass de parâmetros =====
@dataclass
class SegParams:
    """Hyperparameters controlling event segmentation.

    These parameters tune the behaviour of the event detection
    algorithm. They correspond closely to the variables in the original
    implementation but have been grouped into a dataclass for
    readability and ease of configuration. See the original code for
    descriptions of each field.
    """
    
    # tick sizes and lot multipliers for symbols A and B
    tick_a: float = 0.5
    tick_b: float = 0.5
    lot_mult_a: float = 1.0
    lot_mult_b: float = 5.0

    # EMAs for volume, range and order rate (time constants in minutes)
    tau_vol_min: float = 1.0
    tau_var_min: float = 1.0
    tau_order_min: float = 1.0

    end_grace_s: int = 1  # inclui +1 segundo no próprio evento antes de encerrar
    # Maximum event duration (seconds)
    max_dur_s: int = 120

    # Absorption detection thresholds
    absorb_ticks_thr: int = 2

    player_vol_mult: float = 2.0  # fator K do limiar: K * mean_vol_5m
    vol_spike_mult: float = 2.0    # multiplicador K para o spike de volume
    vol_spike_mult_cum: float = 8.0 # multiplicador K para o spike acumulado de volume
    range_decay_tau_s: float = 120.0 # (DECAIMENTO) constante de tempo (seg) da exponencial
    range_div: float = 2.0
    # Speed (taxa de variação)
    speed_win_s: int = 3        # janela curta dentro do evento (ex.: 5 segundos)
    speed_mult: float = 3.0     # multiplicador K para disparo: speed_evt >= K * mean_speed_5m
    rate_mult: float = 7.0  # fator K para o corte de order rate
    rate_mult_evt: float = 2.0  # fator K para o corte de order rate dentro do evento

    # Which symbol to use as the price reference ("a" for WDO/WIN or "b" for DOL/IND)
    price_ref_symbol: str = "a"

    # Top players of the day (optional)
    top_buyers_a: Tuple[int, ...] = ()
    top_sellers_a: Tuple[int, ...] = ()
    top_buyers_b: Tuple[int, ...] = ()
    top_sellers_b: Tuple[int, ...] = ()

    # ----------------------------------------------------------
    # >>> Fábricas de configuração por ativo <<<
    # ----------------------------------------------------------
    @classmethod
    def for_indice(cls) -> "SegParams":
        """Configuração típica para IND/WIN"""
        return cls(
            tick_a=5.0,
            tick_b=5.0,
            lot_mult_a=1.0,
            lot_mult_b=5.0,
            max_dur_s=120,
            vol_spike_mult=3.0,
            vol_spike_mult_cum=10.0,
            player_vol_mult=2.0,
            speed_win_s=2,
            speed_mult=2.0,
            rate_mult=6.0,
            range_decay_tau_s=120.0,
            price_ref_symbol="a",
        )

def detect_player_big_hit(orders_a, orders_b, metrics, params) -> bool:
    """
    Dispara True se QUALQUER ordem agregada (A ou B), ponderada por lot_mult,
    for >= K * média de 5min do volume (também por segundo).
    K = params.player_vol_mult
    """
    # base = média de 5min; se ainda zero (aquecimento), usa EMA curta
    base = float(metrics.ema_vol_total)
    if base <= 0.0:
        base = float(metrics.ema_vol_sec)
    if base <= 0.0:
        return False  # sem base ainda, não dispara

    thr = float(params.player_vol_mult) * base

    ret = False
    # varre A (ponderado por lot_mult_a)
    if orders_a:
        for o in orders_a:
            lot_eff = float(o.get("lot", 0.0))
            if lot_eff >= thr / 2.0:
                # print(f"player wdo: {lot_eff} thr: {thr}")
                ret = True

    # varre B (ponderado por lot_mult_b)
    if orders_b:
        for o in orders_b:
            lot_eff = float(o.get("lot", 0.0))
            if lot_eff >= thr:
                # print(f"player dol: {lot_eff / 5} thr: {thr}")
                ret = True


    return ret

def detect_vol_spike_two_conditions(
    vol_inst: float,
    vol_cum: float,
    metrics,
    params,
) -> bool:
    """
    Retorna: (inst_hit, cum_hit, thr)
      - inst_hit: volume total do segundo >= K * média 5m
      - cum_hit : volume acumulado dentro do segundo cruzou K * média em algum ponto
      - thr     : limiar usado (K * base)

    Pré-requisito: trades_a e trades_b já com lot multiplicado (t[1]).
    """
    # base do limiar
    base = float(metrics.ema_vol_total)
    if base <= 0.0:
        base = float(metrics.ema_vol_sec)
    if base <= 0.0:  # sem base numérica ainda
        return False

    K0 = float(params.vol_spike_mult)
    thr0 = K0 * base
    K1 = float(params.vol_spike_mult_cum)
    thr1 = K1 * base
    # (1) condição instantânea
    inst_hit = float(vol_inst) >= thr0
    # if inst_hit:
    #     print(f"vol_inst: {vol_inst} thr0: {thr0}")

    # (2) condição acumulada no evento
    cum_hit = float(vol_cum) >= thr1
    # if cum_hit:
    #     print(f"vol_cum: {vol_cum} thr1: {thr1}")

    return inst_hit | cum_hit

def detect_range_hit(range, dur, metrics, params: "SegParams") -> bool:

    # base de comparação: range_5m das métricas (se zero no warm-up, usa range_30m como fallback)
    base_range = float(metrics.range_5m)

    if base_range <= 0.0:
        return False  # sem base confiável ainda

    Kt = math.exp(-dur / params.range_decay_tau_s)
    thr = Kt * base_range / params.range_div

    # if range >= thr:
    #     print(f"range: {range} thr: {thr} dur: {dur} base_range: {base_range}")

    # dispara quando o range do evento ultrapassa o threshold dinâmico
    return range >= thr

def detect_speed_hit(metrics: "RefMetrics", evt: dict, params: "SegParams", min_a, max_a, last_close) -> bool:
    """
    Dispara se speed_evt >= K * mean_speed_5m.
    Usa mean_speed_5m como baseline; se zero, usa ema_speed_sec.
    """
    if max_a is not None and min_a is not None and last_close is not None:
        dpx = abs(float(max_a) - float(min_a))
        dpx = max(dpx, abs(float(last_close) - float(min_a)))
        dpx = max(dpx, abs(float(last_close) - float(max_a)))
    else:
        dpx = 0.0
    
    # baseline
    base = float(metrics.mean_speed_5m)
    if base <= 0.0:
        base = float(metrics.ema_speed_sec)
    if base <= 0.0:
        return False  # sem baseline ainda

    K = float(params.speed_mult)
    thr = K * base

    # if dpx >= thr:
    #     print(f"speed_evt: {dpx} thr: {thr}")

    return dpx >= thr

def detect_order_rate_cut_simple(metrics, evt, n_orders: int, params) -> bool:
    # baseline de ordens/s
    base = float(metrics.mean_ord_5m)
    if base <= 0.0:
        base = float(metrics.ema_order_rate)
    if base <= 0.0:
        return False  # sem baseline ainda

    thr = float(params.rate_mult) * base
    thr2 = float(params.rate_mult_evt) * base

    # condição 1: taxa instantânea no segundo (ordens/s)
    sec_hit = float(n_orders) >= thr

    # condição 2: taxa média do evento (ordens acumuladas / duração em s)
    evt_hit = False
    if evt is not None:
        evt_n = float(evt.get("n_trades", 0))
        evt_dur = float(evt.get("dur", 0.0))
        if evt_dur > 0.0:
            evt_rate = evt_n / evt_dur
            evt_hit = evt_rate >= thr2

    return sec_hit or evt_hit

class RenkoBuilder:
    def __init__(self, brick_size: float, reversal_bricks: int = 2, anchor_mode: str = "round"):
        self.brick_size = float(brick_size)
        self.reversal_bricks = int(reversal_bricks)
        self.anchor_mode = anchor_mode
        self.reset()

    def reset(self, anchor_price: float = None):
        self.last = None if anchor_price is None else self._anchor_to_grid(anchor_price)
        self.dir = 0                # +1 up, -1 down, 0 neutro
        self.closes = []            # preços de fechamento dos tijolos
        self.dirs = []              # direções dos tijolos (+1/-1)
        self.brick_times = []       # timestamps dos tijolos (quando fecharam)
        self.inversion_times = []  # ← NOVO
        self._last_brick_dir = 0
        self._inversion_hit = False

    def _anchor_to_grid(self, price: float) -> float:
        b = self.brick_size
        if self.anchor_mode == "round":
            return round(price / b) * b
        elif self.anchor_mode == "floor":
            return math.floor(price / b) * b
        return price

    def _mark_brick(self, d: int, t=None):
        # detectar inversão (+1 → -1 ou -1 → +1)
        if self._last_brick_dir != 0 and d != self._last_brick_dir:
            self._inversion_hit = True
            if t is not None:
                self.inversion_times.append(t)  # ← guarda o tempo da inversão
        self._last_brick_dir = d
        self.dirs.append(d)
        self.brick_times.append(t)

    def update_tick(self, price: float, t=None):
        if self.last is None:
            self.last = self._anchor_to_grid(price)
            return
        b, rev = self.brick_size, self.reversal_bricks

        # Sobe tijolos
        while price >= self.last + (b if self.dir >= 0 else b * rev):
            self.last += b
            self.dir = +1
            self.closes.append(self.last)
            self._mark_brick(+1, t=t)

        # Desce tijolos
        while price <= self.last - (b if self.dir <= 0 else b * rev):
            self.last -= b
            self.dir = -1
            self.closes.append(self.last)
            self._mark_brick(-1, t=t)

    def pull_inversion_hit(self) -> bool:
        hit = self._inversion_hit
        self._inversion_hit = False
        return hit
    
def feed_renko_from_trades(renko: RenkoBuilder, trades, price_idx=0, t_seconds=None):
    """
    trades: lista de trades do segundo (em ordem). Se t_seconds (epoch do segundo) for dado,
            espelha cada trade como t_seconds + frac.
    """
    n = len(trades)
    if n == 0:
        return
    for k, rec in enumerate(trades):
        price = float(rec[price_idx])
        t = None
        if t_seconds is not None:
            frac = (k + 1) / (n + 1)   # 0<frac<1
            t = t_seconds + frac
        renko.update_tick(price, t=t)

def renko_step_series(renko: RenkoBuilder, last_time=None):
    """
    Constrói (xs, ys) para um gráfico em 'escada' (passo) do Renko no tempo.
    last_time: se quiser estender o último degrau até 'agora' (timestamp float).
    """
    xs = []
    ys = []
    if not renko.closes:
        return xs, ys

    closes = renko.closes
    times = renko.brick_times

    # Garante que tempos sejam floats (epoch)
    times = [float(t) if t is not None else None for t in times]

    # Primeiro degrau começa no tempo do primeiro tijolo
    for i in range(len(closes)):
        t_i = times[i]
        y_i = closes[i]
        if t_i is None:
            continue
        # ponto de “subida” do degrau
        xs.append(datetime.fromtimestamp(t_i, tz=timezone.utc).astimezone(LOCAL_TZ))
        ys.append(y_i)
        # mantém até o próximo tempo (repetindo y)
        t_next = times[i+1] if i+1 < len(times) else None
        if t_next is not None:
            xs.append(datetime.fromtimestamp(t_next, tz=timezone.utc).astimezone(LOCAL_TZ))
            ys.append(y_i)
        elif last_time is not None:
            xs.append(datetime.fromtimestamp(float(last_time), tz=timezone.utc).astimezone(LOCAL_TZ))
            ys.append(y_i)

    return xs, ys

def plot_renko_realtime_with_close(renko: RenkoBuilder, t_close, close_a_vals,
                                   title="Renko (tempo real) + close_a"):
    """Plota Renko + close_a, marcando inversões de direção."""
    # --- limpar valores inválidos ---
    valid = [(t, p) for t, p in zip(t_close, close_a_vals) if p is not None]
    if not valid:
        print("Sem dados válidos de close_a para plot.")
        return

    t_close, close_a_vals = zip(*valid)
    xs_close = [datetime.fromtimestamp(float(t), tz=timezone.utc).astimezone(LOCAL_TZ) for t in t_close]
    ys_close = [float(p) for p in close_a_vals]

    last_t = float(t_close[-1])
    xs_r, ys_r = renko_step_series(renko, last_time=last_t)

    fig, ax = plt.subplots(figsize=(14, 5))
    if xs_r:
        ax.step(xs_r, ys_r, where="post", linewidth=2.0, label=f"Renko {renko.brick_size:g}p")

    # --- linha do close_a ---
    ax.plot(xs_close, ys_close, linewidth=1.0, alpha=0.65, color="tab:blue", label="close_a")

    # --- marcações de inversão ---
    if renko.inversion_times:
        inv_x = [datetime.fromtimestamp(float(t), tz=timezone.utc).astimezone(LOCAL_TZ)
                 for t in renko.inversion_times]
        inv_y = [renko.last] * len(inv_x)  # ou usar valor médio dos últimos tijolos
        ax.scatter(inv_x, inv_y, color="black", marker="v", s=60, label="Inversão")

    # --- estética ---
    ax.set_title(title)
    ax.set_xlabel("tempo (America/Sao_Paulo)")
    ax.set_ylabel("preço")
    ax.grid(True, alpha=0.25)
    loc = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    ax.legend()
    fig.tight_layout()
    plt.show()

def pick_end_causes(cause_flags: dict[str, bool], cause_order: list[str]) -> list[str]:
    """Retorna TODAS as causas ativas (True) na ordem de prioridade."""
    return [k for k in cause_order if cause_flags.get(k, False)]

def push_end_reasons(evt: dict, causes: list[str]) -> None:
    """Acrescenta causas em evt['end_reason'] sem duplicar, preservando a ordem."""
    if not evt or not causes:
        return
    lst = evt.setdefault("end_reason", [])
    for c in causes:
        if c not in lst:
            lst.append(c)

@dataclass
class RefMetrics:
    """Métricas de referência e médias móveis (5 min e 30 min) + VWAP."""

    alpha_vol: float
    alpha_var: float
    alpha_rate: float

    # EMAs rápidas
    ema_vol_sec: float = 0.0
    ema_vol_total: int = 0
    ema_order_rate: float = 0.0

    # tempo
    tempo_total: int = 0

    #abertura
    abertura: float = 0.0

    # Buffers de preço
    buf_5m: List[float] = field(default_factory=list)
    buf_30m: List[float] = field(default_factory=list)

    buf_max_5m: List[float] = field(default_factory=list)
    buf_min_5m: List[float] = field(default_factory=list)
    buf_max_30m: List[float] = field(default_factory=list)
    buf_min_30m: List[float] = field(default_factory=list)

    # Buffers de volume e ordens
    buf_vol_5m: List[float] = field(default_factory=list)
    buf_vol_30m: List[float] = field(default_factory=list)
    buf_ord_5m: List[float] = field(default_factory=list)
    buf_ord_30m: List[float] = field(default_factory=list)

    # Acumuladores VWAP
    cum_vol: float = 0.0
    cum_vwap_num: float = 0.0

    # Resultados calculados
    mean_price_5m: float = 0.0
    mean_price_30m: float = 0.0
    vwap: float = 0.0
    mean_vol_5m: float = 0.0
    mean_vol_30m: float = 0.0
    mean_ord_5m: float = 0.0
    mean_ord_30m: float = 0.0
    range_5m: float = 0.0
    range_30m: float = 0.0

    # --- Volatilidade ---
    prev_close: Optional[float] = None
    var_ema: float = 0.0          # EMA da variância dos retornos
    vol_ema: float = 0.0          # sqrt(var_ema)
    buf_ret2_5m: List[float] = field(default_factory=list)   # últimos 300s de ret^2
    buf_ret2_30m: List[float] = field(default_factory=list)  # últimos 1800s de ret^2
    vol_sma_5m: float = 0.0
    vol_sma_30m: float = 0.0

    # --- Speed (magnitude da variação por segundo) ---
    prev_close: Optional[float] = None         # se já não existir
    ema_speed_sec: float = 0.0                 # EMA de |Δpreço|/s (usa alpha_var como time-constant)
    buf_speed_5m: List[float] = field(default_factory=list)   # últimos 300s de |Δpreço|
    buf_speed_30m: List[float] = field(default_factory=list)  # últimos 1800s de |Δpreço|
    mean_speed_5m: float = 0.0
    mean_speed_30m: float = 0.0

    # HISTÓRICO para plot (novo)
    hist: Dict[str, List[float]] = field(
        default_factory=lambda: {
            "t": [],
            "close": [],
            "vwap": [],
            "mean_price_5m": [],
            "mean_price_30m": [],
            "ema_vol_sec": [],
            "mean_vol_5m": [],
            "mean_vol_30m": [],
            "ema_order_rate": [],
            "mean_ord_5m": [],
            "mean_ord_30m": [],
            "range_5m": [],
            "range_30m": [],
            "vol_ema": [],
            "vol_sma_5m": [],
            "vol_sma_30m": [],
            "speed_ema_sec": [],
            "mean_speed_5m": [],
            "mean_speed_30m": [],
        }
    )

    @classmethod
    def create(cls, tau_vol_min: float, tau_var_min: float, tau_order_min: float) -> "RefMetrics":
        a_vol = 1.0 - math.exp(-1.0 / max(1.0, tau_vol_min * 60.0))
        a_var = 1.0 - math.exp(-1.0 / max(1.0, tau_var_min * 60.0))
        a_rate = 1.0 - math.exp(-1.0 / max(1.0, tau_order_min * 60.0))
        return cls(a_vol, a_var, a_rate)

    def setopenday(self, pa, pb):
        p = pa if pa is not None else pb
        self.abertura = p 

    def update(self,
               close: Optional[float],
               vol: float,
               n_orders: float,
               max_a: float = 0.0,
               min_a: float = 0.0) -> None:
        """Atualiza métricas a cada segundo."""

        # Volume EMA (curto prazo)
        self.ema_vol_sec = (1.0 - self.alpha_vol) * self.ema_vol_sec + self.alpha_vol * float(vol)
        # --- EMA de taxa de ordens ---
        self.ema_order_rate = (1.0 - self.alpha_rate) * self.ema_order_rate + self.alpha_rate * float(n_orders)

        # --- Atualiza buffers de preço ---
        if close is not None:
            self.buf_5m.append(close)
            self.buf_30m.append(close)
            self.buf_5m = self.buf_5m[-300:]      # 5 min a 1 Hz
            self.buf_30m = self.buf_30m[-1800:]   # 30 min

        # -- Atualiza buffers de preço com max/min do segundo ---
        if max_a is not None and min_a is not None:
            self.buf_max_5m.append(max_a)
            self.buf_min_5m.append(min_a)
            self.buf_max_30m.append(max_a)
            self.buf_min_30m.append(min_a)
            self.buf_max_5m = self.buf_max_5m[-300:]
            self.buf_min_5m = self.buf_min_5m[-300:]
            self.buf_max_30m = self.buf_max_30m[-1800:]
            self.buf_min_30m = self.buf_min_30m[-1800:]

        # --- VWAP ---
        if vol > 0 and close is not None:
            self.cum_vol += vol
            self.cum_vwap_num += close * vol
            self.vwap = self.cum_vwap_num / max(self.cum_vol, 1e-12)

        # --- Media de volume ---
        self.tempo_total += 1
        self.ema_vol_total = float(self.cum_vol) / self.tempo_total

        # --- Buffers de volume e ordens ---
        self.buf_vol_5m.append(vol)
        self.buf_vol_30m.append(vol)
        self.buf_vol_5m = self.buf_vol_5m[-300:]
        self.buf_vol_30m = self.buf_vol_30m[-1800:]

        self.buf_ord_5m.append(n_orders)
        self.buf_ord_30m.append(n_orders)
        self.buf_ord_5m = self.buf_ord_5m[-300:]
        self.buf_ord_30m = self.buf_ord_30m[-1800:]

        # --- Médias simples ---
        self.mean_price_5m = float(np.mean(self.buf_5m)) if self.buf_5m else 0.0
        self.mean_price_30m = float(np.mean(self.buf_30m)) if self.buf_30m else 0.0
        self.mean_vol_5m = float(np.mean(self.buf_vol_5m)) if self.buf_vol_5m else 0.0
        self.mean_vol_30m = float(np.mean(self.buf_vol_30m)) if self.buf_vol_30m else 0.0
        self.mean_ord_5m = float(np.mean(self.buf_ord_5m)) if self.buf_ord_5m else 0.0
        self.mean_ord_30m = float(np.mean(self.buf_ord_30m)) if self.buf_ord_30m else 0.0

        # --- Range máximo (5m e 30m) ---
        def calc_range(buf_max, buf_min):
            return (max(buf_max) - min(buf_min)) if len(buf_max) >= 2 else 0.0

        self.range_5m = calc_range(self.buf_max_5m, self.buf_min_5m)
        self.range_30m = calc_range(self.buf_max_30m, self.buf_min_30m)

        # --- Retorno log por segundo (se possível) ---
        ret = None
        if close is not None and self.prev_close is not None and self.prev_close > 0:
            ret = math.log(close / self.prev_close)

        # --- EMA da variância e SMAs de volatilidade ---
        if ret is not None:
            r2 = ret * ret
            # EMA da variância
            self.var_ema = (1.0 - self.alpha_var) * self.var_ema + self.alpha_var * r2
            self.vol_ema = math.sqrt(max(self.var_ema, 0.0))

            # Buffers janelados (5m/30m)
            self.buf_ret2_5m.append(r2)
            self.buf_ret2_30m.append(r2)
            self.buf_ret2_5m  = self.buf_ret2_5m[-300:]
            self.buf_ret2_30m = self.buf_ret2_30m[-1800:]

            # Vols SMA (desvio-padrão da janela)
            self.vol_sma_5m  = math.sqrt(float(np.mean(self.buf_ret2_5m)))  if self.buf_ret2_5m  else 0.0
            self.vol_sma_30m = math.sqrt(float(np.mean(self.buf_ret2_30m))) if self.buf_ret2_30m else 0.0

        # --- Speed: |Δpreço| por segundo ---
        if close is not None and self.prev_close is not None:
            dpx = abs(float(max_a) - float(min_a))
            dpx = max(dpx, abs(float(self.prev_close) - float(min_a)))
            dpx = max(dpx, abs(float(self.prev_close) - float(max_a)))
            # EMA rápida de speed (reaproveita alpha_var como time-constant para "velocidade")
            self.ema_speed_sec = (1.0 - self.alpha_var) * self.ema_speed_sec + self.alpha_var * dpx

            # buffers janelados
            self.buf_speed_5m.append(dpx)
            self.buf_speed_30m.append(dpx)
            self.buf_speed_5m  = self.buf_speed_5m[-300:]
            self.buf_speed_30m = self.buf_speed_30m[-1800:]

            # médias simples
            self.mean_speed_5m  = float(np.mean(self.buf_speed_5m))  if self.buf_speed_5m  else 0.0
            self.mean_speed_30m = float(np.mean(self.buf_speed_30m)) if self.buf_speed_30m else 0.0

        # Atualize prev_close no fim:
        if close is not None:
            self.prev_close = close

    # >>> NOVO: tira um snapshot para o histórico
    def snapshot(self, t: int, close: Optional[float]) -> None:
        self.hist["t"].append(float(t))
        self.hist["close"].append(float(close) if close is not None else (self.hist["close"][-1] if self.hist["close"] else 0.0))
        self.hist["vwap"].append(float(self.vwap))
        self.hist["mean_price_5m"].append(float(self.mean_price_5m))
        self.hist["mean_price_30m"].append(float(self.mean_price_30m))
        self.hist["ema_vol_sec"].append(float(self.ema_vol_sec))
        self.hist["mean_vol_5m"].append(float(self.mean_vol_5m))
        self.hist["mean_vol_30m"].append(float(self.mean_vol_30m))
        self.hist["ema_order_rate"].append(float(self.ema_order_rate))
        self.hist["mean_ord_5m"].append(float(self.mean_ord_5m))
        self.hist["mean_ord_30m"].append(float(self.mean_ord_30m))
        self.hist["range_5m"].append(float(self.range_5m))
        self.hist["range_30m"].append(float(self.range_30m))
        self.hist["vol_ema"].append(float(self.vol_ema))
        self.hist["vol_sma_5m"].append(float(self.vol_sma_5m))
        self.hist["vol_sma_30m"].append(float(self.vol_sma_30m))
        self.hist["speed_ema_sec"].append(float(self.ema_speed_sec))
        self.hist["mean_speed_5m"].append(float(self.mean_speed_5m))
        self.hist["mean_speed_30m"].append(float(self.mean_speed_30m))

@dataclass
class EventSegmenter:

    p: SegParams
    metrics: RefMetrics = field(init=False)
    evt: Optional[Dict[str, object]] = None
    calm_streak: int = 0
    last_event_end: Optional[int] = None
    last_close_a: Optional[float] = None
    last_close_b: Optional[float] = None
    run_sign_a: int = 0
    run_ticks_a: float = 0.0
    run_secs_a: int = 0
    run_sign_b: int = 0
    run_ticks_b: float = 0.0
    run_secs_b: int = 0
    speed_price_streak: int = 0
    last_boundary_at: Optional[int] = None
    _last_order_a: Optional[Dict[str, float]] = None
    _last_order_b: Optional[Dict[str, float]] = None
    rate_hi_streak: int = 0
    speed_price_streak_internal: int = 0
    teste: int = 0
    t_close_hist: list = field(default_factory=list)
    close_a_hist: list = field(default_factory=list)

    player_state_a: Dict[int, Dict[str, float]] = field(default_factory=dict)
    player_state_b: Dict[int, Dict[str, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.metrics = RefMetrics.create(self.p.tau_vol_min, self.p.tau_var_min, self.p.tau_order_min)
        # Decide configuração do Renko conforme ativo
        if self.p.tick_a <= 1.0:       # Dólar / WDO / DOL
            brick = 2.0
        else:                          # Índice / WIN / IND
            brick = 50.0

        self.renko2 = RenkoBuilder(brick_size=brick, reversal_bricks=2)

    def reset(self) -> None:
        """Reset the segmenter state. Call at start of a new day."""
        self.metrics = RefMetrics.create(self.p.tau_vol_min, self.p.tau_var_min, self.p.tau_order_min)
        self.evt = None
        self.calm_streak = 0
        self.last_event_end = None
        self.last_close_a = None
        self.last_close_b = None
        self.run_sign_a = self.run_ticks_a = self.run_secs_a = 0
        self.run_sign_b = self.run_ticks_b = self.run_secs_b = 0
        self.speed_price_streak = 0
        self.last_boundary_at = None
        self._last_order_a = None
        self._last_order_b = None
        self.rate_hi_streak = 0
        self.speed_price_streak_internal = 0
        # Clear per-player state
        self.player_state_a.clear()
        self.player_state_b.clear()

    def _update_player_positions(self, lote: float, preco: float, agc: int, agv: int, symbol: str) -> None:

        sym_is_a = (str(symbol).lower() == "a")
        state = self.player_state_a if sym_is_a else self.player_state_b
        tick = float(self.p.tick_a) if sym_is_a else float(self.p.tick_b)
        tick_value = 5.0 if sym_is_a else 25.0 # R$/tick por lote

        def _apply(agent_id: int, delta_pos: float) -> None:
            if agent_id is None or agent_id < 0 or delta_pos == 0.0:
                return

            entry = state.get(agent_id)
            if entry is None:
                entry = {"pos": 0.0, "avg_price": 0.0, "profit": 0.0}
                state[agent_id] = entry

            pos_old = float(entry["pos"])
            avg_old = float(entry["avg_price"])
            prof_old = float(entry["profit"])

            pos_new = pos_old + delta_pos

            # Caso 1: abre posição ou aumenta magnitude na mesma direção (mantém média ponderada)
            if pos_old == 0.0 or (pos_old > 0.0 and delta_pos > 0.0) or (pos_old < 0.0 and delta_pos < 0.0):
                if abs(pos_old) < 1e-6:
                    avg_new = float(preco) - (prof_old*tick / (delta_pos*tick_value))
                    prof_tot = 0.0
                else:
                    avg_new = ((avg_old * abs(pos_old)) + (float(preco) * abs(delta_pos))) / (abs(pos_old) + abs(delta_pos))
                    prof_tot = prof_old

            # Caso 2: reduz magnitude (realiza lucro/prejuízo) ou inverte
            else:
                closing_qty = min(abs(pos_old), abs(delta_pos))
                sign_old = 1.0 if pos_old > 0.0 else -1.0
                prof_inc = (float(preco) - avg_old) * sign_old * closing_qty * tick_value / tick
                
                leftover = abs(delta_pos) - closing_qty  # sobra que inverte direção
                if leftover > 1e-6:
                    # Inverteu: base passa a ser o preço da nova posição
                    avg_new = float(preco) - ((prof_old + prof_inc) * tick) / (pos_new * tick_value)
                    prof_tot = 0.0
                else:
                    if abs(pos_new) > 1e-6:
                        # Preço médio de ZERAGEM incorporando lucro acumulado
                        avg_new = avg_old - (prof_inc * tick) / (tick_value * pos_new)
                        prof_tot = 0.0
                    else:
                        avg_new = 0.0  # zerou a posição
                        prof_tot = prof_old + prof_inc

            entry["pos"] = pos_new
            entry["avg_price"] = avg_new
            entry["profit"] = prof_tot

        # Aplica comprador (+lote) e vendedor (-lote)
        _apply(int(agc), float(lote))
        _apply(int(agv), -float(lote))

    def aggregate_orders_by_player(self, trades: List[List[float]], symbol) -> List[Dict[str, float]]:
        out: List[Dict[str, float]] = []
        cur: Optional[Dict[str, float]] = None
        for tr in trades:
            preco = float(tr[0])
            lote = float(tr[1])
            agc = int(tr[2])
            agv = int(tr[3])
            aggr = int(tr[4])
            # Determine side based on aggressor code (1=buy, 2=sell, other=neutral)
            if aggr == 1:
                side = 1
            elif aggr == 2:
                side = -1
            else:
                side = 0

            self._update_player_positions(lote, preco, agc, agv, symbol)
            # Identify the broker on the active side
            agag = agc if side == 1 else (agv if side == -1 else -1)
            if side == 0:
                # Neutral trades break any current group
                if cur:
                    out.append(cur)
                    cur = None
                continue
            if cur and side == cur["side"] and agag == cur["broker"]:
                # Continue the current group
                cur["lot"] += lote
                cur["p_close"] = preco
                cur["pmin"] = min(cur["pmin"], preco)
                cur["pmax"] = max(cur["pmax"], preco)
                cur["n"] += 1
            else:
                # Start a new group, closing the previous one if present
                if cur:
                    out.append(cur)
                cur = {
                    "side": side,
                    "broker": agag,
                    "lot": lote,
                    "p_open": preco,
                    "p_close": preco,
                    "pmin": preco,
                    "pmax": preco,
                    "n": 1,
                }
        # Append any pending group at the end
        if cur:
            out.append(cur)
        return out

    def step(self, s: int, trades_a_raw: List[List[float]], trades_b_raw: List[List[float]]) -> List[Dict[str, object]]:
        # Apply lot multipliers specified in params
        trades_a = [[t[0], (t[1] * self.p.lot_mult_a)] + t[2:] for t in trades_a_raw]
        trades_b = [[t[0], (t[1] * self.p.lot_mult_b)] + t[2:] for t in trades_b_raw]

        # Summaries for each symbol
        smry_a = summarize_second(trades_a)
        smry_b = summarize_second(trades_b)

        open_a = self.last_close_a if self.last_close_a is not None else trades_a[0][0] if trades_a else None
        open_b = self.last_close_b if self.last_close_b is not None else trades_b[0][0] if trades_b else None

        close_a = smry_a["close"] if smry_a["close"] is not None else self.last_close_a if self.last_close_a is not None else self.last_close_b
        close_b = smry_b["close"] if smry_b["close"] is not None else self.last_close_b if self.last_close_b is not None else self.last_close_a

        vol_a = smry_a["vol"]
        vol_b = smry_b["vol"]

        a_orders = smry_a["n"] 
        b_orders = smry_b["n"]

        min_a = smry_a["pmin"] if smry_a["pmin"] is not None else self.last_close_a
        max_a = smry_a["pmax"] if smry_a["pmax"] is not None else self.last_close_a
        min_b = smry_b["pmin"] if smry_b["pmin"] is not None else self.last_close_b
        max_b = smry_b["pmax"] if smry_b["pmax"] is not None else self.last_close_b

        min_g = min(min_a, min_b) if min_a is not None and min_b is not None else (min_a if min_a is not None else min_b)
        max_g = max(max_a, max_b) if max_a is not None and max_b is not None else (max_a if max_a is not None else max_b)

        orders_a = self.aggregate_orders_by_player(trades_a, symbol="a")
        orders_b = self.aggregate_orders_by_player(trades_b, symbol="b")

        vol_comb = float(vol_a + vol_b)
        n_orders = int(a_orders + b_orders)

        # Update global baselines using the combined volume and order rate
        self.metrics.update(close_a, vol_comb, n_orders, max_a, min_a)

        events_out: List[Dict[str, object]] = []

        # --- Renko tick-a-tick (2p) + inversão ---
        feed_renko_from_trades(self.renko2, trades_a_raw, price_idx=0, t_seconds=float(s))
        # if close_a is not None:
        #     self.t_close_hist.append(s)
        #     self.close_a_hist.append(close_a)
        #     # plot_renko_realtime_with_close(self.renko2, self.t_close_hist, self.close_a_hist, title="Renko 2p + close_a")


        inv_hit = self.renko2.pull_inversion_hit()

        # self.metrics.snapshot(s, close_a)

        # --- Abrir primeiro evento se necessário (houve trade neste segundo) ---
        if self.evt is None and (a_orders + b_orders) > 0:
            self.metrics.setopenday(trades_a[0][0] if len(trades_a) else None,trades_b[0][0] if len(trades_b) else None)
            self.evt = {
                "start": s,
                "end": s,
                "dur": 1,
                "open": open_a if open_a is not None else open_b,
                "min": min_g,
                "max": max_g,
                "close": close_a if close_a is not None else close_b,
                "vol": vol_comb,
                "n_trades": n_orders,
                "range": (max_g - min_g) if (min_g is not None and max_g is not None) else 0.0,
                "tt_a": trades_a_raw,
                "tt_b": trades_b_raw,
                "end_reason": [],
            }
            # buffers/accumulators do evento (PNL, speed, etc.)
            self.evt["acc"] = event_acc_init()
            self.evt.setdefault("speed_win_s", int(self.p.speed_win_s))  # p/ speed

            # acumuladores “ricos” do seu evento
            event_accumulate_second(
                self.evt["acc"],
                trades_a, trades_b,
                smry_a, smry_b,
                close_a, close_b,
                self.last_close_a, self.last_close_b,
                self.p.tick_a, self.p.tick_b,
                smry_a["n"], smry_b["n"],
                self.p,
            )

            # atualiza last closes e sai (evento aberto)
            self.last_close_a, self.last_close_b = close_a, close_b
            return events_out

        # se continua sem evento, só atualiza last closes e sai
        if self.evt is None:
            self.last_close_a, self.last_close_b = close_a, close_b
            return events_out

        # --- acumular dados do segundo no evento aberto ---
        # tempo/duração
        self.evt["end"] = s
        self.evt["dur"] = self.evt["end"] - self.evt["start"] + 1

        # extremos/range
        self.evt["min"] = min(self.evt["min"], min_g)
        self.evt["max"] = max(self.evt["max"], max_g)
        self.evt["range"] = float(self.evt["max"] - self.evt["min"])
        self.evt["close"] = close_a if close_a is not None else close_b

        # volume e ordens
        self.evt["vol"]      += vol_comb
        self.evt["n_trades"] += n_orders

        # tt
        self.evt["tt_a"].extend(trades_a_raw)
        self.evt["tt_b"].extend(trades_b_raw)

        # acumuladores “ricos” do seu evento
        event_accumulate_second(
            self.evt["acc"],
            trades_a, trades_b,
            smry_a, smry_b,
            close_a, close_b,
            self.last_close_a, self.last_close_b,
            self.p.tick_a, self.p.tick_b,
            smry_a["n"], smry_b["n"],
            self.p,
            player_state_a=self.player_state_a,
            player_state_b=self.player_state_b,
        )

        # --- DETECTORES ---
        end_by_time_pre  = (self.evt["dur"] >= self.p.max_dur_s)

        player_big_hit   = detect_player_big_hit(orders_a, orders_b, self.metrics, self.p)

        # volume: use a função simplificada que você adotou (instante + média do evento)
        vol_spike_hit    = detect_vol_spike_two_conditions(
            vol_inst=vol_comb,
            vol_cum=self.evt["vol"],                 # passe o evento para olhar média do evento (evt["vol"]/evt["dur"])
            metrics=self.metrics,
            params=self.p,
        )

        # range dinâmico (usa range do evento e duração)
        range_cut        = detect_range_hit(
            range=self.evt["range"],
            dur=self.evt["dur"],
            metrics=self.metrics,
            params=self.p,
        )

        # speed (janela curta dentro do evento vs baseline 5m)
        speed_price_cut  = detect_speed_hit(self.metrics, self.evt, self.p, min_a, max_a, self.last_close_a)

        # order rate: segundo atual OU taxa média do evento
        order_rate_cut   = detect_order_rate_cut_simple(self.metrics, self.evt, n_orders, self.p)

        cause_flags = {
            "inversion": inv_hit,
            "player":    player_big_hit,
            "vol":       vol_spike_hit,
            "range":     range_cut,
            "speed":     speed_price_cut,
            "rate":      order_rate_cut,
            "time":      end_by_time_pre,
        }
        cause_order = ["inversion", "player", "vol", "range", "speed", "rate", "time"]

        # --- decidir fechamento ---
        end_causes = pick_end_causes(cause_flags, cause_order)
        has_cut = bool(end_causes)

        # Se acendeu corte e ainda não há "graça" pendente, agenda encerrar em s + end_grace_s
        if has_cut and not self.evt.get("grace_pending", False):
            self.evt["grace_pending"] = True
            self.evt["grace_until"]   = int(s) + int(getattr(self.p, "end_grace_s", 1))
            self.evt["end_primary"]   = end_causes[0]  # se quiser registrar a principal
            push_end_reasons(self.evt, end_causes)     # salva TODAS as flags ativas no s do corte

        # Se já está em "graça" e chegamos no segundo-agendado, FINALIZA agora (após acumular s)
        finalize_now = bool(self.evt.get("grace_pending")) and (int(s) >= int(self.evt.get("grace_until", s+1)))

        if finalize_now:
            # Também registra flags que permaneceram ativas no segundo de encerramento
            if has_cut:
                push_end_reasons(self.evt, end_causes)

            # finalizar/enriquecer + emitir
            self.evt = finalize_event_enrichment(
                self.evt, self.p, self.metrics, self.p.tick_a, self.p.tick_b, extrair_preco
            )
            events_out.append(self.evt)

            # abrir novo evento imediatamente no mesmo segundo (se houver atividade)
            self.evt = {
                "start": s,
                "end": s,
                "dur": 1,
                "open": close_a,
                "min": close_a,
                "max": close_a,
                "close": close_a,
                "vol": 0.0,
                "n_trades": 0.0,
                "range": 0.0,
                "tt_a": [],
                "tt_b": [],
                "end_reason": [],
            }
            self.evt["acc"] = event_acc_init()
            self.evt.setdefault("speed_win_s", int(self.p.speed_win_s))

        # --- atualizar últimos closes e sair ---
        self.last_close_a, self.last_close_b = close_a, close_b
        return events_out