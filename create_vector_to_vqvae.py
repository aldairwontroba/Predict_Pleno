# realtime_segmenter.py
from math import exp
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

# ----------------- Config (knobs principais) -----------------
@dataclass
class SegParams:
    # ticks e pesos
    tick_a: float = 0.5      # WDO/WIN
    tick_b: float = 0.5      # DOL/IND
    lot_mult_a: float = 1.0  # WDO/WIN
    lot_mult_b: float = 5.0  # DOL/IND

    # EMAs (10–30 min p/ convergir)
    tau_vol_min: float = 7.0
    tau_var_min: float = 12.0

    # inversão (fronteira)
    inv_k_up: float = 1.0
    inv_k_down: float = 0.7
    inv_min_ticks: int = 2
    inv_min_sec: int = 1
    inv_vol_frac: float = 0.6
    inv_cooldown_s: int = 2
    inv_both_signs: bool = False

    # baseline de ticks/s para inversão
    ticks_thr_mult: int = 6
    ticks_thr_min: int = 2

    # calmaria / velocidade / range (fecham fronteira)
    max_dur_s: int = 40
    var_rate_mult_start: float = 8.0
    var_rate_mult_end: float = 3.4
    range_mult_start: float = 1.0
    range_mult_end: float = 0.6

    # GATES / CONSISTÊNCIA
    min_event_sec_speed: int = 6   # idade mínima do evento p/ poder cortar por speed
    min_event_sec_calm:  int = 20   # idem para calmaria
    speed_consec_sec:    int = 3   # precisa de speed alto em 2s seguidos

    # COOLDOWN GLOBAL DE FRONTEIRA
    boundary_cooldown_s: int = 0   # tempo mínimo entre cortes (exceto time_pre)

    # (opcional p/ reduzir vol/calm ainda mais)
    calm_consec_sec: int = 10      # era 1
    start_vol_mult:  float = 4.5  # era 1.8
    vol_1s_hard:     float = 1600  # era 300

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

# ----------------- Estado das referências -----------------

@dataclass
class RefMetrics:
    alpha_vol: float
    alpha_var: float
    ema_vol_sec: float = 1.0
    ema_var5m: float = 1.0
    buf_a: List[float] = field(default_factory=list)
    buf_b: List[float] = field(default_factory=list)

    @classmethod
    def create(cls, tau_vol_min: float, tau_var_min: float) -> "RefMetrics":
        a_vol = 1.0 - exp(-1.0 / (tau_vol_min * 60.0))
        a_var = 1.0 - exp(-1.0 / (tau_var_min * 60.0))
        return cls(a_vol, a_var)

    def update(self, close_a: Optional[float], close_b: Optional[float], vol_comb: float):
        self.ema_vol_sec = (1.0 - self.alpha_vol) * self.ema_vol_sec + self.alpha_vol * float(vol_comb)
        if close_a is not None:
            self.buf_a.append(float(close_a)); self.buf_a = self.buf_a[-300:]
        if close_b is not None:
            self.buf_b.append(float(close_b)); self.buf_b = self.buf_b[-300:]
        var5_a = (max(self.buf_a) - min(self.buf_a)) if len(self.buf_a)>=2 else 0.0
        var5_b = (max(self.buf_b) - min(self.buf_b)) if len(self.buf_b)>=2 else 0.0
        self.ema_var5m = (1.0 - self.alpha_var) * self.ema_var5m + self.alpha_var * max(var5_a, var5_b)

# ----------------- Detector -----------------
def safe_min(a, b):
    if a is None: return b
    if b is None: return a
    return a if a < b else b

def safe_max(a, b):
    if a is None: return b
    if b is None: return a
    return a if a > b else b

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

    def __post_init__(self):
        self.metrics = RefMetrics.create(self.p.tau_vol_min, self.p.tau_var_min)

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
        # aplica multiplicadores de lote e faz resumos
        trades_a = [ [t[0], (t[1]*self.p.lot_mult_a)] + t[2:] for t in trades_a_raw ]
        trades_b = [ [t[0], (t[1]*self.p.lot_mult_b)] + t[2:] for t in trades_b_raw ]

        smry_a = summarize_second(trades_a)
        smry_b = summarize_second(trades_b)

        # closes “atuais”
        close_a = smry_a["close"] if smry_a["close"] is not None else self.last_close_a
        close_b = smry_b["close"] if smry_b["close"] is not None else self.last_close_b
        vol_comb = smry_a["vol"] + smry_b["vol"]

        # atualiza referências
        self.metrics.update(close_a, close_b, vol_comb)
        ema_var5m_ticks = max(self.metrics.ema_var5m / max(self.p.tick_a,1e-9),
                              self.metrics.ema_var5m / max(self.p.tick_b,1e-9))
        # baseline para inversão
        base_ticks_per_s = max(ema_var5m_ticks / 300.0, 1e-9)
        ticks_thr = max(self.p.ticks_thr_min, int(round(base_ticks_per_s * self.p.ticks_thr_mult)))

        # deltas de 1s em ticks (close-to-close)
        dt_a = int(round(ticks_move(self.last_close_a, close_a, self.p.tick_a))) if self.last_close_a is not None and close_a is not None else 0
        dt_b = int(round(ticks_move(self.last_close_b, close_b, self.p.tick_b))) if self.last_close_b is not None and close_b is not None else 0

        # atualiza runs (e detecta flips)
        def upd_run(dt, run_sign, run_ticks, run_secs):
            if dt > 0:
                if run_sign >= 0:  return (+1, run_ticks + dt, run_secs + 1, False)
                else:              return (+1, dt, 1, True)
            elif dt < 0:
                if run_sign <= 0:  return (-1, run_ticks + (-dt), run_secs + 1, False)
                else:              return (-1, -dt, 1, True)
            else:
                return (run_sign, run_ticks, run_secs, False)

        new_sign_a, new_ticks_a, new_secs_a, flipped_a = upd_run(dt_a, self.run_sign_a, self.run_ticks_a, self.run_secs_a)
        new_sign_b, new_ticks_b, new_secs_b, flipped_b = upd_run(dt_b, self.run_sign_b, self.run_ticks_b, self.run_secs_b)

        def inv_ok(prev_ticks, prev_secs, step_now_ticks, vol_now, thr):
            need_u = max(self.p.inv_min_ticks, int(round(self.p.inv_k_up   * thr)))
            need_d = max(self.p.inv_min_ticks, int(round(self.p.inv_k_down * thr)))
            has_u  = (prev_ticks >= need_u) and (prev_secs >= self.p.inv_min_sec)
            has_d  = (abs(step_now_ticks) >= need_d)
            has_v  = (vol_now >= self.p.inv_vol_frac * self.metrics.ema_vol_sec)
            return has_u and has_d and has_v

        cooldown_ok = (self.evt is not None) or (self.last_event_end is None) or ((s - self.last_event_end) >= self.p.inv_cooldown_s)
        inv_start = False
        if cooldown_ok:
            if flipped_a and inv_ok(self.run_ticks_a, self.run_secs_a, dt_a, vol_comb, ticks_thr): inv_start = True
            if not inv_start and flipped_b and inv_ok(self.run_ticks_b, self.run_secs_b, dt_b, vol_comb, ticks_thr): inv_start = True
            if self.p.inv_both_signs and not (flipped_a and flipped_b): inv_start = False

        # aplica novos runs
        self.run_sign_a, self.run_ticks_a, self.run_secs_a = new_sign_a, new_ticks_a, new_secs_a
        self.run_sign_b, self.run_ticks_b, self.run_secs_b = new_sign_b, new_ticks_b, new_secs_b

        # helpers locais
        def _fallback_price(primary, *alts):
            for v in (primary, *alts):
                if v is not None:
                    return float(v)
            return None  # só se realmente não houver nada

        # snapshot do segundo (para abrir/estender)
        open_a = _fallback_price(smry_a["open"], close_a, self.last_close_a, (self.evt["plast_a"] if self.evt else None))
        open_b = _fallback_price(smry_b["open"], close_b, self.last_close_b, (self.evt["plast_b"] if self.evt else None))
        pmin_a = _fallback_price(smry_a["pmin"], close_a, self.last_close_a, (self.evt["plast_a"] if self.evt else None))
        pmax_a = _fallback_price(smry_a["pmax"], close_a, self.last_close_a, (self.evt["plast_a"] if self.evt else None))
        pmin_b = _fallback_price(smry_b["pmin"], close_b, self.last_close_b, (self.evt["plast_b"] if self.evt else None))
        pmax_b = _fallback_price(smry_b["pmax"], close_b, self.last_close_b, (self.evt["plast_b"] if self.evt else None))

        snap = {
            "open_a": open_a, "open_b": open_b,
            "pmin_a": pmin_a, "pmax_a": pmax_a,
            "pmin_b": pmin_b, "pmax_b": pmax_b,
            "plast_a": close_a if close_a is not None else (self.last_close_a if self.last_close_a is not None else open_a),
            "plast_b": close_b if close_b is not None else (self.last_close_b if self.last_close_b is not None else open_b),
            "vol": vol_comb,
            "n_trades": smry_a["n"] + smry_b["n"],
            # player agregado (A+B)
            "max_player_lot": max([o["lot"] for o in aggregate_orders_by_player(trades_a)] + [0.0]) +
                            max([o["lot"] for o in aggregate_orders_by_player(trades_b)] + [0.0]),
        }

        def _safe_span(ev_min, ev_max, snap_min, snap_max):
            vals = []
            if ev_min  is not None: vals.append(ev_min)
            if ev_max  is not None: vals.append(ev_max)
            if snap_min is not None: vals.append(snap_min)
            if snap_max is not None: vals.append(snap_max)
            if len(vals) >= 2:
                return max(vals) - min(vals)
            return 0.0
        
        # inicia primeiro evento quando vier o primeiro segundo com trade
        events_out = []
        if self.evt is None and snap["n_trades"] > 0:
            self.evt = {
                "start": s, "end": s, "dur": 1,
                "p0_a": snap["open_a"], "p0_b": snap["open_b"],
                "pmin_a": snap["pmin_a"] if snap["pmin_a"] is not None else snap["plast_a"],
                "pmax_a": snap["pmax_a"] if snap["pmax_a"] is not None else snap["plast_a"],
                "pmin_b": snap["pmin_b"] if snap["pmin_b"] is not None else snap["plast_b"],
                "pmax_b": snap["pmax_b"] if snap["pmax_b"] is not None else snap["plast_b"],
                "plast_a": snap["plast_a"], "plast_b": snap["plast_b"],
                "vol": snap["vol"], "n_trades": snap["n_trades"],
                "ema_vol_start": self.metrics.ema_vol_sec,
                "ema_var5m_start_ticks": ema_var5m_ticks,
            }
            self.calm_streak = 0
            self.speed_hi_streak = 0
            self.last_close_a, self.last_close_b = close_a, close_b
            return events_out  # nada a fechar ainda

        if self.evt is None:
            # continua sem evento (ainda não começou o dia)
            self.last_close_a, self.last_close_b = close_a, close_b
            return events_out

        # --------- decisão de fronteira (corte) ---------
        # 1) tempo, player, volume, inversão
        end_by_time_pre = (self.evt["dur"] + 1) > self.p.max_dur_s
        player_big_hit  = (snap["max_player_lot"] >= self.p.start_vol_mult * self.metrics.ema_vol_sec)
        vol_spike_hit   = (snap["vol"] >= self.p.start_vol_mult * self.metrics.ema_vol_sec) or (snap["vol"] >= self.p.vol_1s_hard)
        inv_hit         = inv_start

        # 2) calmaria — com gate de idade
        t_evt = self.evt["dur"]  # ainda não agregamos s
        vol_thresh_now = linear_decay(
            t_evt,
            self.p.start_vol_mult * self.evt["ema_vol_start"],
            1.0 * self.evt["ema_vol_start"],
            self.p.max_dur_s
        )
        self.calm_streak = (self.calm_streak + 1) if (snap["vol"] < vol_thresh_now) else 0
        calm_cut = (t_evt >= self.p.min_event_sec_calm) and (self.calm_streak >= self.p.calm_consec_sec)

        # 3) range e speed — “e se anexar s?”
        range_a_if = _safe_span(self.evt["pmin_a"], self.evt["pmax_a"], snap["pmin_a"], snap["pmax_a"])
        range_b_if = _safe_span(self.evt["pmin_b"], self.evt["pmax_b"], snap["pmin_b"], snap["pmax_b"])
        range_t_if = max(range_a_if / max(self.p.tick_a,1e-9), range_b_if / max(self.p.tick_b,1e-9))

        da_if = abs(ticks_move(self.evt["p0_a"], snap["plast_a"], self.p.tick_a)) if self.evt["p0_a"] and snap["plast_a"] else 0.0
        db_if = abs(ticks_move(self.evt["p0_b"], snap["plast_b"], self.p.tick_b)) if self.evt["p0_b"] and snap["plast_b"] else 0.0
        d_ticks_if = max(da_if, db_if)
        speed_if   = d_ticks_if / max(self.evt["dur"] + 1, 1)

        base_rate = max(self.evt["ema_var5m_start_ticks"] / 300.0, 1e-9)
        rate_mult = linear_decay(t_evt, self.p.var_rate_mult_start, self.p.var_rate_mult_end, self.p.max_dur_s)
        speed_thr = rate_mult * base_rate
        range_thr = linear_decay(t_evt, self.p.range_mult_start, self.p.range_mult_end, self.p.max_dur_s) * max(self.evt["ema_var5m_start_ticks"], 1e-9)

        # 3a) speed com gate de idade + 2s consecutivos
        speed_hi = (speed_if >= speed_thr)
        self.speed_hi_streak = (self.speed_hi_streak + 1) if speed_hi else 0
        speed_cut = (t_evt >= self.p.min_event_sec_speed) and (self.speed_hi_streak >= self.p.speed_consec_sec)

        # 3b) range (pré-corte) direto
        range_cut = (range_t_if >= range_thr)

        # 4) COOLDOWN GLOBAL (exceto time_pre)
        cooldown_ok = (self.last_boundary_at is None) or ((s - self.last_boundary_at) >= self.p.boundary_cooldown_s)
        boundary_allowed = cooldown_ok or end_by_time_pre

        # 5) FLAGS e motivo principal (prioridade)
        cause_flags = {
            "inversion": inv_hit,
            "player":    player_big_hit,
            "vol":       vol_spike_hit,
            "speed":     speed_cut,
            "range":     range_cut,
            "calm":      calm_cut,
            "time":      end_by_time_pre,
        }
        boundary_now = boundary_allowed and any(cause_flags.values())

        cause_order = ["inversion","player","vol","speed","range","calm","time"]
        cause = next((c for c in cause_order if cause_flags[c]), None)


        if boundary_now:
            # fecha anterior em s-1
            self.evt["end"] = s - 1
            self.evt["dur"] = self.evt["end"] - self.evt["start"] + 1
            self.evt["end_reason"] = cause if cause != "time" else "time_pre"
            self.evt["d_ticks_max"] = max(
                abs(ticks_move(self.evt["p0_a"], self.evt["plast_a"], self.p.tick_a)) if self.evt["p0_a"] and self.evt["plast_a"] else 0.0,
                abs(ticks_move(self.evt["p0_b"], self.evt["plast_b"], self.p.tick_b)) if self.evt["p0_b"] and self.evt["plast_b"] else 0.0
            )
            self.evt["range_ticks"] = max(
                (self.evt["pmax_a"] - self.evt["pmin_a"]) / max(self.p.tick_a,1e-9) if self.evt["pmin_a"] is not None else 0.0,
                (self.evt["pmax_b"] - self.evt["pmin_b"]) / max(self.p.tick_b,1e-9) if self.evt["pmin_b"] is not None else 0.0
            )
            events_out.append(self.evt)
            self.last_event_end = self.evt["end"]
            self.last_boundary_at = s           # marca o cooldown
            # abre novo em s com flags do corte
            self.evt = {
                "start": s, "end": s, "dur": 1,
                "p0_a": snap["open_a"], "p0_b": snap["open_b"],
                "pmin_a": snap["pmin_a"], "pmax_a": snap["pmax_a"],
                "pmin_b": snap["pmin_b"], "pmax_b": snap["pmax_b"],
                "plast_a": snap["plast_a"], "plast_b": snap["plast_b"],
                "vol": snap["vol"], "n_trades": snap["n_trades"],
                "ema_vol_start": self.metrics.ema_vol_sec,
                "ema_var5m_start_ticks": ema_var5m_ticks,
                "start_reason": cause or "boundary",
                "start_flags": cause_flags,   # <- agora aparece no seu relatório
            }
            self.calm_streak = 0
            self.speed_hi_streak = 0
        else:
            # agrega s no evento atual
            self.evt["end"] = s
            self.evt["dur"] = self.evt["end"] - self.evt["start"] + 1

            # A (símbolo A)
            if smry_a["pmin"] is not None or smry_a["pmax"] is not None:
                self.evt["pmin_a"] = safe_min(self.evt["pmin_a"], smry_a["pmin"])
                self.evt["pmax_a"] = safe_max(self.evt["pmax_a"], smry_a["pmax"])
            if close_a is not None:
                self.evt["plast_a"] = close_a

            # B (símbolo B)
            if smry_b["pmin"] is not None or smry_b["pmax"] is not None:
                self.evt["pmin_b"] = safe_min(self.evt["pmin_b"], smry_b["pmin"])
                self.evt["pmax_b"] = safe_max(self.evt["pmax_b"], smry_b["pmax"])
            if close_b is not None:
                self.evt["plast_b"] = close_b

            self.evt["vol"]      += snap["vol"]
            self.evt["n_trades"] += snap["n_trades"]

        # atualiza closes
        self.last_close_a, self.last_close_b = close_a, close_b
        return events_out

from pathlib import Path
import numpy as np
import re

DATA_DIR  = Path(r"E:\Mercado BMF&BOVESPA\tryd\consolidados_npz")
DAY       = "20240424"
PAIR      = ("wdo", "dol")   # ou ("win","ind")

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
    # (ajuste os knobs se quiser)
    # start_vol_mult=1.8, vol_1s_hard=300.0,
    # inv_k_up=1.6, inv_k_down=1.2, inv_min_ticks=2, inv_min_sec=2,
    # ticks_thr_mult=10, ticks_thr_min=2,
    # max_dur_s=30, calm_consec_sec=1, range_mult_start=1.5, ...
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
import math
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from datetime import datetime

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

def _event_stats_arrays(eventos_list):
    if not eventos_list:
        return [], [], [], [], []
    dur    = [int(ev["dur"]) for ev in eventos_list]
    vol    = [float(ev["vol"]) for ev in eventos_list]
    rng_t  = [float(ev.get("range_ticks", 0.0)) for ev in eventos_list]
    spd    = [ (float(ev.get("d_ticks_max",0.0)) / max(1, int(ev["dur"]))) for ev in eventos_list ]  # ticks/s
    reason = [ev.get("end_reason","") for ev in eventos_list]
    return dur, vol, rng_t, spd, reason

dur, vol, rng_t, spd, reason = _event_stats_arrays(eventos)

if len(dur) == 0:
    print("Sem eventos para estatística.")
else:
    # hist de duração (s)
    plt.figure(figsize=(10,4))
    plt.hist(dur, bins=range(1, 32), edgecolor="none")
    plt.title("Eventos — Duração (s)")
    plt.xlabel("segundos"); plt.ylabel("contagem")
    plt.tight_layout(); plt.show()

    # hist de volume (log-bins)
    vmin, vmax = max(1.0, min(vol)), max(vol)
    nb = 40
    bins = np.logspace(math.log10(vmin), math.log10(vmax+1e-9), nb)
    plt.figure(figsize=(10,4))
    plt.hist(vol, bins=bins)
    plt.xscale("log")
    plt.title("Eventos — Volume ponderado (log)")
    plt.xlabel("volume"); plt.ylabel("contagem")
    plt.tight_layout(); plt.show()

    # scatter duração x range (ticks), colorindo por motivo de encerramento
    color_map = {"time":"C1","calm":"C2","speed":"C3","range":"C4","":"C0"}
    colors = [color_map.get(r,"C0") for r in reason]
    plt.figure(figsize=(10,4))
    plt.scatter(dur, rng_t, s=8, c=colors, alpha=0.7)
    plt.title("Eventos — Duração vs Range (ticks)")
    plt.xlabel("duração (s)"); plt.ylabel("range (ticks)")
    plt.tight_layout(); plt.show()

    # scatter duração x velocidade média (ticks/s)
    plt.figure(figsize=(10,4))
    plt.scatter(dur, spd, s=8, c=colors, alpha=0.7)
    plt.title("Eventos — Duração vs Velocidade média (ticks/s)")
    plt.xlabel("duração (s)"); plt.ylabel("ticks/s")
    plt.tight_layout(); plt.show()


# ================== PLOT TICK-A-TICK (tempo real no eixo X) ==================
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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
            xs_time.append(datetime.fromtimestamp(t_real))  # usa horário local da máquina
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