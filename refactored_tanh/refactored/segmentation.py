"""
Core classes and functions for segmenting trade data into events.

This module encapsulates the event segmentation logic originally found
in `create_vector_to_vqvae.py`. The goal is to provide a clean API
for processing streams of trade data and generating enriched event
records. The code has been broken down into smaller functions and
documented for clarity. It depends on the helper functions defined in
``refactored.utils`` for summarising trades, grouping orders and
extracting price features.

Users should instantiate :class:`EventSegmenter` with a
``SegParams`` instance. Then call :meth:`step` with the timestamp and
trades for each second. The returned list of events contains all
events that ended during that call. At the end of a trading day the
last event (if any) should be closed manually by setting its
``end_reason`` and appending it to the output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Iterable
from collections import defaultdict, Counter
import math
import numpy as np

import utils

__all__ = [
    "SegParams",
    "RefMetrics",
    "EventSegmenter",
    "event_accumulate_second",
    "finalize_event_enrichment",
]


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
    tau_vol_min: float = 5.0
    tau_var_min: float = 2.0
    var_win_sec: int = 120
    tau_order_min: float = 5.0

    # Cooldown time between event boundaries
    boundary_cooldown_s: int = 0

    # Volume and player thresholds
    start_vol_mult: float = 5.0
    vol_1s_hard: float = 2000.0
    player_mult: float = 5.0

    # Range decay thresholds
    range_start_mult: float = 1.0
    range_end_mult: float = 0.4
    half_life_frac: float = 1.0 / 3.0

    # Maximum event duration (seconds)
    max_dur_s: int = 60

    # Renko settings
    renko_nticks: int = 2
    renko_symbol: str = "a"

    # Order rate thresholds
    tau_order_rate_min: float = 8.0
    order_rate_mult: float = 1.5
    order_rate_consec_sec: int = 2

    # Speed thresholds
    speed_price_mult: float = 20.0
    speed_price_consec_sec: int = 3
    min_event_sec_speed: int = 3

    # Player vs volume threshold
    player_mult_vs_vol_ema: float = 5.0

    # Range thresholds with decay during the event
    range_mult_start: float = 1.5
    range_mult_end: float = 0.8
    decay_half_life_frac: float = 1.0 / 3.0

    # Absorption detection thresholds
    absorb_ticks_thr: int = 2
    absorb_max_wait_s: int = 10

    # Which symbol to use as the price reference ("a" for WDO/WIN or "b" for DOL/IND)
    price_ref_symbol: str = "a"

    # Top players of the day (optional)
    top_buyers_a: Tuple[int, ...] = ()
    top_sellers_a: Tuple[int, ...] = ()
    top_buyers_b: Tuple[int, ...] = ()
    top_sellers_b: Tuple[int, ...] = ()


def exp_decay_mult(t: int, start_mult: float, end_mult: float, half_life_sec: float) -> float:
    """Exponential decay multiplier.

    Returns a value that decays from ``start_mult`` to ``end_mult`` with a
    half‑life of ``half_life_sec`` seconds. When ``t`` is zero the
    ``start_mult`` is returned; for large ``t`` the value asymptotically
    approaches ``end_mult``.
    """
    if t <= 0:
        return start_mult
    k = 0.5 ** (t / max(half_life_sec, 1e-9))
    return end_mult + (start_mult - end_mult) * k


def renko_reversal_this_second(
    anchor: Optional[float],
    direction: int,
    pmin: Optional[float],
    pmax: Optional[float],
    brick: float,
) -> Tuple[bool, Optional[float], int]:
    """Detect a Renko reversal within a second.

    Given the current Renko anchor (last close), direction and the
    second's high/low, determine whether a reversal has occurred.
    Returns a tuple ``(reversal_bool, new_anchor, new_dir)``. If no
    reversal is detected the anchor and direction are updated to absorb
    any full bricks moved in the same direction.
    """
    if anchor is None or pmin is None or pmax is None:
        return False, anchor, direction

    rev = False
    # Check for reversal: move two bricks in the opposite direction
    if direction >= 0:
        # fall: two bricks down from anchor
        if pmin <= anchor - 2 * brick:
            rev = True
            anchor = anchor - brick
            direction = -1
    if direction <= 0 and not rev:
        # rise: two bricks up from anchor
        if pmax >= anchor + 2 * brick:
            rev = True
            anchor = anchor + brick
            direction = 1

    # Absorb extra bricks in the same direction
    if direction == 1 and pmax is not None:
        n_up = int((pmax - anchor) // brick)
        if n_up > 0:
            anchor += n_up * brick
    if direction == -1 and pmin is not None:
        n_dn = int((anchor - pmin) // brick)
        if n_dn > 0:
            anchor -= n_dn * brick

    return rev, anchor, direction


def event_acc_init() -> Dict[str, object]:
    """Initialise an accumulator for per‑event statistics.

    Returns a dictionary with keys corresponding to various metrics that
    accumulate over the life of an event. This dictionary is attached to
    the event under the ``"acc"`` key and is updated by
    :func:`event_accumulate_second`.
    """
    return {
        # price stats (WDO)
        "price_count": defaultdict(int),
        "price_vol": defaultdict(float),
        "secs_at_price": defaultdict(int),
        # group stats
        "max_group_lot": 0.0,
        "max_group_ticks": 0.0,
        # volume stats
        "buy_vol_a": 0.0,
        "sell_vol_a": 0.0,
        "tot_vol_a": 0.0,
        "buy_vol_b": 0.0,
        "sell_vol_b": 0.0,
        "tot_vol_b": 0.0,
        # group counts
        "g_buy": 0,
        "g_sell": 0,
        "g_neu": 0,
        # order rate
        "orders_started_sum": 0.0,
        "rate_max": 0.0,
        # streaks
        "streak_buy": 0,
        "streak_sell": 0,
        "max_streak_buy": 0,
        "max_streak_sell": 0,
        # per second movement
        "persec": [],
        # absorption tracking (based on reference symbol)
        "run_sign": 0,
        "run_ticks": 0.0,
        "run_secs": 0,
        "run_dir_vol": 0.0,
        "retrace_ticks": 0.0,
        "retrace_secs": 0,
        "absorb_buy": [],
        "absorb_sell": [],
        # top player volumes
        "top_buy_vol_a": 0.0,
        "top_sell_vol_a": 0.0,
        "top_buy_vol_b": 0.0,
        "top_sell_vol_b": 0.0,
    }


@dataclass
class RefMetrics:
    """Exponential moving averages used as global baselines.

    This object tracks EMAs of volume, range and order rate. It also
    keeps rolling buffers of closing prices used to estimate five‑minute
    range baselines. Each call to :meth:`update` updates the EMAs and
    pushes prices into the buffers. Rolling buffers are trimmed to
    maintain a maximum length.
    """

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
        # Convert time constants in minutes to per‑second alphas
        a_vol = 1.0 - math.exp(-1.0 / max(1.0, tau_vol_min * 60.0))
        a_var = 1.0 - math.exp(-1.0 / max(1.0, tau_var_min * 60.0))
        a_rate = 1.0 - math.exp(-1.0 / max(1.0, tau_order_min * 60.0))
        return cls(a_vol, a_var, a_rate)

    def update(self,
               close_a: Optional[float],
               close_b: Optional[float],
               vol_comb: float,
               orders_started_this_sec: float) -> None:
        """Update moving averages and rolling buffers for this second."""
        # Volume EMA
        self.ema_vol_sec = (1.0 - self.alpha_vol) * self.ema_vol_sec + self.alpha_vol * float(vol_comb)
        # Append closing prices for range estimation (approx 5 minutes at 1 Hz)
        if close_a is not None:
            self.buf_a.append(float(close_a))
            self.buf_a = self.buf_a[-300:]
        if close_b is not None:
            self.buf_b.append(float(close_b))
            self.buf_b = self.buf_b[-300:]
        # 5‑minute range baseline (max difference between buffered prices)
        var5_a = (max(self.buf_a) - min(self.buf_a)) if len(self.buf_a) >= 2 else 0.0
        var5_b = (max(self.buf_b) - min(self.buf_b)) if len(self.buf_b) >= 2 else 0.0
        self.ema_var5m = (1.0 - self.alpha_var) * self.ema_var5m + self.alpha_var * max(var5_a, var5_b)
        # Order rate EMA
        self.ema_order_rate = (1.0 - self.alpha_rate) * self.ema_order_rate + self.alpha_rate * float(orders_started_this_sec)


def _safe_mean(seq: Iterable[float], fallback: Optional[float] = None) -> float:
    """Compute the mean of a sequence, filtering out non‑finite values.
    If no valid values are present, returns ``fallback`` if it is finite
    or ``0.0`` otherwise."""
    vals = []
    for v in seq or []:
        try:
            f = float(v)
        except Exception:
            continue
        if math.isfinite(f):
            vals.append(f)
    if vals:
        return float(sum(vals) / len(vals))
    fb = None
    try:
        fb = float(fallback)
    except Exception:
        fb = None
    return float(fb) if (fb is not None and math.isfinite(fb)) else 0.0


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
    p: SegParams,
) -> None:
    """Accumulate detailed statistics for the current second into ``ev_acc``.

    This function updates the provided accumulator dictionary with
    per‑second metrics such as price histograms, volume breakdown,
    group counts, streaks, per‑second movement and absorption statistics.
    It is called once per second while an event is active. See the
    original code for detailed explanations of each metric.
    """
    # Price histograms (WDO only)
    price_count, price_vol, sec_price = utils._bucket_price_stats_wdo(trades_a, close_a)
    for k, v in price_count.items():
        ev_acc["price_count"][k] += v
    for k, v in price_vol.items():
        ev_acc["price_vol"][k] += v
    if sec_price is not None:
        ev_acc["secs_at_price"][sec_price] += 1
    # Volume by aggressor
    buy_a, sell_a, tot_a = utils._vol_by_aggr(trades_a)
    buy_b, sell_b, tot_b = utils._vol_by_aggr(trades_b)
    ev_acc["buy_vol_a"] += buy_a
    ev_acc["sell_vol_a"] += sell_a
    ev_acc["tot_vol_a"] += tot_a
    ev_acc["buy_vol_b"] += buy_b
    ev_acc["sell_vol_b"] += sell_b
    ev_acc["tot_vol_b"] += tot_b
    # Group stats per symbol
    mlot_a, mtick_a, g_buy_a, g_sell_a, g_neu_a = utils._group_stats_one_symbol(trades_a, tick_a)
    mlot_b, mtick_b, g_buy_b, g_sell_b, g_neu_b = utils._group_stats_one_symbol(trades_b, tick_b)
    ev_acc["max_group_lot"] = max(ev_acc["max_group_lot"], mlot_a, mlot_b)
    ev_acc["max_group_ticks"] = max(ev_acc["max_group_ticks"], mtick_a, mtick_b)
    ev_acc["g_buy"] += (g_buy_a + g_buy_b)
    ev_acc["g_sell"] += (g_sell_a + g_sell_b)
    ev_acc["g_neu"] += (g_neu_a + g_neu_b)
    # Order rate
    rate_now = float(starts_a + starts_b)
    ev_acc["orders_started_sum"] += rate_now
    ev_acc["rate_max"] = max(ev_acc["rate_max"], rate_now)
    # Streaks per second (dominant side by group counts)
    if (g_buy_a + g_buy_b) > (g_sell_a + g_sell_b):
        ev_acc["streak_buy"] += 1
        ev_acc["streak_sell"] = 0
    elif (g_sell_a + g_sell_b) > (g_buy_a + g_buy_b):
        ev_acc["streak_sell"] += 1
        ev_acc["streak_buy"] = 0
    else:
        ev_acc["streak_buy"] = 0
        ev_acc["streak_sell"] = 0
    ev_acc["max_streak_buy"] = max(ev_acc["max_streak_buy"], ev_acc["streak_buy"])
    ev_acc["max_streak_sell"] = max(ev_acc["max_streak_sell"], ev_acc["streak_sell"])
    # Per‑second movement (reference symbol)
    use_a = (p.price_ref_symbol.lower() == "a")
    dtk_a = utils.ticks_delta(last_close_a, close_a, tick_a)
    dtk_b = utils.ticks_delta(last_close_b, close_b, tick_b)
    dtk_ref = dtk_a if use_a else dtk_b
    # Directional volumes for reference symbol
    dir_buy_vol = buy_a + buy_b
    dir_sell_vol = sell_a + sell_b
    ev_acc["persec"].append({
        "dtk_a": dtk_a,
        "dtk_b": dtk_b,
        "buy_a": buy_a,
        "sell_a": sell_a,
        "buy_b": buy_b,
        "sell_b": sell_b,
        "dtk_ref": dtk_ref,
    })
    # Absorption detection based on reference symbol
    sgn = 1 if dtk_ref > 0 else (-1 if dtk_ref < 0 else 0)
    if sgn == 0:
        pass  # no change
    elif ev_acc["run_sign"] == 0:
        # Start new leg
        ev_acc["run_sign"] = sgn
        ev_acc["run_ticks"] = abs(dtk_ref)
        ev_acc["run_secs"] = 1
        ev_acc["retrace_ticks"] = 0.0
        ev_acc["retrace_secs"] = 0
        ev_acc["run_dir_vol"] = dir_buy_vol if sgn > 0 else dir_sell_vol
    elif sgn == ev_acc["run_sign"]:
        # Continue current leg
        ev_acc["run_ticks"] += abs(dtk_ref)
        ev_acc["run_secs"] += 1
        ev_acc["run_dir_vol"] += dir_buy_vol if sgn > 0 else dir_sell_vol
        # Reset retrace
        ev_acc["retrace_ticks"] = 0.0
        ev_acc["retrace_secs"] = 0
    else:
        # Retrace in opposite direction
        ev_acc["retrace_ticks"] += abs(dtk_ref)
        ev_acc["retrace_secs"] += 1
        if ev_acc["retrace_ticks"] >= p.absorb_ticks_thr and ev_acc["run_ticks"] >= p.absorb_ticks_thr:
            rec = {"ticks": ev_acc["retrace_ticks"], "secs": ev_acc["retrace_secs"]}
            if ev_acc["run_sign"] > 0:
                ev_acc["absorb_sell"].append(rec)
            else:
                ev_acc["absorb_buy"].append(rec)
            # Reset and start new leg
            ev_acc["run_sign"] = sgn
            ev_acc["run_ticks"] = abs(dtk_ref)
            ev_acc["run_secs"] = 1
            ev_acc["run_dir_vol"] = dir_buy_vol if sgn > 0 else dir_sell_vol
            ev_acc["retrace_ticks"] = 0.0
            ev_acc["retrace_secs"] = 0
    # Top player volumes (if provided in parameters)
    if p.top_buyers_a or p.top_sellers_a:
        for tr in trades_a:
            lot = float(tr[1]) if len(tr) > 1 else 1.0
            ag = int(tr[4]) if len(tr) > 4 else 0
            b_comp = int(tr[2]) if len(tr) > 2 else -1
            b_vend = int(tr[3]) if len(tr) > 3 else -1
            if ag > 0 and b_comp in p.top_buyers_a:
                ev_acc["top_buy_vol_a"] += lot
            if ag < 0 and b_vend in p.top_sellers_a:
                ev_acc["top_sell_vol_a"] += lot
    if p.top_buyers_b or p.top_sellers_b:
        for tr in trades_b:
            lot = float(tr[1]) if len(tr) > 1 else 1.0
            ag = int(tr[4]) if len(tr) > 4 else 0
            b_comp = int(tr[2]) if len(tr) > 2 else -1
            b_vend = int(tr[3]) if len(tr) > 3 else -1
            if ag > 0 and b_comp in p.top_buyers_b:
                ev_acc["top_buy_vol_b"] += lot
            if ag < 0 and b_vend in p.top_sellers_b:
                ev_acc["top_sell_vol_b"] += lot


def finalize_event_enrichment(
    ev: Dict[str, object],
    p: SegParams,
    metrics: RefMetrics,
    tick_a: float,
    tick_b: float,
    extrair_preco_fn = utils.extrair_preco,
) -> Dict[str, object]:
    """Enrich a completed event with derived metrics.

    This function takes an event dictionary and its accumulator
    (attached under ``"acc"``), computes summary statistics and
    context information, then stores them under separate keys. After
    enrichment the accumulator is removed from the event.

    Returns the enriched event.
    """
    acc = ev.pop("acc", None)
    if not acc:
        return ev
    # Helper to find the key with the maximum value in a dict
    def _argmax_in_dict(dct: Dict[float, float]) -> Tuple[Optional[float], float]:
        if not dct:
            return None, 0.0
        k = max(dct.items(), key=lambda kv: kv[1])[0]
        return k, dct[k]
    price_most_traded, _ = _argmax_in_dict(acc["price_count"])
    price_most_volume, _ = _argmax_in_dict(acc["price_vol"])
    price_most_time, _ = _argmax_in_dict(acc["secs_at_price"])
    ev["price_info"] = {
        "open": ev.get("p0_a"),
        "close": ev.get("plast_a"),
        "high": ev.get("pmax_a"),
        "low": ev.get("pmin_a"),
        "price_most_traded": price_most_traded,
        "price_most_volume": price_most_volume,
        "price_most_time": price_most_time,
        "max_group_ticks": acc["max_group_ticks"],
    }
    # Volume info
    buy_a, sell_a, tot_a = acc["buy_vol_a"], acc["sell_vol_a"], acc["tot_vol_a"]
    buy_b, sell_b, tot_b = acc["buy_vol_b"], acc["sell_vol_b"], acc["tot_vol_b"]
    tot_all = tot_a + tot_b
    n_orders = max(1.0, float(acc["orders_started_sum"]))
    ev["volume_info"] = {
        "buy_vol": buy_a + buy_b,
        "sell_vol": sell_a + sell_b,
        "total_vol_incl_cross": tot_all,
        "avg_vol_per_order": tot_all / n_orders,
        "max_group_vol": acc["max_group_lot"],
    }
    # Trade info
    ev["trade_info"] = {
        "groups_buy": acc["g_buy"],
        "groups_sell": acc["g_sell"],
        "groups_neutral": acc["g_neu"],
        "rate_avg": acc["orders_started_sum"] / max(1, int(ev["dur"])),
        "rate_max": acc["rate_max"],
        "max_streak_buy": acc["max_streak_buy"],
        "max_streak_sell": acc["max_streak_sell"],
    }
    # Movement info
    def _abs_dtk(p0: Optional[float], p1: Optional[float], tk: float) -> float:
        if p0 is None or p1 is None or not tk or tk <= 0:
            return 0.0
        return abs((float(p1) - float(p0)) / float(tk))
    dtk_a = _abs_dtk(ev.get("p0_a"), ev.get("plast_a"), tick_a)
    dtk_b = _abs_dtk(ev.get("p0_b"), ev.get("plast_b"), tick_b)
    def _vol_per_tick_stats(persec: List[Dict[str, float]], symbol: str) -> Tuple[Dict[str, float], Dict[str, float]]:
        vals_up: List[float] = []
        vals_dn: List[float] = []
        for r in persec:
            if symbol == "a":
                dtk = r["dtk_a"]
                buy = r["buy_a"]
                sell = r["sell_a"]
            else:
                dtk = r["dtk_b"]
                buy = r["buy_b"]
                sell = r["sell_b"]
            adtk = abs(dtk)
            if adtk <= 0:
                continue
            if dtk > 0 and buy > 0:
                vals_up.append(buy / adtk)
            if dtk < 0 and sell > 0:
                vals_dn.append(sell / adtk)
        def _stats(lst: List[float]) -> Dict[str, object]:
            if not lst:
                return {"mean": 0.0, "min": 0.0, "max": 0.0, "n": 0}
            arr = np.asarray(lst, dtype=float)
            return {
                "mean": float(arr.mean()),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "n": int(arr.size),
            }
        return _stats(vals_up), _stats(vals_dn)
    up_a, dn_a = _vol_per_tick_stats(acc["persec"], "a")
    up_b, dn_b = _vol_per_tick_stats(acc["persec"], "b")
    # Participation of WDO in displacement (per side)
    buy_share_a = (buy_a / (buy_a + buy_b)) if (buy_a + buy_b) > 0 else 0.0
    sell_share_a = (sell_a / (sell_a + sell_b)) if (sell_a + sell_b) > 0 else 0.0
    # Absorption stats
    def _abs_stats(lst: List[Dict[str, float]]) -> Dict[str, float]:
        if not lst:
            return {"count": 0, "ticks_mean": 0.0, "ticks_max": 0.0, "vel_mean": 0.0}
        ticks = np.asarray([x["ticks"] for x in lst], dtype=float)
        vels = np.asarray([x["ticks"] / max(1, x["secs"]) for x in lst], dtype=float)
        return {
            "count": len(lst),
            "ticks_mean": float(ticks.mean()),
            "ticks_max": float(ticks.max()),
            "vel_mean": float(vels.mean()),
        }
    abs_buy = _abs_stats(acc["absorb_buy"])
    abs_sell = _abs_stats(acc["absorb_sell"])
    # Ease of movement
    use_a = (p.price_ref_symbol.lower() == "a")
    ref_tick = tick_a if use_a else tick_b
    ref_open = ev.get("p0_a") if use_a else ev.get("p0_b")
    ref_close = ev.get("plast_a") if use_a else ev.get("plast_b")
    net_ticks = _abs_dtk(ref_open, ref_close, ref_tick)
    dir_sign = 0
    if ref_open is not None and ref_close is not None:
        diff = float(ref_close) - float(ref_open)
        dir_sign = 1 if diff > 0 else (-1 if diff < 0 else 0)
    opp_vol = (sell_a + sell_b) if dir_sign > 0 else ((buy_a + buy_b) if dir_sign < 0 else 0.0)
    ease_of_move = (net_ticks / max(1.0, opp_vol)) if dir_sign != 0 else 0.0
    ev["move_info"] = {
        "range_ticks": ev.get("range_ticks", 0.0),
        "d_ticks_max": ev.get("d_ticks_max", 0.0),
        "dtk_open_close_a": dtk_a,
        "dtk_open_close_b": dtk_b,
        "vol_per_tick_up_a": up_a,
        "vol_per_tick_dn_a": dn_a,
        "vol_per_tick_up_b": up_b,
        "vol_per_tick_dn_b": dn_b,
        "buy_share_wdo": buy_share_a,
        "sell_share_wdo": sell_share_a,
        "absorptions_buy": abs_buy,
        "absorptions_sell": abs_sell,
        "ease_of_move": ease_of_move,
    }
    # Players info
    ev["players_info"] = {
        "top_buy_vol_wdo": acc["top_buy_vol_a"],
        "top_sell_vol_wdo": acc["top_sell_vol_a"],
        "top_buy_vol_dol": acc["top_buy_vol_b"],
        "top_sell_vol_dol": acc["top_sell_vol_b"],
    }
    # Context info
    use_a = (p.price_ref_symbol.lower() == "a")
    ctx_price = ev.get("plast_a") if use_a else ev.get("plast_b")
    if ctx_price is None:
        ctx_price = ev.get("plast_b") if use_a else ev.get("plast_a")
    if ctx_price is None:
        ctx_price = 0.0
    if use_a:
        sma15m = _safe_mean(getattr(metrics, "buf15_a", []), ctx_price)
    else:
        sma15m = _safe_mean(getattr(metrics, "buf15_b", []), ctx_price)
    try:
        ctx_feats = extrair_preco_fn(np.array([float(ctx_price)], dtype=float))[0].tolist()
    except Exception:
        ctx_feats = [0.0] * 5
    ev["context_info"] = {
        "ctx_price": float(ctx_price),
        "ema_vol_sec": float(getattr(metrics, "ema_vol_sec", 0.0) or 0.0),
        "ema_order_rate": float(getattr(metrics, "ema_order_rate", 0.0) or 0.0),
        "sma15m_price_wdo": float(sma15m),
        "price_feats": ctx_feats,
        "active_flags": list(ev.get("start_flags", {}).keys()),
        "end_flags": ev.get("end_reason", ""),
    }
    return ev


@dataclass
class EventSegmenter:
    """Segment trade data into events based on dynamic thresholds.

    The :class:`EventSegmenter` maintains state as it processes
    trade data second by second. When certain conditions are met, it
    closes the current event and starts a new one. The implementation is
    derived from the original script but encapsulated to provide a
    simpler public API. To use the segmenter:

    1. Instantiate it with a :class:`SegParams` instance.
    2. Call :meth:`reset` before beginning a new session.
    3. For each second, call :meth:`step` with the timestamp and lists of
       trades for symbols A and B. It returns a list of events that
       ended during that second (usually zero or one).
    4. At end of day, if an event is open, close it manually by
       setting its ``end_reason`` and appending it to the output.
    """
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

    def __post_init__(self) -> None:
        self.metrics = RefMetrics.create(self.p.tau_vol_min, self.p.tau_var_min, self.p.tau_order_min)

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

    def step(self, s: int, trades_a_raw: List[List[float]], trades_b_raw: List[List[float]]) -> List[Dict[str, object]]:
        """Process trades for one second and return any completed events.

        Parameters
        ----------
        s : int
            The epoch second corresponding to the trades.
        trades_a_raw : list of list of floats
            Trades for symbol A during this second. Each row should be
            ``[price, lot, broker_buy, broker_sell, aggressor, ...]``.
        trades_b_raw : list of list of floats
            Trades for symbol B during this second.

        Returns
        -------
        list of dict
            A list of events that ended in this step. Usually zero or one
            event will be returned.
        """
        # Apply lot multipliers specified in params
        trades_a = [[t[0], (t[1] * self.p.lot_mult_a)] + t[2:] for t in trades_a_raw]
        trades_b = [[t[0], (t[1] * self.p.lot_mult_b)] + t[2:] for t in trades_b_raw]
        # Summaries for each symbol
        smry_a = utils.summarize_second(trades_a)
        smry_b = utils.summarize_second(trades_b)
        close_a = smry_a["close"] if smry_a["close"] is not None else self.last_close_a
        close_b = smry_b["close"] if smry_b["close"] is not None else self.last_close_b
        vol_comb = smry_a["vol"] + smry_b["vol"]
        # Coalesce original orders
        starts_a, self._last_order_a = utils.coalesce_orders_one_sec(trades_a, self._last_order_a)
        starts_b, self._last_order_b = utils.coalesce_orders_one_sec(trades_b, self._last_order_b)
        orders_started = starts_a + starts_b
        # Update global baselines
        self.metrics.update(close_a, close_b, vol_comb, orders_started)
        # Baseline range in ticks over ~five minutes
        ema_var5m_ticks = max(
            self.metrics.ema_var5m / max(self.p.tick_a, 1e-9),
            self.metrics.ema_var5m / max(self.p.tick_b, 1e-9),
        )
        # Renko reversal detection
        use_a_for_renko = (self.p.renko_symbol.lower() == "a")
        pmin = smry_a["pmin"] if use_a_for_renko else smry_b["pmin"]
        pmax = smry_a["pmax"] if use_a_for_renko else smry_b["pmax"]
        tick = self.p.tick_a if use_a_for_renko else self.p.tick_b
        brick = self.p.renko_nticks * max(tick, 1e-9)
        if getattr(self, "renko_anchor", None) is None:
            base = close_a if use_a_for_renko else close_b
            if base is not None:
                self.renko_anchor = round(base / brick) * brick
                self.renko_dir = 0
        inv_hit = False
        if getattr(self, "renko_anchor", None) is not None:
            inv_hit, self.renko_anchor, self.renko_dir = renko_reversal_this_second(
                self.renko_anchor, self.renko_dir, pmin, pmax, brick
            )
        # Speed per second (ticks/s)
        def _ticks_this_sec(last_close: Optional[float], close: Optional[float], tk: float) -> float:
            if last_close is None or close is None or tk <= 0:
                return 0.0
            return abs((float(close) - float(last_close)) / float(tk))
        ticks_a_1s = _ticks_this_sec(self.last_close_a, close_a, self.p.tick_a)
        ticks_b_1s = _ticks_this_sec(self.last_close_b, close_b, self.p.tick_b)
        ticks_per_s_this_sec = max(ticks_a_1s, ticks_b_1s)
        # Baseline ticks/s from range baseline over 2–5 minutes
        base_rate_now = max(ema_var5m_ticks / 120.0, 1e-9)
        speed_hi = (ticks_per_s_this_sec >= abs(self.p.speed_price_mult * base_rate_now))
        self.speed_price_streak = (self.speed_price_streak + 1) if speed_hi else 0
        speed_price_cut = (self.evt is not None) and (
            self.evt["dur"] >= self.p.min_event_sec_speed and self.speed_price_streak >= self.p.speed_price_consec_sec
        )
        # Order rate thresholds
        rate_now = float(orders_started)
        rate_hi = (rate_now >= self.p.order_rate_mult * max(self.metrics.ema_order_rate, 1e-9))
        self.rate_hi_streak = (self.rate_hi_streak + 1) if rate_hi else 0
        order_rate_cut = (self.rate_hi_streak >= self.p.order_rate_consec_sec)
        # Player vs volume thresholds
        max_player_lot = (
            max([o["lot"] for o in utils.aggregate_orders_by_player(trades_a)] + [0.0]) +
            max([o["lot"] for o in utils.aggregate_orders_by_player(trades_b)] + [0.0])
        )
        player_big_hit = (max_player_lot >= self.p.player_mult_vs_vol_ema * self.metrics.ema_vol_sec)
        # Volume spike thresholds with decay within the event
        t_evt = (self.evt["dur"] if self.evt else 0)
        hl = self.p.decay_half_life_frac * self.p.max_dur_s
        ema_vol_ref = (self.evt["ema_vol_start"] if self.evt else self.metrics.ema_vol_sec)
        vol_thr_now = exp_decay_mult(t_evt, self.p.start_vol_mult * ema_vol_ref, 1.0 * ema_vol_ref, hl)
        vol_spike_hit = (vol_comb >= vol_thr_now) or (vol_comb >= self.p.vol_1s_hard)
        # Range cut with decay within the event
        def _safe_span(ev_min: Optional[float], ev_max: Optional[float], snap_min: Optional[float], snap_max: Optional[float]) -> float:
            vals: List[float] = []
            for v in [ev_min, ev_max, snap_min, snap_max]:
                if v is not None:
                    vals.append(float(v))
            return (max(vals) - min(vals)) if len(vals) >= 2 else 0.0
        range_a_if = _safe_span(
            self.evt["pmin_a"] if self.evt else None,
            self.evt["pmax_a"] if self.evt else None,
            smry_a["pmin"],
            smry_a["pmax"],
        )
        range_b_if = _safe_span(
            self.evt["pmin_b"] if self.evt else None,
            self.evt["pmax_b"] if self.evt else None,
            smry_b["pmin"],
            smry_b["pmax"],
        )
        range_t_if = max(
            range_a_if / max(self.p.tick_a, 1e-9),
            range_b_if / max(self.p.tick_b, 1e-9),
        )
        range_baseline_ticks = max(ema_var5m_ticks, 1e-9)
        range_thr = exp_decay_mult(
            t_evt,
            self.p.range_mult_start * range_baseline_ticks,
            self.p.range_mult_end * range_baseline_ticks,
            hl,
        )
        range_cut = (range_t_if >= range_thr)
        # Max time pre‑cut
        end_by_time_pre = (self.evt is not None) and ((self.evt["dur"] + 1) > self.p.max_dur_s)
        # Check if we need to open first event
        events_out: List[Dict[str, object]] = []
        if self.evt is None and (smry_a["n"] + smry_b["n"]) > 0:
            def _fallback(primary: Optional[float], *alts: Optional[float]) -> Optional[float]:
                for v in (primary,) + alts:
                    if v is not None:
                        return float(v)
                return None
            open_a = _fallback(smry_a["open"], close_a, self.last_close_a)
            open_b = _fallback(smry_b["open"], close_b, self.last_close_b)
            self.evt = {
                "start": s,
                "end": s,
                "dur": 1,
                "p0_a": open_a,
                "p0_b": open_b,
                "pmin_a": smry_a["pmin"] if smry_a["pmin"] is not None else open_a,
                "pmax_a": smry_a["pmax"] if smry_a["pmax"] is not None else open_a,
                "pmin_b": smry_b["pmin"] if smry_b["pmin"] is not None else open_b,
                "pmax_b": smry_b["pmax"] if smry_b["pmax"] is not None else open_b,
                "plast_a": close_a if close_a is not None else open_a,
                "plast_b": close_b if close_b is not None else open_b,
                "vol": vol_comb,
                "n_trades": smry_a["n"] + smry_b["n"],
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
        # Decision for event boundary
        cooldown_ok = (self.last_boundary_at is None) or ((s - self.last_boundary_at) >= self.p.boundary_cooldown_s)
        cause_flags = {
            "inversion": inv_hit,
            "player": player_big_hit,
            "vol": vol_spike_hit,
            "range": range_cut,
            "speed": speed_price_cut,
            "rate": order_rate_cut,
            "time": end_by_time_pre,
        }
        cause_order = ["inversion", "player", "vol", "range", "speed", "rate", "time"]
        cause = next((c for c in cause_order if cause_flags[c]), None)
        boundary_now = (cooldown_ok or end_by_time_pre) and (cause is not None)
        if boundary_now:
            # Close previous event at s-1
            self.evt["end"] = s - 1
            self.evt["dur"] = self.evt["end"] - self.evt["start"] + 1
            self.evt["end_reason"] = "time_pre" if cause == "time" else cause
            # Final metrics at event end
            def _mv(p0: Optional[float], p1: Optional[float], tk: float) -> float:
                if p0 is None or p1 is None or tk <= 0:
                    return 0.0
                return abs((p1 - p0) / tk)
            self.evt["d_ticks_max"] = max(
                _mv(self.evt["p0_a"], self.evt["plast_a"], self.p.tick_a),
                _mv(self.evt["p0_b"], self.evt["plast_b"], self.p.tick_b),
            )
            self.evt["range_ticks"] = max(
                (self.evt["pmax_a"] - self.evt["pmin_a"]) / max(self.p.tick_a, 1e-9) if self.evt["pmin_a"] is not None else 0.0,
                (self.evt["pmax_b"] - self.evt["pmin_b"]) / max(self.p.tick_b, 1e-9) if self.evt["pmin_b"] is not None else 0.0,
            )
            self.evt["ticks_per_s"] = self.evt["d_ticks_max"] / max(self.evt["dur"], 1)
            self.evt = finalize_event_enrichment(
                self.evt, self.p, self.metrics, self.p.tick_a, self.p.tick_b, utils.extrair_preco
            )
            events_out.append(self.evt)
            self.last_event_end = self.evt["end"]
            self.last_boundary_at = s
            # Start new event at s
            self.evt = {
                "start": s,
                "end": s,
                "dur": 1,
                "p0_a": smry_a["open"] if smry_a["open"] is not None else close_a,
                "p0_b": smry_b["open"] if smry_b["open"] is not None else close_b,
                "pmin_a": smry_a["pmin"] if smry_a["pmin"] is not None else close_a,
                "pmax_a": smry_a["pmax"] if smry_a["pmax"] is not None else close_a,
                "pmin_b": smry_b["pmin"] if smry_b["pmin"] is not None else close_b,
                "pmax_b": smry_b["pmax"] if smry_b["pmax"] is not None else close_b,
                "plast_a": close_a,
                "plast_b": close_b,
                "vol": vol_comb,
                "n_trades": smry_a["n"] + smry_b["n"],
                "ema_vol_start": self.metrics.ema_vol_sec,
                "ema_var5m_start_ticks": ema_var5m_ticks,
                "start_reason": cause or "boundary",
                "start_flags": cause_flags,
            }
            self.evt["acc"] = event_acc_init()
            # Reset streaks
            self.speed_price_streak = 0
            self.rate_hi_streak = 0
        else:
            # Continue current event
            self.evt["end"] = s
            self.evt["dur"] = self.evt["end"] - self.evt["start"] + 1
            if smry_a["pmin"] is not None:
                self.evt["pmin_a"] = min(self.evt["pmin_a"], smry_a["pmin"])
                self.evt["pmax_a"] = max(self.evt["pmax_a"], smry_a["pmax"])
                if close_a is not None:
                    self.evt["plast_a"] = close_a
            if smry_b["pmin"] is not None:
                self.evt["pmin_b"] = min(self.evt["pmin_b"], smry_b["pmin"])
                self.evt["pmax_b"] = max(self.evt["pmax_b"], smry_b["pmax"])
                if close_b is not None:
                    self.evt["plast_b"] = close_b
            self.evt["vol"] += vol_comb
            self.evt["n_trades"] += (smry_a["n"] + smry_b["n"])
            event_accumulate_second(
                self.evt["acc"],
                trades_a,
                trades_b,
                smry_a,
                smry_b,
                close_a,
                close_b,
                self.last_close_a,
                self.last_close_b,
                self.p.tick_a,
                self.p.tick_b,
                starts_a,
                starts_b,
                self.p,
            )
        # Update last closes
        self.last_close_a, self.last_close_b = close_a, close_b
        return events_out