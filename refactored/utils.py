"""
Utility functions for event segmentation and data processing.

This module contains helper functions used throughout the segmentation
pipeline. They include functions for summarizing trades on a per-second
basis, aggregating trades by player, computing moving statistics,
coalescing orders, and extracting periodic price features.

The functions defined here are largely adapted from the original
`create_vector_to_vqvae.py` script but have been cleaned up and
documented for clarity. The goal is to separate pure computation from
higher‑level classes so that they can be independently unit tested and
reused.
"""

from __future__ import annotations

from collections import defaultdict
from typing import List, Tuple, Dict, Optional, Iterable
import math
import numpy as np

__all__ = [
    "summarize_second",
    "aggregate_orders_by_player",
    "_vol_by_aggr",
    "_group_stats_one_symbol",
    "_bucket_price_stats_wdo",
    "coalesce_orders_one_sec",
    "ticks_delta",
    "extrair_preco",
]


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


def aggregate_orders_by_player(trades: List[List[float]]) -> List[Dict[str, float]]:
    """Aggregate contiguous trades by player and side within a second.

    Trades are expected in the form ``[price, lot, broker_buy, broker_sell, aggressor, ...]``.
    The aggressor column indicates the side: positive for buy, negative for sell,
    and zero for neutral trades. For each contiguous sequence of trades with
    the same side and broker, this function returns an aggregate record with
    the total lot and price boundaries.
    """
    out: List[Dict[str, float]] = []
    cur: Optional[Dict[str, float]] = None
    for tr in trades:
        preco = float(tr[0])
        lote = float(tr[1]) if len(tr) > 1 else 1.0
        aggr = int(tr[4]) if len(tr) > 4 else 0
        side = 1 if aggr > 0 else (-1 if aggr < 0 else 0)
        broker = int(tr[2]) if side == 1 else (int(tr[3]) if side == -1 else -1)
        if cur and side == cur["side"] and broker == cur["broker"]:
            # Continue current group
            cur["lot"] += lote
            cur["p_close"] = preco
            cur["pmin"] = min(cur["pmin"], preco)
            cur["pmax"] = max(cur["pmax"], preco)
            cur["n"] += 1
        else:
            # Close previous group and start a new one
            if cur:
                out.append(cur)
            cur = {
                "side": side,
                "broker": broker,
                "lot": lote,
                "p_open": preco,
                "p_close": preco,
                "pmin": preco,
                "pmax": preco,
                "n": 1,
            }
    if cur:
        out.append(cur)
    return out


def _vol_by_aggr(trades: Iterable[Iterable[float]]) -> Tuple[float, float, float]:
    """Return the total buy, sell and total volume for a list of trades.

    Each trade row should include lot size and an aggressor flag. The
    aggressor flag (index 4) determines whether the trade counts towards
    buy or sell volume. Trades without an aggressor flag are treated as
    neutral volume.
    """
    buy = sell = tot = 0.0
    for tr in trades:
        lot = float(tr[1]) if len(tr) > 1 else 1.0
        ag = int(tr[4]) if len(tr) > 4 else 0
        tot += lot
        if ag > 0:
            buy += lot
        elif ag < 0:
            sell += lot
    return buy, sell, tot


def _group_stats_one_symbol(trades: List[List[float]], tick: float) -> Tuple[float, float, int, int, int]:
    """Compute group statistics for a single symbol within a second.

    Returns a tuple containing:
        max_lot: largest aggregated lot in this second
        max_ticks: maximum number of ticks moved by any group
        g_buy: count of buy side groups
        g_sell: count of sell side groups
        g_neutral: count of neutral groups
    """
    groups = aggregate_orders_by_player(trades)
    max_lot = 0.0
    max_ticks = 0.0
    g_buy = g_sell = g_neu = 0
    for g in groups:
        max_lot = max(max_lot, float(g["lot"]))
        dtk = abs(ticks_delta(g.get("p_open"), g.get("p_close"), tick))
        max_ticks = max(max_ticks, dtk)
        if g["side"] > 0:
            g_buy += 1
        elif g["side"] < 0:
            g_sell += 1
        else:
            g_neu += 1
    return max_lot, max_ticks, g_buy, g_sell, g_neu


def _bucket_price_stats_wdo(trades_a: List[List[float]], close_a: Optional[float]) -> Tuple[Dict[float, int], Dict[float, float], Optional[float]]:
    """Accumulate per‑price statistics for the WDO symbol.

    For WDO (symbol 'a'), this function computes three histograms:
    - price_count: number of trades at each price
    - price_vol: volume at each price
    - secs_at_price: seconds spent at each closing price
    """
    price_count: Dict[float, int] = defaultdict(int)
    price_vol: Dict[float, float] = defaultdict(float)
    for tr in trades_a:
        p = float(tr[0])
        lot = float(tr[1]) if len(tr) > 1 else 1.0
        price_count[p] += 1
        price_vol[p] += lot
    sec_price = None if close_a is None else float(close_a)
    return price_count, price_vol, sec_price


def coalesce_orders_one_sec(trades: List[List[float]], last_open: Optional[Dict[str, float]]) -> Tuple[int, Optional[Dict[str, float]]]:
    """Approximate the number of distinct orders started within a second.

    Uses broker and side information to estimate when a new order begins.
    Returns the number of new orders and the last open order to be carried
    over to the next second.
    """
    starts = 0
    cur = last_open  # may be carried over from previous second

    def _side(tr: List[float]) -> int:
        a = int(tr[4]) if len(tr) > 4 else 0
        return 1 if a > 0 else (-1 if a < 0 else 0)

    def _broker(tr: List[float], side: int) -> int:
        return int(tr[2]) if side == 1 else (int(tr[3]) if side == -1 else -1)

    for tr in trades:
        s = _side(tr)
        if s == 0:
            # Neutral trade ends any current order
            if cur:
                cur = None
            continue
        b = _broker(tr, s)
        if cur and cur["side"] == s and cur["broker"] == b:
            # Continue current order
            cur["lot"] += float(tr[1]) if len(tr) > 1 else 1.0
        else:
            # New order
            starts += 1
            cur = {"side": s, "broker": b, "lot": float(tr[1]) if len(tr) > 1 else 1.0}
    return starts, cur


def ticks_delta(p0: Optional[float], p1: Optional[float], tick: float) -> float:
    """Compute the difference between two prices in ticks.
    Returns 0 if either price is ``None`` or tick is non‑positive.
    """
    if p0 is None or p1 is None or tick <= 0:
        return 0.0
    return (float(p1) - float(p0)) / float(tick)


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