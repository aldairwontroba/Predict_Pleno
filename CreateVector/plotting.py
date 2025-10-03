"""
Plotting utilities for inspecting event segmentation results.

This module groups together plotting functions that generate various
visualisations from event DataFrames. These functions are intended to
be called interactively by users to explore the distribution of event
metrics, relationships between variables, and the distribution of
events across time of day and trigger types. Additional helper
functions facilitate plotting tick‑by‑tick price series coloured by
event ID.
"""

from __future__ import annotations

from typing import Dict, Tuple, List, Iterable, Optional, Any
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import pandas as pd
from datetime import datetime, timezone
import matplotlib.dates as mdates

__all__ = [
    "plot_event_distributions",
    "plot_relationships",
    "plot_counts_by_hour_and_reason",
    "print_top_outliers",
    "plot_tick_series",
]


def plot_event_distributions(df: pd.DataFrame) -> None:
    """Plot histograms and ECDFs of basic event metrics.

    Generates a series of plots summarising the distribution of event
    durations, volumes, ranges and speeds. If the DataFrame is empty
    the function prints a message and returns.
    """
    if df.empty:
        print("Sem eventos no DataFrame.")
        return
    # Duration histogram
    plt.figure(figsize=(10, 4))
    vals = df["dur_s"].dropna()
    bins = np.arange(1, max(60, int(vals.max()) + 2))
    plt.hist(vals, bins=bins, edgecolor="none")
    plt.title("Eventos — Duração (s)")
    plt.xlabel("segundos")
    plt.ylabel("contagem")
    plt.tight_layout()
    plt.show()
    # Volume histogram (log)
    if "vol" in df and df["vol"].notna().any():
        v = df["vol"].dropna()
        v = v[v > 0]
        if len(v) > 0:
            nb = 40
            vmin, vmax = max(1.0, v.min()), v.max()
            bins = np.logspace(np.log10(vmin), np.log10(vmax + 1e-9), nb)
            plt.figure(figsize=(10, 4))
            plt.hist(v, bins=bins)
            plt.xscale("log")
            plt.title("Eventos — Volume ponderado (log)")
            plt.xlabel("volume")
            plt.ylabel("contagem")
            plt.tight_layout()
            plt.show()
    # Range histogram
    if "range_ticks" in df:
        plt.figure(figsize=(10, 4))
        plt.hist(df["range_ticks"].dropna(), bins=50)
        plt.title("Eventos — Range (ticks)")
        plt.xlabel("ticks")
        plt.ylabel("contagem")
        plt.tight_layout()
        plt.show()
    # Speed histogram
    if "ticks_per_s" in df:
        plt.figure(figsize=(10, 4))
        plt.hist(df["ticks_per_s"].dropna(), bins=50)
        plt.title("Eventos — Velocidade média (ticks/s)")
        plt.xlabel("ticks/s")
        plt.ylabel("contagem")
        plt.tight_layout()
        plt.show()
    # ECDF helper
    def _plot_ecdf(series: pd.Series, title: str, xlabel: str) -> None:
        x = np.sort(series.dropna().values)
        if len(x) == 0:
            return
        y = np.arange(1, len(x) + 1) / len(x)
        plt.figure(figsize=(8, 4))
        plt.plot(x, y, drawstyle="steps-post")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("ECDF")
        plt.tight_layout()
        plt.show()
    _plot_ecdf(df["dur_s"], "ECDF — Duração", "segundos")
    if "vol" in df:
        # Add small constant to avoid log10 of zero
        _plot_ecdf(np.log10(df["vol"].replace(0, np.nan)), "ECDF — log10(Volume)", "log10(volume)")
    if "range_ticks" in df:
        _plot_ecdf(df["range_ticks"], "ECDF — Range (ticks)", "ticks")


def plot_relationships(df: pd.DataFrame) -> None:
    """Plot relationships between selected event metrics.

    Generates hexbin plots for volume vs range and volume vs speed. Also
    produces boxplots of key metrics grouped by the reason for event
    termination. If the DataFrame is empty the function returns.
    """
    if df.empty:
        return
    if {"vol", "range_ticks"}.issubset(df.columns):
        plt.figure(figsize=(6, 5))
        plt.hexbin(df["vol"], df["range_ticks"], gridsize=40, mincnt=1)
        plt.xscale("log")
        plt.title("Hexbin — Volume vs Range (ticks)")
        plt.xlabel("volume (log)")
        plt.ylabel("range (ticks)")
        plt.tight_layout()
        plt.show()
    if {"vol", "ticks_per_s"}.issubset(df.columns):
        plt.figure(figsize=(6, 5))
        plt.hexbin(df["vol"], df["ticks_per_s"], gridsize=40, mincnt=1)
        plt.xscale("log")
        plt.title("Hexbin — Volume vs Velocidade (ticks/s)")
        plt.xlabel("volume (log)")
        plt.ylabel("ticks/s")
        plt.tight_layout()
        plt.show()
    if "end_reason" in df.columns:
        for metric in ["dur_s", "vol", "range_ticks", "ticks_per_s", "n_trades"]:
            if metric in df.columns:
                sub = df[["end_reason", metric]].dropna()
                if sub.empty:
                    continue
                groups = [g[metric].values for _, g in sub.groupby("end_reason")]
                labels = [str(k) for k, _ in sub.groupby("end_reason")]
                plt.figure(figsize=(10, 4))
                plt.boxplot(groups, labels=labels, showfliers=False)
                plt.title(f"Boxplot — {metric} por end_reason")
                plt.xlabel("end_reason")
                plt.ylabel(metric)
                plt.tight_layout()
                plt.show()


def plot_counts_by_hour_and_reason(df: pd.DataFrame) -> None:
    """Plot counts of events by hour of day and termination reason."""
    if df.empty or "hour" not in df.columns:
        return
    per_hour = df.groupby("hour").size().reindex(range(0, 24), fill_value=0)
    plt.figure(figsize=(10, 3))
    plt.bar(per_hour.index, per_hour.values)
    plt.title("Eventos por hora")
    plt.xlabel("hora")
    plt.ylabel("contagem")
    plt.tight_layout()
    plt.show()
    if "end_reason" in df.columns:
        pivot = df.pivot_table(index="hour", columns="end_reason", values="start_ts", aggfunc="count").fillna(0).reindex(range(0, 24), fill_value=0)
        pivot.plot(kind="bar", stacked=True, figsize=(12, 4))
        plt.title("Eventos por hora (stacked por end_reason)")
        plt.xlabel("hora")
        plt.ylabel("contagem")
        plt.tight_layout()
        plt.show()
        cnt = df["end_reason"].value_counts()
        plt.figure(figsize=(8, 3))
        plt.bar(cnt.index.astype(str), cnt.values)
        plt.title("Eventos por end_reason")
        plt.xlabel("end_reason")
        plt.ylabel("contagem")
        plt.tight_layout()
        plt.show()


def print_top_outliers(df: pd.DataFrame, top: int = 10) -> None:
    """Print tables of the top events by selected metrics."""
    def _show(title: str, ser: pd.Series) -> None:
        print(f"\n== Top {top} por {title} ==")
        if ser.empty:
            print("(vazio)")
            return
        idx = ser.sort_values(ascending=False).head(top).index
        cols = [
            "start_time",
            "end_time",
            "dur_s",
            "end_reason",
            "vol",
            "range_ticks",
            "ticks_per_s",
            "n_trades",
            "p0_a",
            "plast_a",
            "pmin_a",
            "pmax_a",
        ]
        cols = [c for c in cols if c in df.columns]
        print(df.loc[idx, cols].to_string(index=False))
    if "range_ticks" in df:
        _show("range_ticks", df["range_ticks"].dropna())
    if "vol" in df:
        _show("volume", df["vol"].dropna())
    if "ticks_per_s" in df:
        _show("ticks_per_s", df["ticks_per_s"].dropna())
    if "n_trades" in df:
        _show("n_trades", df["n_trades"].dropna())


def _build_event_sec_map(eventos_list: List[Dict[str, Any]]) -> Dict[int, int]:
    """Map each second to an event index."""
    evmap: Dict[int, int] = {}
    for k, ev in enumerate(eventos_list):
        for s in range(int(ev["start"]), int(ev["end"]) + 1):
            evmap[s] = k
    return evmap


def _tick_series_time(sec_map: Dict[int, List[List[float]]], evmap: Dict[int, int]) -> Tuple[List[datetime], List[float], List[int]]:
    """Build time series (datetime, price, event ID) for tick‑by‑tick plotting."""
    xs_time: List[datetime] = []
    ys: List[float] = []
    ev_ids: List[int] = []
    for s in sorted(sec_map.keys()):
        trades = sec_map[s]
        n = len(trades)
        if n == 0:
            continue
        for k, tr in enumerate(trades):
            frac = (k + 1) / (n + 1)
            t_real = s + frac
            dt_utc = datetime.fromtimestamp(t_real, tz=timezone.utc)
            xs_time.append(dt_utc)
            ys.append(float(tr[0]))
            ev_ids.append(evmap.get(s, -1))
    return xs_time, ys, ev_ids


def _plot_segments_time(xs: List[datetime], ys: List[float], ev_ids: List[int], title: str) -> None:
    """Plot tick‑by‑tick price series coloured by event ID."""
    if not xs:
        print("Sem dados para plot.")
        return
    segments: List[Tuple[List[datetime], List[float], int]] = []
    start = 0
    last = ev_ids[0]
    for i in range(1, len(xs)):
        if ev_ids[i] != last:
            segments.append((xs[start:i], ys[start:i], last))
            start = i
            last = ev_ids[i]
    segments.append((xs[start:], ys[start:], last))
    fig, ax = plt.subplots(figsize=(14, 5))
    for sx, sy, _eid in segments:
        ax.plot(sx, sy, linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("tempo")
    ax.set_ylabel("preço")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()


def plot_tick_series(
    sec_price: Dict[int, List[List[float]]],
    eventos: List[Dict[str, Any]],
    title: Optional[str] = None,
    symbol_label: Optional[str] = None,
) -> None:
    """High‑level helper to plot tick‑by‑tick price series coloured by events."""
    evmap = _build_event_sec_map(eventos)
    xs_time, ys_price, ev_ids = _tick_series_time(sec_price, evmap)
    label = symbol_label or "Preço"
    ttl = title or f"Tick-a-tick — {label} por evento"
    _plot_segments_time(xs_time, ys_price, ev_ids, ttl)


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