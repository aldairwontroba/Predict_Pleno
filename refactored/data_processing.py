"""
Data loading and processing utilities for the segmentation framework.

This module contains functions to load NPZ files containing trade data,
convert timestamps to epoch seconds, group trades by second while
preserving order, and convert event dictionaries into Pandas
DataFrames. It also provides helper functions to expand nested
structures and normalise event features for machine learning.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Iterable, Any
import numpy as np
import pandas as pd
import math

from .segmentation import EventSegmenter, SegParams
from . import utils

__all__ = [
    "lot_multiplier",
    "find_files_for_day",
    "load_npz",
    "to_epoch_seconds",
    "group_by_second_preserving_order",
    "process_day",
    "events_to_df",
    "expand_list_columns",
    "build_feature_df",
    "normalize_events_to_vectors",
]


def lot_multiplier(sym: str) -> float:
    """Return the lot multiplier for a given symbol.

    By convention DOL/IND contracts have a lot multiplier of 5, while
    WDO/WIN have a multiplier of 1. The symbol string is case‑insensitive.
    """
    return 5.0 if sym.lower() in ("dol", "ind") else 1.0


def find_files_for_day(data_dir: Path, day: str, pair: Tuple[str, str]) -> Dict[str, Path]:
    """Locate NPZ files for the requested trading day and symbol pair.

    Filenames are expected to follow the pattern ``{day}_{symbol}.npz``.
    Returns a dictionary mapping symbol names to their corresponding
    ``Path`` objects. The keys are lower‑cased.
    """
    pat = re.compile(rf"^({day})_({pair[0]}|{pair[1]})\.npz$", re.IGNORECASE)
    out: Dict[str, Path] = {}
    for p in data_dir.glob("*.npz"):
        m = pat.match(p.name)
        if m:
            out[m.group(2).lower()] = p
    return out


def load_npz(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load timestamp and trade arrays from an NPZ file.

    The NPZ is expected to contain two arrays: one representing timestamps
    (e.g. under keys ``t``, ``time``, ``datetime``, etc.) and one
    representing trade rows (under keys ``tt``, ``x``, ``data``, etc.).
    Returns a tuple ``(t, TT)``. The timestamp array may be numpy datetime
    or integer values; conversion to epoch seconds is left to
    :func:`to_epoch_seconds`.
    """
    d = np.load(path, allow_pickle=True)
    t = None
    TT = None
    for k in d.files:
        kl = k.lower()
        if kl in ("t", "time", "datetime", "timestamps", "date_time"):
            t = d[k]
        if kl in ("tt", "x", "data"):
            TT = d[k]
    if t is None or TT is None:
        raise ValueError(f"Cannot find timestamp and data arrays in {path.name}: keys {d.files}")
    return t, TT


def to_epoch_seconds(t_arr: np.ndarray) -> np.ndarray:
    """Convert various timestamp formats to integer epoch seconds.

    Handles numpy datetime64 arrays as well as integer timestamps in
    milliseconds or microseconds. If the array contains datetime64 values
    the resolution is rounded to seconds. If integer timestamps are too
    large for seconds they are scaled appropriately.
    """
    if np.issubdtype(t_arr.dtype, np.datetime64):
        return t_arr.astype("datetime64[s]").astype(np.int64)
    t = t_arr.astype(np.int64)
    mx = int(t.max()) if t.size else 0
    if mx > 10 ** 17:
        return (t // 1_000_000_000).astype(np.int64)
    elif mx > 10 ** 14:
        return (t // 1_000_000).astype(np.int64)
    elif mx > 10 ** 11:
        return (t // 1_000).astype(np.int64)
    else:
        return t


def group_by_second_preserving_order(t_sec: np.ndarray, TT: np.ndarray, sym: str) -> Dict[int, List[List[float]]]:
    """Group trade rows by whole second, preserving original order.

    The second is computed by integer division of epoch seconds. The lot
    field (index 1) of each row is multiplied by the symbol‑specific
    lot multiplier. The resulting dictionary maps seconds to lists of
    trade rows, each represented as a Python list.
    """
    mult = lot_multiplier(sym)
    by_sec: Dict[int, List[List[float]]] = {}
    for ts, row in zip(t_sec, TT):
        r = list(row)
        if len(r) > 1:
            r[1] = float(r[1]) * mult
        by_sec.setdefault(int(ts), []).append(r)
    return by_sec


def process_day(
    data_dir: Path,
    day: str,
    pair: Tuple[str, str],
    params: Optional[SegParams] = None,
) -> List[Dict[str, Any]]:
    """Load and process a day's trading data into events.

    This is a convenience function that ties together data loading,
    segmentation and event collection. It expects that NPZ files are
    named according to ``{day}_{symbol}.npz``. A new
    :class:`EventSegmenter` is instantiated with the provided
    parameters or default values based on the symbol pair. All events
    produced are returned as a list of dictionaries.
    """
    files = find_files_for_day(data_dir, day, pair)
    sym_a, sym_b = pair[0].lower(), pair[1].lower()
    if sym_a not in files or sym_b not in files:
        raise FileNotFoundError(f"Missing NPZ files for symbols {pair} on day {day}")
    t_a, TT_a = load_npz(files[sym_a])
    t_b, TT_b = load_npz(files[sym_b])
    sec_a = group_by_second_preserving_order(to_epoch_seconds(t_a), TT_a, sym_a)
    sec_b = group_by_second_preserving_order(to_epoch_seconds(t_b), TT_b, sym_b)
    secs = sorted(set(sec_a.keys()) | set(sec_b.keys()))
    tick_size_map = {"wdo": 0.5, "dol": 0.5, "win": 5.0, "ind": 5.0}
    tick_a = tick_size_map.get(sym_a, 0.5)
    tick_b = tick_size_map.get(sym_b, 0.5)
    p = params or SegParams(
        tick_a=tick_a,
        tick_b=tick_b,
        lot_mult_a=lot_multiplier(sym_a),
        lot_mult_b=lot_multiplier(sym_b),
    )
    seg = EventSegmenter(p)
    seg.reset()
    eventos: List[Dict[str, Any]] = []
    for s in secs:
        trades_a = sec_a.get(s, [])
        trades_b = sec_b.get(s, [])
        eventos.extend(seg.step(s, trades_a, trades_b))
    if seg.evt is not None:
        # Close the last event at end of day
        seg.evt["end_reason"] = "eod"
        eventos.append(seg.evt)
    return eventos


def events_to_df(eventos: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert a list of event dictionaries into a Pandas DataFrame.

    The resulting DataFrame contains both basic event metadata (start/end
    times, durations, volume and price extrema) as well as flattened
    representations of nested dictionaries found in the enriched
    statistics. Columns for flags are prefixed with ``flag_`` and set
    to 1 for active flags. Nested dictionaries are flattened using
    underscores to concatenate keys.
    """
    rows: List[Dict[str, Any]] = []
    for ev in eventos:
        base: Dict[str, Any] = {
            "start_ts": ev.get("start"),
            "end_ts": ev.get("end"),
            "dur_s": ev.get("dur"),
            "start_reason": ev.get("start_reason"),
            "end_reason": ev.get("end_reason"),
            "vol": ev.get("vol"),
            "n_trades": int(ev.get("n_trades", 0) or 0),
            "range_ticks": ev.get("range_ticks"),
            "d_ticks_max": ev.get("d_ticks_max"),
            "ticks_per_s": ev.get("ticks_per_s"),
            "p0_a": ev.get("p0_a"),
            "p0_b": ev.get("p0_b"),
            "plast_a": ev.get("plast_a"),
            "plast_b": ev.get("plast_b"),
            "pmin_a": ev.get("pmin_a"),
            "pmax_a": ev.get("pmax_a"),
            "pmin_b": ev.get("pmin_b"),
            "pmax_b": ev.get("pmax_b"),
        }
        flags = ev.get("start_flags", {}) or {}
        for k, v in flags.items():
            base[f"flag_{k}"] = 1 if v else 0
        # Flatten nested dictionaries
        def _flatten_dict(d: Any, prefix: str) -> Dict[str, Any]:
            out: Dict[str, Any] = {}
            if not isinstance(d, dict):
                return out
            for k, v in d.items():
                key = f"{prefix}_{k}"
                if isinstance(v, dict):
                    out.update(_flatten_dict(v, key))
                else:
                    out[key] = v
            return out
        for blk in (
            "price_info",
            "volume_info",
            "trade_info",
            "move_info",
            "players_info",
            "context_info",
        ):
            base.update(_flatten_dict(ev.get(blk, {}), blk))
        # Human‑readable times and hour of day
        try:
            base["start_time"] = pd.to_datetime(int(base["start_ts"]), unit="s") if base["start_ts"] else None
            base["end_time"] = pd.to_datetime(int(base["end_ts"]), unit="s") if base["end_ts"] else None
            base["hour"] = base["start_time"].hour if base.get("start_time") is not None else None
        except Exception:
            base["start_time"] = base["end_time"] = None
            base["hour"] = None
        rows.append(base)
    df = pd.DataFrame(rows)
    # Convert selected columns to numeric
    numeric_cols = [
        "dur_s",
        "vol",
        "n_trades",
        "range_ticks",
        "d_ticks_max",
        "ticks_per_s",
        "p0_a",
        "p0_b",
        "plast_a",
        "plast_b",
        "pmin_a",
        "pmax_a",
        "pmin_b",
        "pmax_b",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "hour" in df.columns:
        df["hour"] = pd.to_numeric(df["hour"], errors="coerce")
    return df


def expand_list_columns(df: pd.DataFrame, candidate_prefixes: Iterable[str] = ("context_info_price_feats",)) -> pd.DataFrame:
    """Expand columns containing lists or arrays into separate columns.

    This helper looks for DataFrame columns whose names start with one of
    the specified prefixes. If any values in such a column are lists,
    tuples or numpy arrays, the column is expanded into multiple
    subcolumns with a numerical suffix. The original column is dropped.
    """
    out = df.copy()
    for col in list(out.columns):
        if any(col.startswith(p) for p in candidate_prefixes):
            is_listy = out[col].apply(lambda v: isinstance(v, (list, tuple, np.ndarray)))
            if is_listy.any():
                max_len = int(out[col][is_listy].map(len).max())
                for i in range(max_len):
                    out[f"{col}_{i}"] = out[col].apply(
                        lambda v: (v[i] if isinstance(v, (list, tuple, np.ndarray)) and len(v) > i else np.nan)
                    )
                out.drop(columns=[col], inplace=True)
    return out


def build_feature_df(df: pd.DataFrame) -> pd.DataFrame:
    """Select and prepare numeric features from an event DataFrame.

    This function removes raw price columns and retains derived metrics
    such as volume, trade and movement statistics as well as context
    features. Flags are kept as binary indicators. The result is a
    DataFrame suitable for normalisation and machine learning.
    """
    df2 = expand_list_columns(df)
    keep_prefixes = [
        "volume_info_",
        "trade_info_",
        "move_info_",
        "players_info_",
        "context_info_",
    ]
    base_feats = [
        "dur_s",
        "vol",
        "n_trades",
        "range_ticks",
        "d_ticks_max",
        "ticks_per_s",
    ]
    flag_cols = [c for c in df2.columns if c.startswith("flag_")]
    cols: List[str] = []
    for c in base_feats:
        if c in df2.columns:
            cols.append(c)
    cols.extend(flag_cols)
    for c in df2.columns:
        if any(c.startswith(p) for p in keep_prefixes):
            if not any(x in c for x in ["p0", "plast", "pmin", "pmax", "price_raw", "price_abs"]):
                cols.append(c)
    cols = list(dict.fromkeys(cols))
    feat_df = df2[cols].copy()
    for c in feat_df.columns:
        feat_df[c] = pd.to_numeric(feat_df[c], errors="coerce")
    return feat_df


def _minmax_clip_norm(series: pd.Series, p_lo: float = 1.0, p_hi: float = 99.0, log1p: bool = False) -> pd.Series:
    """Clip and scale a numeric series to the range [0, 1] using robust percentiles.

    Optionally applies a log(1 + x) transform before clipping. Values are
    clipped between the ``p_lo`` and ``p_hi`` percentiles and then
    scaled so that the minimum maps to 0 and the maximum to 1.
    """
    s = series.astype(float).copy()
    if log1p:
        s = np.log1p(np.clip(s, a_min=0, a_max=None))
    valid = s[~np.isnan(s)]
    if valid.empty:
        lo, hi = 0.0, 1.0
    else:
        lo, hi = np.nanpercentile(valid, [p_lo, p_hi])
        if hi <= lo:
            hi = lo + 1e-9
    s = s.clip(lo, hi)
    s = (s - lo) / (hi - lo)
    return s


def normalize_events_to_vectors(
    df: pd.DataFrame,
    range_mode: str = "0-1",
) -> Tuple[np.ndarray, List[str], Dict[str, List[str]]]:
    """Normalise event features into a 2D array suitable for ML models.

    Returns a tuple of ``(X, feature_names, scaler_meta)`` where ``X``
    is a numpy array of shape (n_events, n_features), ``feature_names``
    is the list of column names corresponding to each column in ``X``, and
    ``scaler_meta`` holds lists of columns treated as linear, log‑like
    or binary features. The ``range_mode`` parameter controls whether
    values remain in [0,1] or are scaled to [-1,1].
    """
    feat_df = build_feature_df(df)
    bin_feats = [c for c in feat_df.columns if c.startswith("flag_")]
    log_like = [c for c in feat_df.columns if any(x in c.lower() for x in [
        "vol", "volume", "lot", "rate", "count", "n_trades", "absorption", "intensity"
    ])]
    linear_like = [c for c in feat_df.columns if c not in bin_feats and c not in log_like]
    scaler_meta = {
        "log_like": log_like,
        "linear_like": linear_like,
        "bin_feats": bin_feats,
    }
    Z = pd.DataFrame(index=feat_df.index)
    for c in linear_like:
        Z[c] = _minmax_clip_norm(feat_df[c], p_lo=1, p_hi=99, log1p=False)
    for c in log_like:
        Z[c] = _minmax_clip_norm(feat_df[c], p_lo=1, p_hi=99, log1p=True)
    for c in bin_feats:
        Z[c] = feat_df[c].fillna(0.0).clip(0, 1)
    feature_names = list(Z.columns)
    X = Z[feature_names].values.astype("float32")
    if range_mode == "-1-1":
        X = X * 2.0 - 1.0
    return X, feature_names, scaler_meta