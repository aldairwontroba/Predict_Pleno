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

from segmentation import EventSegmenter, SegParams


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
    method: str = "tanh",
    range_mode: str | None = None,
    tanh_target: float = 0.8,
) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    """Normalise event features into a 2D array suitable for ML models.

    This routine transforms the numeric event features into a normalised
    representation.  Compared with the original implementation, the
    default ``method`` now performs a robust normalisation based on
    hyperbolic tangent (``tanh``) scaling.  Each continuous feature is
    centred by its median and scaled by a robust measure of spread
    (95\% inter‑percentile range).  A per‑feature constant is chosen
    such that approximately 95\% of the resulting values lie within
    ±0.95 before the final tanh is applied.  Binary features (those
    beginning with ``flag_``) are preserved as 0/1 and continuous
    features representing periodic signals (``context_info_price_feats_*``)
    are passed through unchanged (they already lie in [−1,1]).  Features
    that encode a share or proportion (substring ``_share_``) are kept
    in their original [0,1] range.  The returned ``scaler_meta``
    contains the per‑feature parameters used for scaling (median,
    robust scale and constant) so that the same transformation can be
    applied to future data.

    Parameters
    ----------
    df : pd.DataFrame
        Event dataframe from :func:`events_to_df`.
    method : str, optional
        Normalisation method.  Supported values are ``"tanh"`` (default)
        and ``"minmax"`` (legacy behaviour).  When ``"minmax"`` is
        selected, values are clipped between the 1st and 99th
        percentiles and scaled to [0,1]; logarithmic features use a
        log(1+x) transform.  The ``range_mode`` argument from older
        versions is preserved for backwards compatibility but is
        ignored when ``method="tanh"``.
    range_mode : str | None, optional
        Legacy parameter controlling whether min–max scaled values
        remain in [0,1] or are linearly mapped to [−1,1].  Only used
        when ``method="minmax"``.

    Returns
    -------
    X : np.ndarray
        Normalised feature matrix of shape ``(n_events, n_features)``.
    feature_names : list of str
        Names of the features in the same order as columns of ``X``.
    scaler_meta : dict
        Metadata describing the transformation.  For ``method="tanh"``
        this includes per‑feature parameters (median, spread, constant,
        flags indicating log/periodic/share/binary) and the chosen
        percentile thresholds.  For ``method="minmax"`` this mirrors
        the previous ``scaler_meta`` structure (lists of log-like,
        linear-like and binary features).
    """
    # Backwards compatibility: if 'method' is passed as an old range_mode ("0-1"/"-1-1"),
    # interpret accordingly.
    if method in ("0-1", "-1-1") and range_mode is None:
        # Old call signature: second positional argument was range_mode
        range_mode = method
        method = "minmax"

    feat_df = build_feature_df(df)
    # Identify feature categories
    bin_feats = [c for c in feat_df.columns if c.startswith("flag_")]
    periodic_feats = [c for c in feat_df.columns if c.startswith("context_info_price_feats_")]
    share_feats = [c for c in feat_df.columns if "_share_" in c]
    # Log‑like features are heavy‑tailed volumes, rates and counts
    log_like = [c for c in feat_df.columns if any(x in c.lower() for x in [
        "vol", "volume", "lot", "rate", "count", "n_trades", "absorption", "intensity"
    ])]
    # Remove duplicates from lists
    log_like = [c for c in log_like if c not in bin_feats and c not in periodic_feats and c not in share_feats]
    # Determine linear features (anything not classified above)
    linear_like = [c for c in feat_df.columns if c not in bin_feats and c not in periodic_feats and c not in share_feats and c not in log_like]

    if method == "minmax":
        # Legacy min–max scaling behaviour
        Z = pd.DataFrame(index=feat_df.index)
        for c in linear_like:
            Z[c] = _minmax_clip_norm(feat_df[c], p_lo=1, p_hi=99, log1p=False)
        for c in log_like:
            Z[c] = _minmax_clip_norm(feat_df[c], p_lo=1, p_hi=99, log1p=True)
        for c in bin_feats:
            Z[c] = feat_df[c].fillna(0.0).clip(0, 1)
        for c in periodic_feats:
            # Already in [−1,1], map to [0,1]
            Z[c] = (feat_df[c].astype(float).clip(-1, 1) + 1.0) / 2.0
        for c in share_feats:
            Z[c] = feat_df[c].astype(float).clip(0, 1)
        feature_names = list(Z.columns)
        X = Z[feature_names].values.astype("float32")
        if range_mode == "-1-1":
            X = X * 2.0 - 1.0
        scaler_meta = {
            "log_like": log_like,
            "linear_like": linear_like,
            "bin_feats": bin_feats,
            "periodic_feats": periodic_feats,
            "share_feats": share_feats,
        }
        return X, feature_names, scaler_meta
    # Default: robust tanh scaling
    # Prepare containers for output
    X_df = pd.DataFrame(index=feat_df.index)
    scaler_meta: Dict[str, Any] = {
        "method": "tanh",
        "tanh_target": tanh_target,
        "feature_params": {},  # per‑feature parameters (median, scale, constant, flags)
    }
    # Helper to compute robust tanh scaling for a series
    def _tanh_scale(series: pd.Series, is_log: bool) -> Tuple[pd.Series, Dict[str, Any]]:
        s = series.astype(float)
        # Apply log1p if requested and values are non‑negative
        if is_log:
            # Negative values may occur due to data errors; clip to zero
            s = np.log1p(np.clip(s, a_min=0, a_max=None))
        # Replace NaN with median after computing statistics
        valid = s[~np.isnan(s)]
        if valid.empty or float(valid.max() - valid.min()) == 0.0:
            # Constant or all missing: return zeros
            params = {
                "median": float(valid.median()) if not valid.empty else 0.0,
                "spread": 1.0,
                "constant": 0.0,
                "is_log": is_log,
            }
            return pd.Series(np.zeros(len(s)), index=s.index), params
        m = float(valid.median())
        # Robust spread: 2.5th and 97.5th percentiles
        q025, q975 = np.percentile(valid, [2.5, 97.5])
        spread = (q975 - q025) / 2.0
        if spread <= 0:
            spread = float(valid.std()) if valid.std() > 0 else 1.0
        # Compute robust z‑scores
        z = (s - m) / spread
        # Determine constant: map 95th percentile of |z| to the desired tanh_target
        z_valid = z[~np.isnan(z)]
        q_abs = np.percentile(np.abs(z_valid), 95)
        if q_abs <= 0:
            const = 1.0
        else:
            # np.arctanh expects argument in (−1,1); tanh_target should be within this interval
            const = float(np.arctanh(tanh_target) / q_abs)
        # Apply tanh scaling
        scaled = np.tanh(const * z)
        # Replace any NaN values (from original missing data) with 0.0
        scaled = np.where(np.isnan(scaled), 0.0, scaled)
        params = {
            "median": m,
            "spread": spread,
            "constant": const,
            "is_log": is_log,
        }
        return pd.Series(scaled, index=s.index), params
    # Process features
    for c in linear_like:
        scaled, params = _tanh_scale(feat_df[c], is_log=False)
        X_df[c] = scaled
        scaler_meta["feature_params"][c] = params
    for c in log_like:
        scaled, params = _tanh_scale(feat_df[c], is_log=True)
        X_df[c] = scaled
        scaler_meta["feature_params"][c] = params
    for c in periodic_feats:
        # Already ∈ [−1,1]; clip to avoid outliers and fill missing with 0
        s = feat_df[c].astype(float).clip(-1, 1)
        # fill missing values with 0 (centre of periodic domain)
        s = s.fillna(0.0)
        X_df[c] = s
        scaler_meta["feature_params"][c] = {
            "median": 0.0,
            "spread": 1.0,
            "constant": 1.0,
            "is_log": False,
            "is_periodic": True,
        }
    for c in share_feats:
        # Values in [0,1]; clip to ensure bounds and fill missing with 0
        s = feat_df[c].astype(float).clip(0, 1)
        s = s.fillna(0.0)
        X_df[c] = s
        scaler_meta["feature_params"][c] = {
            "median": 0.0,
            "spread": 1.0,
            "constant": 1.0,
            "is_log": False,
            "is_share": True,
        }
    for c in bin_feats:
        s = feat_df[c].fillna(0.0).clip(0, 1)
        X_df[c] = s
        scaler_meta["feature_params"][c] = {
            "median": 0.0,
            "spread": 1.0,
            "constant": 1.0,
            "is_log": False,
            "is_binary": True,
        }
    # Preserve column order consistent with original DataFrame
    feature_names = list(X_df.columns)
    X = X_df[feature_names].values.astype("float32")
    return X, feature_names, scaler_meta