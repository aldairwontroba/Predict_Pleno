import re
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Iterable, Any
import numpy as np

from segmentation import EventSegmenter, SegParams
from plotting import print_event

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
    imprimir=False
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

    if sym_a.lower() in ("wdo", "dol"):
        params = SegParams()  # padrão genérico
    elif sym_a.lower() in ("win", "ind"):
        params = SegParams.for_indice()
        
    seg = EventSegmenter(params)
    seg.reset()
    eventos: List[Dict[str, Any]] = []
    for s in secs:
        trades_a = sec_a.get(s, [])
        trades_b = sec_b.get(s, [])
        events_done = seg.step(s, trades_a, trades_b)
        eventos.extend(events_done)
        if imprimir:
            for ev in events_done:
                print_event(ev)
    if seg.evt is not None:
        # Close the last event at end of day
        seg.evt["end_reason"] = "eod"
        eventos.append(seg.evt)
    return eventos, sec_a, sec_b
