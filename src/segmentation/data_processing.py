import re
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Iterable, Any
import numpy as np

from .segmentation import EventSegmenter, SegParams
from .plotting import print_event

def lot_multiplier(sym: str) -> float:
    return 5.0 if sym.lower() in ("dol", "ind") else 1.0

def find_files_for_pair(
    data_dir: Path,
    startday: str,
    endday: str,
    pair: Tuple[str, str]
) -> List[Dict[str, Path]]:
    """
    Localiza todos os arquivos NPZ entre `startday` e `endday` para o par de símbolos informado.

    Espera arquivos no formato: `YYYYMMDD_SYMBOL.npz`, todos no mesmo diretório (sem subpastas).

    Retorna uma lista de dicionários:
        [
            {"day": "20250910", "wdo": Path("20250910_wdo.npz"), "dol": Path("20250910_dol.npz")},
            {"day": "20250911", "wdo": Path("20250911_wdo.npz"), "dol": Path("20250911_dol.npz")},
            ...
        ]
    """
    start = int(startday)
    end = int(endday)
    sym_a, sym_b = pair[0].lower(), pair[1].lower()

    # padrão: ex: 20250910_wdo.npz
    pat = re.compile(r"^(\d{8})_([A-Za-z0-9]+)\.npz$", re.IGNORECASE)

    # estrutura temporária: {day: {"day": day, "wdo": Path(...), "dol": Path(...)}}
    by_day: Dict[str, Dict[str, Path]] = {}

    for p in data_dir.glob("*.npz"):
        m = pat.match(p.name)
        if not m:
            continue

        day_str, symbol = m.group(1), m.group(2).lower()
        try:
            day = int(day_str)
        except ValueError:
            continue

        if not (start <= day <= end):
            continue
        if symbol not in (sym_a, sym_b):
            continue

        # adiciona no dicionário daquele dia
        if day_str not in by_day:
            by_day[day_str] = {"day": day_str}
        by_day[day_str][symbol] = p

    # ordena por data e retorna como lista
    return [by_day[k] for k in sorted(by_day.keys())]


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

def _infer_day_str_from_paths(day: Dict[str, Path]) -> str:
    """
    Tenta extrair 'YYYYMMDD' do dict do dia. Se houver a chave 'day', usa ela.
    Senão, tenta parsear do nome do arquivo '<YYYYMMDD>_symbol.npz'.
    """
    s = day.get("day")
    if isinstance(s, str) and len(s) == 8 and s.isdigit():
        return s
    pat = re.compile(r"^(\d{8})_", re.IGNORECASE)
    for k, p in day.items():
        if k == "day":
            continue
        m = pat.match(Path(p).name)
        if m:
            return m.group(1)
    return "unknown"

def process_day(out_dir, day, pair, params = None, imprimir=False, save=False, fillna = 0.0):
    
    import numpy as _np

    sym_a, sym_b = pair[0].lower(), pair[1].lower()
    path_a = day[sym_a]
    path_b = day[sym_b]
    day_str = _infer_day_str_from_paths(day)

    t_a, TT_a = load_npz(path_a)
    t_b, TT_b = load_npz(path_b)
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

    if save:
        # converte em matriz
        first_vec = eventos[0].get("vector")
        feature_order = list(first_vec.keys())  # mantém a ordem original de inserção
        m = len(eventos)
        n = len(feature_order)
        mat = _np.full((m, n), fillna, dtype=float)
        for i, ev in enumerate(eventos):
            vec = ev.get("vector") or {}
            for j, k in enumerate(feature_order):
                v = vec.get(k, fillna)
                try:
                    mat[i, j] = float(v)
                except Exception:
                    mat[i, j] = fillna
        # nome de saída
        out_name = f"{day_str}_{pair[0]}_{pair[1]}.npy"
        out_path = Path(out_dir) / out_name
        _np.save(out_path, mat)
        print(f"[ok] {day_str}: salvo {out_path} shape={mat.shape}")

    return eventos, sec_a, sec_b




