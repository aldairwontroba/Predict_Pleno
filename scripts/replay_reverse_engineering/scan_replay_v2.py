#!/usr/bin/env python3
import argparse, csv, datetime as dt, re, sys
from typing import List, Tuple
from datetime import datetime, timezone, timedelta

PRINTABLE = set(range(0x20, 0x7F))

def decode_ts_be_u64(b: bytes, i: int):
    """Detecta um u64 big-endian plausível como timestamp (µs ou ns)."""
    if i+8 > len(b): return None
    v = int.from_bytes(b[i:i+8], 'big', signed=False)
    # faixas plausíveis (1970..2040) em micros ou nanos
    # micros ~ 1.6e15 para 2024; nanos ~ 1.6e18
    try:
        if 1_300_000_000_000_000 <= v <= 2_200_000_000_000_000:
            return dt.datetime.utcfromtimestamp(v/1_000_000).replace(tzinfo=dt.timezone.utc), 'us'
        if 1_300_000_000_000_000_000 <= v <= 2_200_000_000_000_000_000:
            return dt.datetime.utcfromtimestamp(v/1_000_000_000).replace(tzinfo=dt.timezone.utc), 'ns'
    except (OverflowError, OSError, ValueError):
        pass
    return None

def scan_all_timestamps(buf: bytes) -> List[Tuple[int, dt.datetime]]:
    out = []
    i = 0
    while i+8 <= len(buf):
        hit = decode_ts_be_u64(buf, i)
        if hit:
            ts, unit = hit
            out.append((i, ts))
            # pular alguns bytes para evitar múltiplas leituras sobrepostas do mesmo valor
            i += 8
        else:
            i += 1
    return out

def find_all(hay: bytes, needles: List[bytes]) -> List[Tuple[int, bytes]]:
    # retorna [(offset, token_bytes)]
    hits = []
    for n in needles:
        start = 0
        while True:
            j = hay.find(n, start)
            if j < 0: break
            hits.append((j, n))
            start = j + 1
    hits.sort()
    return hits

def to_local(ts_utc, tz_offset):
    # normaliza tz: aceita -03:00, -0300, -3
    s = str(tz_offset).strip().strip("'\"")
    m = re.fullmatch(r'([+-])(\d{1,2})(?::?(\d{2}))?', s)
    if not m:
        raise ValueError(f"TZ inválido: {tz_offset!r}")
    sign = -1 if m.group(1) == '-' else 1
    hh = int(m.group(2)); mm = int(m.group(3) or 0)
    delta = timedelta(hours=sign*hh, minutes=sign*mm)

    # aceita datetime, timestamp em s ou ms
    if isinstance(ts_utc, datetime):
        dt = ts_utc if ts_utc.tzinfo else ts_utc.replace(tzinfo=timezone.utc)
        dt = dt.astimezone(timezone.utc)
    else:
        t = float(ts_utc)
        if t > 1e12:  # provavelmente milissegundos
            t /= 1000.0
        dt = datetime.fromtimestamp(t, tz=timezone.utc)

    return (dt + delta).isoformat()

def hex_preview(b: bytes, maxlen=64) -> str:
    s = b[:maxlen].hex(' ')
    if len(b) > maxlen: s += ' ...'
    return s

def ascii_preview(b: bytes, maxlen=64) -> str:
    s = ''.join(chr(x) if x in PRINTABLE else '.' for x in b[:maxlen])
    if len(b) > maxlen: s += ' ...'
    return s

def main():
    ap = argparse.ArgumentParser(description="Scanner Tryd Replay — herda timestamps e agrupa entre timestamps.")
    ap.add_argument("path", help="arquivo .dat")
    ap.add_argument("--tokens", nargs="+", required=True, help="lista de tokens (ex.: wdom2 dolm2 frp ptxusd)")
    ap.add_argument("--tz", default="-03:00", help="offset local p/ coluna local (ex.: -03:00)")
    ap.add_argument("--window", type=int, default=128, help="bytes após o token para prévia/extração")
    args = ap.parse_args()

    buf = open(args.path, 'rb').read()
    tokens_b = [t.encode('ascii') for t in args.tokens]

    # 1) timestamps no arquivo todo
    ts_list = scan_all_timestamps(buf)  # [(offset, ts_utc)]
    ts_list.sort()
    # Mapear para busca rápida do último ts antes de um offset
    ts_offsets = [o for (o, _) in ts_list]

    def last_ts_before(pos: int):
        import bisect
        i = bisect.bisect_right(ts_offsets, pos) - 1
        if i >= 0:
            return ts_list[i]
        return (None, None)

    # 2) encontrar todos os tokens
    hits = find_all(buf, tokens_b)

    # 3) escrever events.csv
    ev_rows = []
    for off, tok in hits:
        # timestamp herdado = último ts antes do token
        ts_off, ts_utc = last_ts_before(off)
        # bytes até próximo timestamp (útil para “tudo entre timestamps”)
        nxt = len(buf)
        for o,_ in ts_list:
            if o > off:
                nxt = o
                break
        slice_after = buf[off:nxt]
        # pequena prévia para ajudar a inspecionar
        preview_hex = hex_preview(slice_after, args.window)
        preview_ascii = ascii_preview(slice_after, args.window)
        ev_rows.append({
            "offset_hex": f"0x{off:08X}",
            "token": tok.decode(),
            "ts_offset_hex": (f"0x{ts_off:08X}" if ts_off is not None else ""),
            "timestamp_utc": (ts_utc.isoformat() if ts_utc else ""),
            "timestamp_local": (to_local(ts_utc, args.tz) if ts_utc else ""),
            "bytes_until_next_ts": len(slice_after),
            "hex_preview": preview_hex,
            "ascii_preview": preview_ascii
        })

    with open("events.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(ev_rows[0].keys()) if ev_rows else
                           ["offset_hex","token","ts_offset_hex","timestamp_utc","timestamp_local","bytes_until_next_ts","hex_preview","ascii_preview"])
        w.writeheader()
        for r in ev_rows: w.writerow(r)

    # 4) escrever intervals.csv (tudo entre timestamps)
    int_rows = []
    for idx, (o, ts) in enumerate(ts_list):
        start = o
        end   = ts_list[idx+1][0] if idx+1 < len(ts_list) else len(buf)
        # tokens dentro do intervalo
        inside = [t.decode() for pos, t in hits if start <= pos < end]
        int_rows.append({
            "interval_idx": idx,
            "ts_offset_hex": f"0x{start:08X}",
            "timestamp_utc_start": ts.isoformat(),
            "timestamp_local_start": to_local(ts, args.tz),
            "end_offset_hex": f"0x{end:08X}",
            "length_bytes": end-start,
            "tokens_in_interval": " ".join(inside),
            "first_bytes_hex": hex_preview(buf[start:end], 48),
        })
    with open("intervals.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(int_rows[0].keys()) if int_rows else
                           ["interval_idx","ts_offset_hex","timestamp_utc_start","timestamp_local_start","end_offset_hex","length_bytes","tokens_in_interval","first_bytes_hex"])
        w.writeheader()
        for r in int_rows: w.writerow(r)

    print(f"[ok] events.csv: {len(ev_rows)} linhas")
    print(f"[ok] intervals.csv: {len(int_rows)} intervalos")
    print("Dica: ordene events.csv por ts_offset_hex e offset_hex para ver blocos de um mesmo evento.")

if __name__ == "__main__":
    sys.setrecursionlimit(1<<20)
    main()
