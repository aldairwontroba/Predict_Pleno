#!/usr/bin/env python3
import argparse, csv, re, struct
from datetime import datetime, timezone, timedelta
from pathlib import Path

# ---------- util ----------
def read_uvarint(buf, i):
    x = 0; s = 0; start = i
    while i < len(buf):
        b = buf[i]; i += 1
        x |= (b & 0x7F) << s
        if (b & 0x80) == 0:
            return x, i, i - start
        s += 7
        if s > 63:
            break
    return None, start, 0

def zigzag(n):  # se precisar olhar signed
    return (n >> 1) ^ -(n & 1)

def to_local_str(ms, tz_str):
    # ms -> datetime
    dt_utc = datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
    if not tz_str: 
        return dt_utc.isoformat(), ""
    sign = 1 if tz_str[0] == '+' else -1
    hh, mm = map(int, tz_str[1:].split(':'))
    tz = timezone(sign * timedelta(hours=hh, minutes=mm))
    return dt_utc.isoformat(), dt_utc.astimezone(tz).isoformat()

num_re = re.compile(rb"(?<![0-9.])\d{1,7}(?:\.\d{1,4})?(?![0-9.])")  # ASCII números

def slide_floats(block, prefer_ranges):
    """Tenta doubles e floats (LE e BE) a cada byte; devolve candidatos legíveis."""
    cands = []
    def ok_num(x):
        if x != x or x in (float('inf'), float('-inf')):  # NaN/inf
            return False
        ax = abs(x)
        # Evita lixo gigante; mantemos coisas "de preço/qtde"
        return 0 <= ax <= 1e7
    L = len(block)
    for off in range(0, max(0, L - 4) + 1):
        try:
            f_le = struct.unpack_from("<f", block, off)[0]
            if ok_num(f_le):
                for (lo, hi, tag) in prefer_ranges:
                    if lo <= f_le <= hi:
                        cands.append((off, "fLE", f_le, tag))
                        break
        except struct.error: pass
    for off in range(0, max(0, L - 8) + 1):
        try:
            d_le = struct.unpack_from("<d", block, off)[0]
            if ok_num(d_le):
                for (lo, hi, tag) in prefer_ranges:
                    if lo <= d_le <= hi:
                        cands.append((off, "dLE", d_le, tag))
                        break
        except struct.error: pass
    # BE (menos provável, mas tentamos)
    for off in range(0, max(0, L - 4) + 1):
        try:
            f_be = struct.unpack_from(">f", block, off)[0]
            if ok_num(f_be):
                for (lo, hi, tag) in prefer_ranges:
                    if lo <= f_be <= hi:
                        cands.append((off, "fBE", f_be, tag))
                        break
        except struct.error: pass
    for off in range(0, max(0, L - 8) + 1):
        try:
            d_be = struct.unpack_from(">d", block, off)[0]
            if ok_num(d_be):
                for (lo, hi, tag) in prefer_ranges:
                    if lo <= d_be <= hi:
                        cands.append((off, "dBE", d_be, tag))
                        break
        except struct.error: pass
    # dedup aproximado (offset,tipo,valor arredondado)
    seen = set(); out = []
    for off, kind, val, tag in sorted(cands, key=lambda x:(x[0], x[1])):
        key = (off, kind, round(float(val), 6))
        if key not in seen:
            seen.add(key)
            out.append((off, kind, float(val), tag))
    return out

def hexdump(b):
    return " ".join(f"{x:02X}" for x in b)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Extrai recortes por timestamp (epoch ms varint) e tenta decodificar campos.")
    ap.add_argument("path", help=".dat de replay")
    ap.add_argument("--tz", default="-03:00", help="Fuso para coluna local (ex: -03:00, +00:00)")
    ap.add_argument("--radius", type=int, default=192, help="bytes antes/depois do timestamp para recorte")
    ap.add_argument("--max", type=int, default=0, help="limite de eventos (0 = todos)")
    ap.add_argument("--outdir", default="events_by_ts", help="pasta para .bin e CSV")
    ap.add_argument("--csvname", default="events.csv", help="nome do CSV de saída")
    ap.add_argument("--minms", type=int, default=1_600_000_000_000)
    ap.add_argument("--maxms", type=int, default=2_000_000_000_000)
    ap.add_argument("--varints_after", type=int, default=16, help="quantos varints listar após o timestamp")
    args = ap.parse_args()

    p = Path(args.path); data = p.read_bytes()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Varredura completa por uvarints "grandes" (epoch ms)
    hits = []
    i = 0; N = len(data)
    while i < N:
        v, j, ln = read_uvarint(data, i)
        if ln == 0:
            i += 1
            continue
        if args.minms <= (v or 0) <= args.maxms:
            hits.append((i, v, ln))
        i = j

    # Ordena por offset (ordem do arquivo)
    hits.sort(key=lambda x: x[0])
    if args.max and len(hits) > args.max:
        hits = hits[:args.max]

    # Faixas preferidas p/ floats (ajuste conforme o ativo)
    prefer = [
        (0, 100000, "qty/idx"),
        (500, 6000, "price-ish"),     # mini dólar ~5k
        (4800, 5200, "price-USD"),
        (5100, 5200, "price-narrow"),
    ]

    rows = []
    for idx, (off, ms, ln) in enumerate(hits, 1):
        s = max(0, off - args.radius)
        e = min(N, off + args.radius)
        block = data[s:e]

        # salva recorte
        bin_name = f"{idx:05d}_off_{off:08X}_ts_{ms}.bin"
        (outdir / bin_name).write_bytes(block)

        # números ASCII
        ascii_nums = [m.group(0).decode("ascii") for m in num_re.finditer(block)]
        ascii_preview = " ".join(ascii_nums[:20])

        # floats/doubles candidatos
        fl = slide_floats(block, prefer)
        # mantemos os mais próximos do timestamp (ordenar por |offset - (off - s)|)
        center = off - s
        fl_sorted = sorted(fl, key=lambda x:(abs(x[0] - center), x[0]))[:25]
        floats_preview = " ".join(f"{k}@{o:+d}={v:.2f}({tag})" for (o,k,v,tag) in fl_sorted)

        # varints logo DEPOIS do timestamp (no arquivo)
        vseq = []
        pos = (off + ln)
        for _ in range(args.varints_after):
            if pos >= N: break
            vv, jj, l2 = read_uvarint(data, pos)
            if l2 == 0: break
            vseq.append((pos - off, l2, vv))
            pos = jj
        vseq_preview = " ".join(f"{delta:+d}/b{l}:{val}" for (delta, l, val) in vseq[:20])

        utc_iso, local_iso = to_local_str(ms, args.tz)

        rows.append({
            "offset_hex": f"0x{off:08X}",
            "offset": off,
            "ts_ms": ms,
            "ts_utc": utc_iso,
            "ts_local": local_iso,
            "varint_len": ln,
            "window_hex": f"{s:08X}-{e:08X}",
            "ascii_numbers": ascii_preview,
            "float_candidates": floats_preview,
            "next_varints": vseq_preview,
            "bin_file": bin_name,
            "window_hexdump": hexdump(block[:64])  # cabeçalho só p/ olhar rápido
        })

    # CSV
    csv_path = outdir / args.csvname
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        cols = ["offset_hex","offset","ts_ms","ts_utc","ts_local","varint_len","window_hex",
                "ascii_numbers","float_candidates","next_varints","bin_file","window_hexdump"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    print(f"OK: {len(rows)} timestamps → {csv_path}")
    print(f"Recortes salvos em: {outdir}/<N>_off_<offset>_ts_<ms>.bin")
    if rows:
        print("Dica: procure nos float_candidates por valores perto de 5140.00, 5145.50, 5146.00 e tamanhos 5, 15 etc.")

if __name__ == "__main__":
    main()
