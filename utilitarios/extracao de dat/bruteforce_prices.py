# bruteforce_prices.py
# Dado o CSV extraído (v1.6), faz brute-force de possíveis preços
# perto do token: ASCII, float/double (LE/BE), int32/int64 com escalas.
import csv, sys, re, struct, math
from datetime import datetime
from collections import defaultdict

ASCII_NUM = re.compile(r"(?<![\d.])\d{1,6}(?:\.\d{1,4})?(?![\d.])")

def parse_iso(ts):
    return datetime.fromisoformat(ts)

def within(dt, a, b):
    return (a is None or dt >= a) and (b is None or dt <= b)

def safe_float(x):
    return (not math.isnan(x)) and (not math.isinf(x))

def try_floats(buf, base_off, rel_from, rel_to, lo, hi):
    """Varre offsets relativos e tenta float32/float64 LE/BE + escalas."""
    cand=[]
    rel_min = min(rel_from, rel_to)
    rel_max = max(rel_from, rel_to)
    for rel in range(rel_min, rel_max+1):
        i = base_off + rel
        # float32
        if 0 <= i and i+4 <= len(buf):
            try:
                f_le = struct.unpack_from("<f", buf, i)[0]
                f_be = struct.unpack_from(">f", buf, i)[0]
                for tag, v in (("f32LE", f_le), ("f32BE", f_be)):
                    if safe_float(v):
                        for s in (1,0.1,0.01,0.001,10,100,1000):
                            val = v*s
                            if lo <= val <= hi:
                                cand.append((rel, tag, s, float(val)))
            except Exception:
                pass
        # float64
        if 0 <= i and i+8 <= len(buf):
            try:
                d_le = struct.unpack_from("<d", buf, i)[0]
                d_be = struct.unpack_from(">d", buf, i)[0]
                for tag, v in (("f64LE", d_le), ("f64BE", d_be)):
                    if safe_float(v):
                        for s in (1,0.1,0.01,0.001,10,100,1000):
                            val = v*s
                            if lo <= val <= hi:
                                cand.append((rel, tag, s, float(val)))
            except Exception:
                pass
        # int32/int64 com escalas (LE/BE)
        if 0 <= i and i+4 <= len(buf):
            u32le = int.from_bytes(buf[i:i+4], "little", signed=False)
            s32le = int.from_bytes(buf[i:i+4], "little", signed=True)
            u32be = int.from_bytes(buf[i:i+4], "big", signed=False)
            s32be = int.from_bytes(buf[i:i+4], "big", signed=True)
            for tag, v in (("u32LE",u32le),("s32LE",s32le),("u32BE",u32be),("s32BE",s32be)):
                for div in (1,10,100,1000,10000):
                    val = v / div
                    if lo <= val <= hi:
                        cand.append((rel, tag, f"/{div}", float(val)))
        if 0 <= i and i+8 <= len(buf):
            u64le = int.from_bytes(buf[i:i+8], "little", signed=False)
            s64le = int.from_bytes(buf[i:i+8], "little", signed=True)
            u64be = int.from_bytes(buf[i:i+8], "big", signed=False)
            s64be = int.from_bytes(buf[i:i+8], "big", signed=True)
            for tag, v in (("u64LE",u64le),("s64LE",s64le),("u64BE",u64be),("s64BE",s64be)):
                for div in (1,10,100,1000,10000,100000,1000000):
                    val = v / div
                    if lo <= val <= hi:
                        cand.append((rel, tag, f"/{div}", float(val)))
    return cand

def parse_ascii_candidates(ascii_field, lo, hi):
    # ascii_numbers_nearby tem tokens no formato "VAL@+REL" ou "VAL@-REL"
    out=[]
    for token in ascii_field.split():
        if '@' not in token: 
            # fallback: número puro
            m = ASCII_NUM.fullmatch(token)
            if m:
                val=float(m.group(0))
                if lo <= val <= hi:
                    out.append((None,"ascii","",val))
            continue
        val, off = token.split('@', 1)
        m = ASCII_NUM.fullmatch(val)
        if not m: 
            continue
        try:
            rel = int(off)
        except:
            rel = None
        v = float(m.group(0))
        if lo <= v <= hi:
            out.append((rel,"ascii","",v))
    return out

def load_blocks(csv_path, token):
    """Carrega blocos (timestamp + eventos) do CSV v1.6 (ou similar)."""
    rows=[]
    with open(csv_path, newline='', encoding='utf-8') as f:
        r=csv.DictReader(f)
        for row in r:
            if row.get("token")==token:
                rows.append(row)

    blocks=[]
    cur={"ts_utc":"", "ts_local":"", "offset":"", "events":[]}
    for row in rows:
        if row.get("timestamp_ms"):  # abre novo bloco
            if cur["events"]:
                blocks.append(cur)
            cur={"ts_utc":row.get("timestamp_utc",""),
                 "ts_local":row.get("timestamp_local",""),
                 "offset":row.get("offset_hex",""),
                 "events":[]}
        else:
            cur["events"].append(row)
    if cur["events"]:
        blocks.append(cur)
    return blocks

def main():
    if len(sys.argv) < 2:
        print("uso: python bruteforce_prices.py <csv> [--token dolm2] "
              "[--start ISO] [--end ISO] [--min LO] [--max HI] [--top N] "
              "[--bin BIN] [--radius 96] [--expect 5140,5145.5,...]")
        sys.exit(1)

    # argumentos simples (sem argparse para deixar compacto)
    args = {"--token":"dolm2", "--min":"3000", "--max":"8000", "--top":"6", "--radius":"96"}
    i=2
    while i < len(sys.argv):
        k = sys.argv[i-1] if sys.argv[i-1].startswith("--") else None
        if k in ("--token","--start","--end","--min","--max","--top","--bin","--radius","--expect"):
            args[k]=sys.argv[i]; i+=2
        else:
            i+=1

    csv_path = sys.argv[1]
    token    = args["--token"]
    t0 = parse_iso(args["--start"]) if "--start" in args else None
    t1 = parse_iso(args["--end"])   if "--end"   in args else None
    lo = float(args["--min"])
    hi = float(args["--max"])
    topN = int(args["--top"])
    radius = int(args["--radius"])
    expect = []
    if "--expect" in args:
        for s in args["--expect"].split(","):
            try: expect.append(float(s))
            except: pass

    blocks = load_blocks(csv_path, token)

    # opcional: carregar o binário (para floats/ints); se ausente, só ASCII
    bin_path = args.get("--bin")
    buf = None
    if bin_path:
        with open(bin_path,"rb") as f: buf = f.read()

    # ordenar por ts_local
    blocks.sort(key=lambda b: b["ts_local"] or "")

    for b in blocks:
        if not b["ts_local"]: 
            continue
        dt = parse_iso(b["ts_local"])
        if not within(dt, t0, t1):
            continue
        print(f"\n=== bloco @ {b['ts_local']} (UTC {b['ts_utc']}) offset {b['offset']}  events={len(b['events'])}")
        # ordenar eventos pelo offset para ficar na sequência
        evs = sorted(b["events"], key=lambda r: int(r["offset_hex"],16))
        for ev in evs:
            off_hex = ev["offset_hex"]
            base = int(off_hex,16)
            asc  = ev.get("ascii_numbers_nearby","")
            vari = ev.get("varints_after_token","")
            print(f"\n{off_hex}  ascii≈[{asc[:120]}...]")
            cands = []
            # ASCII candidatos
            cands += parse_ascii_candidates(asc, lo, hi)

            # floats/ints candidatos (se tiver bin)
            if buf is not None:
                cands += try_floats(buf, base+len(token), -radius, +radius, lo, hi)

            # consolidar por valor (valor arredondado a 0.01)
            bucket = defaultdict(list)
            for rel, tag, scale, val in cands:
                key = round(val, 2)
                bucket[key].append((rel, tag, scale))
            # rankear: por proximidade dos "expect" se fornecido; senão por frequência e |rel|
            scored=[]
            for key, lst in bucket.items():
                freq = len(lst)
                best_rel = min((abs(x[0]) for x in lst if x[0] is not None), default=9999)
                if expect:
                    # distância mínima ao conjunto esperado
                    dist = min(abs(key - e) for e in expect)
                else:
                    dist = 0.0
                score = (- (10 if key in expect else 0), dist, -freq, best_rel)
                scored.append((score, key, freq, best_rel, lst))
            scored.sort()
            # imprimir top N
            for _, key, freq, best_rel, lst in scored[:topN]:
                locs = ", ".join([f"{t}@{r:+d}{('/'+str(scale) if scale else '')}" if r is not None else f"{t}"
                                  for (r,t,scale) in lst[:5]])
                print(f"  → cand price={key:.2f}  freq={freq}  best_rel={best_rel:+d}  via {locs}")

if __name__=="__main__":
    main()
