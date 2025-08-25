#!/usr/bin/env python3
import argparse, struct, csv, re
from datetime import datetime, timezone, timedelta

# ---------- util de varint estilo protobuf/kryo ----------
def read_uvarint(buf, i):
    x = 0; s = 0; start = i
    while i < len(buf):
        b = buf[i]; i += 1
        x |= (b & 0x7F) << s
        if (b & 0x80) == 0:
            return x, i, i-start
        s += 7
        if s > 63: break
    return None, start, 0

def zigzag(n): return (n >> 1) ^ -(n & 1)

def dump_varints(buf, pos, count=12):
    out=[]; i=pos
    for _ in range(count):
        v, j, ln = read_uvarint(buf, i)
        if ln == 0: break
        out.append((i, ln, v, zigzag(v)))
        i = j
    return out

def try_back_len(buf, pos, max_back=8):
    for b in range(1, max_back+1):
        v, end, ln = read_uvarint(buf, pos-b)
        if ln == b and v is not None and v <= 256:
            return v, pos-b, b
    return None, None, None

def hexdump(b, base=0, width=16):
    lines=[]
    for i in range(0, len(b), width):
        chunk=b[i:i+width]
        hexp=" ".join(f"{x:02X}" for x in chunk)
        asc="".join(chr(x) if 32<=x<=126 else "." for x in chunk)
        lines.append(f"{base+i:08X}  {hexp:<{3*width}}  {asc}")
    return "\n".join(lines)

def ok_float(x): return (x==x) and (abs(x) < 1e7)

# ---------- timestamps ----------
MS_MIN = int(datetime(2008,1,1, tzinfo=timezone.utc).timestamp()*1000)
MS_MAX = int(datetime(2035,1,1, tzinfo=timezone.utc).timestamp()*1000)

def parse_tz(s: str) -> timezone:
    s = s.strip().replace("’", "'").replace("−","-")
    # formatos aceitos: -03:00 | -3 | -0300 | +00 | Z
    if s.upper() == "Z": return timezone.utc
    m = re.fullmatch(r'([+-])\s*(\d{1,2})(?::?(\d{2}))?$', s)
    if not m: raise ValueError(f"TZ inválido: {s!r}")
    sign = -1 if m.group(1) == "-" else 1
    hh = int(m.group(2))
    mm = int(m.group(3) or 0)
    return timezone(sign * timedelta(hours=hh, minutes=mm))

def ms_to_dt(ms: int):
    # seguro para ms (não confunde com datetime)
    return datetime.fromtimestamp(ms/1000.0, tz=timezone.utc)

def find_ms_timestamps(buf, start, end):
    hits = []
    # varre cada offset da janela; lê u64 LE e BE
    for p in range(start, max(start, end-8)+1):
        u64_le = struct.unpack_from("<Q", buf, p)[0]
        if MS_MIN <= u64_le <= MS_MAX:
            hits.append(("u64LE", p, u64_le))
        u64_be = struct.unpack_from(">Q", buf, p)[0]
        if MS_MIN <= u64_be <= MS_MAX:
            hits.append(("u64BE", p, u64_be))
    # remove duplicatas por (pos, valor)
    seen=set(); uniq=[]
    for t in hits:
        k=(t[0],t[1],t[2])
        if k not in seen:
            uniq.append(t); seen.add(k)
    # prioriza BE (foi o que já vimos em dumps) e menor offset
    uniq.sort(key=lambda x: (0 if x[0]=="u64BE" else 1, x[1]))
    return uniq

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="arquivo .dat")
    ap.add_argument("--tokens", nargs="+", default=["wdom2","dolm2","frp","ptxusd","ptxusdc"])
    ap.add_argument("--window", type=int, default=160, help="bytes após o token para inspecionar")
    ap.add_argument("--ahead-varints", type=int, default=12)
    ap.add_argument("--back-bytes", type=int, default=8)
    ap.add_argument("--tz", default=None, help="ex: -03:00, -3, -0300, Z")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--max", type=int, default=999999, help="limite de ocorrências por execução")
    args = ap.parse_args()

    tz_local = parse_tz(args.tz) if args.tz else None

    with open(args.path,"rb") as f:
        buf=f.read()

    tokens=[t.encode("ascii") for t in args.tokens]
    total=0

    csv_rows=[]
    for tok in tokens:
        off=0
        while True:
            j = buf.find(tok, off)
            if j<0: break
            total += 1
            after = j + len(tok)
            end = min(len(buf), after + args.window)

            # contexto p/ visualização
            s = max(0, j-48); e = min(len(buf), j+48)
            ctx = hexdump(buf[s:e], s)

            L, lpos, llen = try_back_len(buf, j, max_back=args.back_bytes)
            vlist = dump_varints(buf, after, count=args.ahead_varints)

            # candidates de timestamp (ms desde epoch)
            ts_hits = find_ms_timestamps(buf, after, end)

            # “sniff” de floats/doubles (pistas)
            align_info = ""
            for align in (0,1,2,3,4):
                p = after + align
                if p+8 <= len(buf):
                    f_le = struct.unpack_from("<f", buf, p)[0]
                    d_le = struct.unpack_from("<d", buf, p)[0]
                    f_be = struct.unpack_from(">f", buf, p)[0]
                    d_be = struct.unpack_from(">d", buf, p)[0]
                    if ok_float(f_le) or ok_float(d_le) or ok_float(f_be) or ok_float(d_be):
                        align_info = f"ALINHO {align}: floatLE={f_le:.6g} doubleLE={d_le:.6g} | floatBE={f_be:.6g} doubleBE={d_be:.6g}"
                        break

            # imprime um resumo enxuto no console
            print("="*100)
            print(f"TOKEN '{tok.decode()}' @ 0x{j:08X}")
            print(ctx)
            if L is not None:
                print(f"- Varint de comprimento antes: len={L} em 0x{lpos:08X} (bytes={llen})")
            else:
                print("- Sem varint de comprimento imediatamente antes.")

            if vlist:
                print("- Varints após o token:")
                for (i, ln, u, sgn) in vlist:
                    print(f"  0x{i:08X} (+{i-after:02d}) len={ln} u={u} s={sgn}")
            else:
                print("- Não consegui ler varints consistentes após o token.")

            if ts_hits:
                print("- Timestamps (ms desde epoch) na janela:")
                for (kind, pos, ms) in ts_hits[:3]:
                    dt_utc = ms_to_dt(ms)
                    if tz_local:
                        dt_loc = dt_utc.astimezone(tz_local)
                        print(f"  {kind}@0x{pos:08X}  {ms}  UTC={dt_utc.isoformat()}  Local={dt_loc.isoformat()}")
                    else:
                        print(f"  {kind}@0x{pos:08X}  {ms}  UTC={dt_utc.isoformat()}")
            else:
                print("- Nenhum candidato óbvio a timestamp ms na janela.")

            if align_info:
                print("-", align_info)

            # linha p/ CSV (pega só o 1º timestamp da janela, se existir)
            ts_ms = ts_hits[0][2] if ts_hits else ""
            ts_utc = ms_to_dt(ts_ms).isoformat() if ts_ms != "" else ""
            ts_loc = ms_to_dt(ts_ms).astimezone(tz_local).isoformat() if (ts_ms != "" and tz_local) else ""
            csv_rows.append({
                "offset_hex": f"0x{j:08X}",
                "token": tok.decode(),
                "ts_ms": ts_ms,
                "timestamp_utc": ts_utc,
                "timestamp_local": ts_loc,
                "varints": " ".join(str(v[2]) for v in vlist[:12]),
            })

            off = j + len(tok)
            if total >= args.max: break

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["offset_hex","token","ts_ms","timestamp_utc","timestamp_local","varints"])
            w.writeheader(); w.writerows(csv_rows)

    if total==0:
        print("Nenhum token encontrado. Tente --tokens com outras strings vistas no hexdump.")

if __name__ == "__main__":
    main()
