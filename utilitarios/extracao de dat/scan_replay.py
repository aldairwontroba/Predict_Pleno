#!/usr/bin/env python3
import argparse, struct

# Varint estilo protobuf/kryo
def read_uvarint(buf, i):
    x = 0
    s = 0
    start = i
    while i < len(buf):
        b = buf[i]
        i += 1
        x |= (b & 0x7F) << s
        if (b & 0x80) == 0:
            return x, i, i-start
        s += 7
        if s > 63:
            break
    return None, start, 0

def zigzag_decode(n):
    # int -> signed
    return (n >> 1) ^ -(n & 1)

def try_back_len(buf, pos, max_back=5):
    # Tenta ver se bytes ANTES da string formam um uvarint que
    # é exatamente o tamanho da string (ex.: 5 para 'wdom2')
    for b in range(1, max_back+1):
        v, end, ln = read_uvarint(buf, pos-b)
        if ln == b and v is not None and v <= 256:
            return v, pos-b, b
    return None, None, None

def dump_varints(buf, pos, count=12):
    out=[]
    i=pos
    for _ in range(count):
        v, j, ln = read_uvarint(buf, i)
        if ln == 0: break
        out.append((i, ln, v, zigzag_decode(v)))
        i = j
    return out

def around(buf, pos, radius=32):
    s = max(0, pos - radius)
    e = min(len(buf), pos + radius)
    return buf[s:e], s

def hexdump(b, base=0, width=16):
    lines=[]
    for i in range(0, len(b), width):
        chunk=b[i:i+width]
        hexp=" ".join(f"{x:02X}" for x in chunk)
        asc="".join(chr(x) if 32<=x<=126 else "." for x in chunk)
        lines.append(f"{base+i:08X}  {hexp:<{3*width}}  {asc}")
    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--tokens", nargs="+", default=["wdom2","dolm2","frp","ptxusd","ptxusdc"])
    ap.add_argument("--hits", type=int, default=20)
    ap.add_argument("--ahead-varints", type=int, default=12)
    ap.add_argument("--back-bytes", type=int, default=8)
    args=ap.parse_args()

    with open(args.file,"rb") as f:
        buf=f.read()

    tokens=[t.encode("ascii") for t in args.tokens]
    total=0
    for tok in tokens:
        off=0
        while True:
            j = buf.find(tok, off)
            if j<0: break
            total+=1
            # tenta achar varint de comprimento imediatamente antes
            L, lpos, llen = try_back_len(buf, j, max_back=args.back_bytes)
            ctx, base = around(buf, j, radius=48)

            print("="*100)
            print(f"TOKEN '{tok.decode()}' @ 0x{j:08X}")
            print(hexdump(ctx, base))
            if L is not None:
                print(f"- Encontrado varint de comprimento antes: len={L} em 0x{lpos:08X} (bytes={llen})")
            else:
                print("- NÃO achei varint claro de comprimento imediatamente antes.")

            # varints logo DEPOIS da string
            after_start = j + len(tok)
            vlist = dump_varints(buf, after_start, count=args.ahead_varints)
            if vlist:
                print(f"- Varints após a string (interpretação: uvarint e zigzag -> signed):")
                for (i, ln, u, s) in vlist:
                    print(f"  0x{i:08X} (+{i-after_start:02d})  len={ln}  u={u}  s={s}")
            else:
                print("- Não consegui ler varints consistentes após a string.")

            # tenta interpretar alguns pares como possíveis floats/doubles alinhados
            # (nem sempre fará sentido — só para pistas)
            for align in (0,1,2,3,4):
                p = after_start + align
                if p+8 <= len(buf):
                    f_le = struct.unpack_from("<f", buf, p)[0]
                    d_le = struct.unpack_from("<d", buf, p)[0]
                    f_be = struct.unpack_from(">f", buf, p)[0]
                    d_be = struct.unpack_from(">d", buf, p)[0]
                    def ok(x): 
                        return (x==x) and (abs(x) < 1e7)
                    if ok(f_le) or ok(d_le) or ok(f_be) or ok(d_be):
                        print(f"- ALINHO {align}: floatLE={f_le:.6g} doubleLE={d_le:.6g} | floatBE={f_be:.6g} doubleBE={d_be:.6g}")
                        break

            off = j + len(tok)
            if total >= args.hits: break

    if total==0:
        print("Nenhum token encontrado; tente adicionar --tokens com outras strings que você viu no hexdump.")

if __name__ == "__main__":
    main()
