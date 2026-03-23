# extract_tryd_ascii_csv_v16.py
# Scanner "leve" para achar tokens (wdom2, dolm2, etc.), varints logo após,
# timestamp em ms (heurística) e números ASCII próximos (preços/qtde).
#
# Melhorias v1.6:
# - Converte timestamp_ms -> timestamp_utc e timestamp_local (via --tz, padrão -03:00).
# - Captura offsets relativos dos números ASCII, ajudando a entender o "layout".
# - Opcional: tenta detectar um varint de "comprimento" imediatamente antes do token
#   (quando for um valor pequeno e plausível para string length), só para registrar.
#
# Observação: este script NÃO decodifica o formato inteiro; ele junta pistas
# consistentes para comparar com o replay/planilha e guiar a engenharia reversa.

import argparse
import re
import csv
from datetime import datetime, timezone, timedelta

DEFAULT_TOKENS = [b"wdom2", b"dolm2", b"frp", b"ptxusd"]

# ------------ utils de varint (protobuf/kryo-like) ------------

def read_uvarint(buf, i):
    """Lê um uvarint começando na posição i.
    Retorna (valor, prox_pos, bytes_lidos) ou (None, i, 0) se falhou."""
    x = 0
    s = 0
    start = i
    while i < len(buf):
        b = buf[i]
        i += 1
        x |= (b & 0x7F) << s
        if (b & 0x80) == 0:
            return x, i, i - start
        s += 7
        if s > 63:
            break
    return None, start, 0

# ------------ timezone helpers ------------

def parse_tz_offset(tz_str):
    """
    Aceita formatos como:
      - "-03", "-3", "+02", "+2"
      - "-03:00", "+02:30"
      - "Z" (UTC)
    Retorna um datetime.timezone.
    """
    if tz_str is None:
        return timezone.utc

    tz_str = tz_str.strip().upper().replace("’","'").replace("“","").replace("”","").replace('"','').replace("'", "")
    if tz_str == "Z":
        return timezone.utc

    # Deve começar com + ou -
    if not tz_str or tz_str[0] not in "+-":
        raise ValueError(f"Offset inválido: {tz_str}")

    sign = 1 if tz_str[0] == "+" else -1
    core = tz_str[1:]

    if ":" in core:
        hh_s, mm_s = core.split(":", 1)
        if not hh_s:
            raise ValueError(f"Offset inválido: {tz_str}")
        hh = int(hh_s)
        mm = int(mm_s) if mm_s else 0
    else:
        # Só horas
        hh = int(core)
        mm = 0

    delta = timedelta(hours=sign * hh, minutes=sign * mm)
    return timezone(delta)

def ts_ms_to_iso(ts_ms, tz):
    """Converte timestamp em milissegundos para (iso_utc, iso_local)."""
    if ts_ms is None:
        return "", ""
    # ts_ms é int em milissegundos
    dt_utc = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
    dt_loc = dt_utc.astimezone(tz)
    return dt_utc.isoformat(), dt_loc.isoformat()

# ------------ scanner principal ------------

def try_back_len_equal_token(buf, pos, tok_len, max_back=8):
    """
    Heurística: alguns encoders guardam um varint "len" imediatamente antes da string.
    Aqui verificamos se, alguns bytes antes do token, existe um uvarint == tok_len.
    Isso NÃO é framing do evento; é só uma pista de como a string foi gravada.
    """
    for b in range(1, max_back + 1):
        v, _, ln = read_uvarint(buf, pos - b)
        if ln == b and v == tok_len:
            return v, pos - b, b
    return None, None, None

def scan_file(path, tokens, window=80, outcsv="out.csv",
              ahead_varints=12, tz_offset="-03:00"):
    with open(path, "rb") as f:
        data = f.read()

    # números ASCII do tipo 5135, 5135.00, 0.25, etc. (limites pensados p/ preço/qty/id curtos)
    num_re = re.compile(rb"(?<![0-9.])\d{1,6}(?:\.\d{1,4})?(?![0-9.])")

    tz = parse_tz_offset(tz_offset)

    rows = []
    for tok in tokens:
        off = 0
        while True:
            j = data.find(tok, off)
            if j < 0:
                break

            token_len = len(tok)

            # 1) Heurística de "len" imediatamente antes do token (só como pista)
            tok_len_uvar, len_pos, len_bytes = try_back_len_equal_token(
                data, j, token_len, max_back=8
            )

            # 2) Ler varints "depois do token"
            pos = j + token_len
            seq = []
            for _ in range(ahead_varints):
                v, pos2, ln = read_uvarint(data, pos)
                if ln == 0:
                    break
                seq.append((pos, ln, v))   # (offset, bytes_lidos, valor)
                pos = pos2

            # 3) Achar timestamp_ms por heurística (ms ~ 13 dígitos)
            ts_ms = None
            for (_, _, v) in seq:
                if 1_600_000_000_000 <= v <= 2_200_000_000_000:
                    ts_ms = v
                    break
            ts_utc, ts_local = ts_ms_to_iso(ts_ms, tz)

            # 4) Capturar números ASCII ao redor do token, com offsets relativos
            s = max(0, j - window)
            e = min(len(data), j + window)
            around = data[s:e]
            ascii_hits = []
            base_off = s
            for m in num_re.finditer(around):
                num = m.group(0).decode("ascii")
                abs_off = base_off + m.start()
                rel_off = abs_off - j  # relativo ao token
                ascii_hits.append(f"{num}@{rel_off:+d}")

            # 5) Preparar dump sintético dos varints (valor@+rel,bytes=ln)
            varint_dump = []
            for (p, ln, v) in seq:
                rel = p - (j + token_len)
                varint_dump.append(f"{v}@{rel:+d},b{ln}")

            rows.append({
                "offset_hex": f"0x{j:08X}",
                "token": tok.decode("ascii", "ignore"),
                "timestamp_ms": ts_ms if ts_ms is not None else "",
                "timestamp_utc": ts_utc,
                "timestamp_local": ts_local,
                # logs auxiliares para engenharia reversa:
                "varints_after_token": " ".join(varint_dump[:24]),
                "ascii_numbers_nearby": " ".join(ascii_hits[:20]),
                "len_before_token": (str(tok_len_uvar) if tok_len_uvar is not None else ""),
                "len_before_token_pos_hex": (f"0x{len_pos:08X}" if len_pos is not None else ""),
                "len_before_token_bytes": (str(len_bytes) if len_bytes is not None else "")
            })

            off = j + token_len

    # 6) Gravar CSV
    with open(outcsv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "offset_hex",
                "token",
                "timestamp_ms",
                "timestamp_utc",
                "timestamp_local",
                "varints_after_token",
                "ascii_numbers_nearby",
                "len_before_token",
                "len_before_token_pos_hex",
                "len_before_token_bytes",
            ]
        )
        w.writeheader()
        w.writerows(rows)

    print(f"ok: {len(rows)} hits → {outcsv}")

# ------------ CLI ------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("file", help="caminho do .dat")
    ap.add_argument("--out", default="tryd_extract_v16.csv", help="arquivo CSV de saída")
    ap.add_argument("--window", type=int, default=80, help="janela (bytes) para capturar números ASCII ao redor do token")
    ap.add_argument("--ahead-varints", type=int, default=12, help="quantos varints tentar ler após o token")
    ap.add_argument("--tokens", nargs="*", default=[t.decode() for t in DEFAULT_TOKENS], help="tokens a procurar")
    ap.add_argument("--tz", default="-03:00", help="offset local ex.: -03:00, -03, +02:30, Z")
    args = ap.parse_args()

    scan_file(
        args.file,
        [t.encode("ascii") for t in args.tokens],
        window=args.window,
        outcsv=args.out,
        ahead_varints=args.ahead_varints,
        tz_offset=args.tz,
    )
