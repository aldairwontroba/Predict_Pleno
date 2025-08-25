# hunt_dolm2_window.py
import csv, sys, re
from datetime import datetime, timezone, timedelta

PRICE_RX = re.compile(r"\b([45]\d{3}\.\d{2})\b")  # dólar ~ 4000–5999.xx (ajuste se precisar)
INT_RX   = re.compile(r"\b\d{1,5}\b")

def parse_iso(ts):
    # aceita "2024-05-17T08:55:21.472000-03:00" e também sem micros
    return datetime.fromisoformat(ts)

def within(dt, a, b):
    return (a is None or dt >= a) and (b is None or dt <= b)

def pick_price(ascii_field):
    # pega primeiro preço plausível (5xxx.xx) preferindo o mais perto do token (ordem já vem por offset)
    # ascii_numbers_nearby está no formato "5140.00@-36 2@+4 ..."
    prices=[]
    for token in ascii_field.split():
        if '@' not in token: continue
        val, off = token.split('@', 1)
        if PRICE_RX.fullmatch(val):
            try:
                offi = int(off)
            except:
                offi = 0
            prices.append((abs(offi), float(val), offi))
    if not prices: return None, None
    prices.sort()
    _, p, rel = prices[0]
    return p, rel

def guess_lot(ascii_field):
    # tenta achar um lote pequeno 1..100 perto do preço (heurística boba)
    nums=[]
    for token in ascii_field.split():
        if '@' not in token: continue
        val, off = token.split('@', 1)
        if val.isdigit():
            n=int(val)
            if 1 <= n <= 100:
                try:
                    offi = int(off)
                except:
                    offi = 0
                nums.append((abs(offi), n))
    if not nums: return None
    nums.sort()
    return nums[0][1]

def guess_broker(varints_field, ascii_field):
    # heurística simples:
    # 1) Se houver "1537@+??,b2" logo após tags conhecidas (4098/11010/8578), pegue algum número ASCII próximo (1..1000) como broker
    # 2) Se não, pegue o menor número ASCII de 1..200 que esteja relativamente perto do token.
    #    (ajuste depois, quando o padrão do campo "corretora" ficar claro)
    ints = varints_field.split()
    tags = set()
    for it in ints:
        v = it.split('@',1)[0]
        try:
            tags.add(int(v))
        except:
            pass

    # fallback por proximidade
    cand=[]
    for token in ascii_field.split():
        if '@' not in token: continue
        val, off = token.split('@', 1)
        if val.isdigit():
            n=int(val)
            if 1 <= n <= 1000:
                try:
                    offi = int(off)
                except:
                    offi = 0
                cand.append((abs(offi), n))
    if not cand: return None
    cand.sort()
    return cand[0][1]

def is_cancel(varints_field):
    # cancelações têm um varint "enorme" logo no começo (9 bytes, começa ~ 47620...)
    parts = varints_field.split()
    if not parts: return False
    head = parts[0]
    v = head.split('@',1)[0]
    try:
        x = int(v)
        return x >= 10**15  # bem grande
    except:
        return False

def label_side(ascii_field, price_rel):
    # ainda não temos o bit "side" decodificado. Heurística:
    # - Para esse feed, os adds de venda costumam aparecer com vários números do lado positivo?
    # Isso é fraco… então deixo "?" por enquanto.
    return "?"  # vamos rotular depois que amarrarmos a tag de side

def main():
    if len(sys.argv) < 2:
        print("uso: python hunt_dolm2_window.py tryd_extract_v16.csv [ini_iso_local] [fim_iso_local]")
        sys.exit(1)

    path = sys.argv[1]
    t0 = parse_iso(sys.argv[2]) if len(sys.argv) > 2 else None
    t1 = parse_iso(sys.argv[3]) if len(sys.argv) > 3 else None

    rows=[]
    with open(path, newline='', encoding='utf-8') as f:
        r=csv.DictReader(f)
        for row in r:
            if row.get("token")!="dolm2": continue
            ts_loc = row.get("timestamp_local","")
            rows.append(row)

    # Agrupar por "blocos": cada linha com timestamp abre um bloco; as sem timestamp pertencem ao último bloco
    blocks=[]
    cur={"ts_utc":"", "ts_local":"", "events":[]}
    for row in rows:
        if row.get("timestamp_ms"):
            # abre novo bloco
            if cur["events"]:
                blocks.append(cur)
            cur={"ts_utc": row.get("timestamp_utc",""),
                 "ts_local": row.get("timestamp_local",""),
                 "events":[]}
        else:
            cur["events"].append(row)
    if cur["events"]:
        blocks.append(cur)

    # filtrar por janela pedida
    out=[]
    for b in blocks:
        if not b["ts_local"]: continue
        dt = parse_iso(b["ts_local"])
        if not within(dt, t0, t1): continue
        for ev in b["events"]:
            price, prel = pick_price(ev["ascii_numbers_nearby"])
            lot       = guess_lot(ev["ascii_numbers_nearby"])
            broker    = guess_broker(ev["varints_after_token"], ev["ascii_numbers_nearby"])
            cancel    = is_cancel(ev["varints_after_token"])
            side      = label_side(ev["ascii_numbers_nearby"], prel)
            out.append({
                "ts_local_block": b["ts_local"],
                "offset": ev["offset_hex"],
                "cancel": "Y" if cancel else "N",
                "price": price,
                "lot": lot,
                "broker_guess": broker,
                "side": side,
                "ascii": ev["ascii_numbers_nearby"],
                "varints": ev["varints_after_token"],
            })

    # ordenar por offset para ficar exatamente na sequência que você vê
    out.sort(key=lambda x: int(x["offset"],16))

    # imprimir enxuto
    for e in out:
        print(f'{e["offset"]}  {e["ts_local_block"]}  '
              f'{"CANCEL" if e["cancel"]=="Y" else "ADD  "}  '
              f'price={e["price"]}  lot={e["lot"]}  broker≈{e["broker_guess"]}  side={e["side"]}')

if __name__=="__main__":
    main()
