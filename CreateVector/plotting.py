from typing import Dict, Tuple, List, Iterable, Optional, Any
import matplotlib.pyplot as plt
from datetime import datetime, timezone
import matplotlib.dates as mdates
from collections import Counter, defaultdict
from datetime import datetime
import math
from matplotlib.cm import get_cmap
from zoneinfo import ZoneInfo  # py>=3.9

def _as_list_flags(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    # se for string única, vira lista
    return [x]

def _primary_reason(ev):
    # se você estiver salvando 'end_primary', respeita; senão, 1º da lista
    if "end_primary" in ev and ev["end_primary"]:
        return ev["end_primary"]
    flags = _as_list_flags(ev.get("end_reason"))
    return flags[0] if flags else None

def _percentiles(vals, ps=(0, 25, 50, 75, 90, 95, 99, 100)):
    if not vals:
        return {p: 0 for p in ps}
    xs = sorted(vals)
    res = {}
    for p in ps:
        if p <= 0:   res[p] = xs[0]
        elif p >= 100: res[p] = xs[-1]
        else:
            k = (len(xs)-1) * (p/100.0)
            f = math.floor(k)
            c = math.ceil(k)
            if f == c:
                res[p] = xs[int(k)]
            else:
                res[p] = xs[f] + (xs[c]-xs[f]) * (k - f)
    return res

def print_event_stats(eventos):
    total = len(eventos)
    if total == 0:
        print("Sem eventos.")
        return

    # ===== 1) Causa primária (end_reason[0] ou end_primary) =====
    prim = [_primary_reason(ev) for ev in eventos]
    reason_cnt = Counter(prim)
    print("\n== Causa primária do encerramento ==")
    for k, c in reason_cnt.most_common():
        pct = 100.0 * c / total
        print(f"{(k or 'unknown'):>10}: {c:5d}  ({pct:5.1f}%)")

    # ===== 2) Frequência de TODAS as flags ativas no fechamento =====
    all_flags = []
    for ev in eventos:
        all_flags.extend(_as_list_flags(ev.get("end_reason")))
    flag_cnt = Counter(all_flags)
    print("\n== Flags ativas no encerramento (todas) ==")
    for k, c in flag_cnt.most_common():
        pct = 100.0 * c / total
        print(f"{k:>10}: {c:5d}  ({pct:5.1f}%)")

    # ===== 3) Eventos por hora =====
    per_hour = Counter()
    for ev in eventos:
        try:
            h = datetime.fromtimestamp(int(ev["start"])).hour
        except Exception:
            # fallback: se 'start' não for epoch, tenta usar modulo 24
            h = int(ev.get("start", 0)) % 24
        per_hour[h] += 1

    print("\n== Eventos por hora ==")
    for h in sorted(per_hour):
        c = per_hour[h]; pct = 100.0 * c / total
        print(f"{h:02d}h: {c:5d}  ({pct:5.1f}%)")

    # ===== 4) Duração dos eventos (segundos) =====
    durs = [int(ev.get("dur", 0)) for ev in eventos]
    P = _percentiles(durs)
    mean = sum(durs)/total if total else 0
    print("\n== Duração dos eventos (s) ==")
    print(f"n={total}  mean={mean:0.1f}  min={P[0]:.0f}  p25={P[25]:.0f}  p50={P[50]:.0f}  "
          f"p75={P[75]:.0f}  p90={P[90]:.0f}  p95={P[95]:.0f}  p99={P[99]:.0f}  max={P[100]:.0f}")

    # ===== 5) Duração por causa primária =====
    durs_by_cause = defaultdict(list)
    for ev in eventos:
        durs_by_cause[_primary_reason(ev)].append(int(ev.get("dur", 0)))

    print("\n== Duração por causa primária (p50 / p75 / p90) ==")
    for cause, lst in sorted(durs_by_cause.items(), key=lambda kv: (kv[0] is None, str(kv[0]))):
        if not lst:
            continue
        P = _percentiles(lst, ps=(50, 75, 90))
        print(f"{(cause or 'unknown'):>10}:  p50={P[50]:.0f}s  p75={P[75]:.0f}s  p90={P[90]:.0f}s")

    # ===== 6) (Opcional) Combos de flags no encerramento =====
    combos = Counter(tuple(sorted(set(_as_list_flags(ev.get("end_reason"))))) for ev in eventos)
    print("\n== Combos de flags no encerramento (top 10) ==")
    for combo, c in combos.most_common(10):
        tag = ",".join(combo) if combo else "none"
        pct = 100.0 * c / total
        print(f"{tag:>30}: {c:5d}  ({pct:5.1f}%)")


    # ================== PLOT TICK-A-TICK (tempo real no eixo X) ==================


LOCAL_TZ = ZoneInfo("America/Sao_Paulo")

def _build_event_sec_map(eventos_list):
    """sec -> id_evento; fora de evento = -1."""
    evmap = {}
    for k, ev in enumerate(eventos_list):
        s0 = int(ev["start"]); s1 = int(ev["end"])
        for s in range(s0, s1 + 1):
            evmap[s] = k
    return evmap

def _tick_series_time(sec_map, evmap):
    """
    Constrói série tick-a-tick:
    xs_time: datetimes (espalhando trades dentro do segundo)
    ys: preços
    ev_ids: id do evento (ou -1)
    sec_map: dict[int -> list[trade]], trade[0] = price
    """
    xs_time, ys, ev_ids = [], [], []
    for s in sorted(sec_map.keys()):
        trades = sec_map[s]
        n = len(trades)
        if n == 0:
            continue
        # espalha no intervalo [s, s+1) p/ preservar ordem intrassegundo
        for k, tr in enumerate(trades):
            frac = (k + 1) / (n + 1)    # 0<frac<1
            t_real = s + frac
            # epoch segundos -> timezone local
            dt_utc = datetime.fromtimestamp(t_real, tz=timezone.utc)
            dt_loc = dt_utc.astimezone(LOCAL_TZ)
            xs_time.append(dt_loc)
            ys.append(float(tr[0]))
            ev_ids.append(evmap.get(s, -1))
    return xs_time, ys, ev_ids

def _choose_color(eid, base_cmap, n_colors=20):
    """Cores estáveis por evento; -1 (fora) fica cinza."""
    if eid < 0:
        return (0.6, 0.6, 0.6, 0.8)  # cinza
    # usa tab20 como paleta base e cicla
    idx = eid % n_colors
    return base_cmap(idx / max(1, n_colors - 1))

def _plot_segments_time(xs, ys, ev_ids, eventos=None,
                        title="Tick-a-tick — cores por evento (tempo real)",
                        shade_events=True):
    if not xs:
        print("Sem dados para plot.")
        return

    # quebra em segmentos contíguos de mesmo evento (para desenhar com 1 cor cada bloco)
    segments = []
    start = 0
    last = ev_ids[0]
    for i in range(1, len(xs)):
        if ev_ids[i] != last:
            segments.append((xs[start:i], ys[start:i], last))
            start = i
            last = ev_ids[i]
    segments.append((xs[start:], ys[start:], last))

    fig, ax = plt.subplots(figsize=(14, 5))

    # sombreia janelas dos eventos (opcional)
    if shade_events and eventos:
        for k, ev in enumerate(eventos):
            t0 = datetime.fromtimestamp(int(ev["start"]), tz=timezone.utc).astimezone(LOCAL_TZ)
            t1 = datetime.fromtimestamp(int(ev["end"]) + 1, tz=timezone.utc).astimezone(LOCAL_TZ)
            ax.axvspan(t0, t1, color=_choose_color(k, get_cmap("tab20")), alpha=0.08, lw=0)

    cmap = get_cmap("tab20")
    for sx, sy, eid in segments:
        ax.plot(sx, sy, linewidth=0.9, color=_choose_color(eid, cmap))

    ax.set_title(title)
    ax.set_xlabel("tempo (America/Sao_Paulo)")
    ax.set_ylabel("preço")

    # formato de datas bonitinho
    loc = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))

    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    plt.show()

def print_event(ev, label="A", columns=4, colw=20, line_width=110):
    """
    Imprime SOMENTE ev["vector"], com:
      - sumário em colunas fixas
      - detalhes longos (vpt_* e absorptions_*) em linhas separadas
    Ajuste columns / colw / line_width conforme o seu terminal.
    """
    import textwrap as tw

    vec = ev.get("vector", None)
    if not isinstance(vec, dict) or not vec:
        print("═" * line_width)
        print(f"[EVENT {label}] (sem vector)")
        print("═" * line_width + "\n")
        return

    # ---------- helpers ----------
    def hbar(ch="═", w=line_width): return ch * max(20, w)

    def fmt_num(x, nd=2):
        try:
            if x is None: return "-"
            if isinstance(x, int): return f"{x:d}"
            return f"{float(x):.{nd}f}"
        except Exception:
            return "-"

    def fmt_pct(x, nd=1):
        try:
            return f"{100.0 * float(x):.{nd}f}%"
        except Exception:
            return "-"

    def fmt_stats(dct, order=("n","mean","max","sum","ticks_mean","ticks_max","vel_mean")):
        """
        Converte dicionários de stats em string amigável.
        Suporta formatos de _stats_list (mean,max,sum,n) e _abs_stats (count,ticks_mean,ticks_max,vel_mean).
        """
        if not isinstance(dct, dict) or not dct:
            return "-"
        # normaliza alguns aliases
        dd = dict(dct)
        if "count" in dd and "n" not in dd:
            dd["n"] = dd.pop("count")

        pretty = {
            "n": "n",
            "mean": "μ",
            "max": "max",
            "sum": "Σ",
            "ticks_mean": "ticksμ",
            "ticks_max": "ticksMax",
            "vel_mean": "velμ",
        }
        parts = []
        for k in order:
            if k in dd:
                parts.append(f"{pretty.get(k,k)}={fmt_num(dd[k], 2)}")
        # extras
        for k, v in dd.items():
            if k not in order:
                parts.append(f"{k}={fmt_num(v, 2)}")
        return ", ".join(parts)

    def decimals_for(key, val):
        """Precisão por campo (default=2)."""
        int_keys = {
            "dt", "g_b", "g_s", "g_n",
            "max_streak_buy", "max_streak_sell"
        }
        zero_dec_vols = {
            "b_vol", "s_vol", "t_vol", "d_vol",
            "top_buy_vol_wdo", "top_sell_vol_wdo",
            "top_buy_vol_dol", "top_sell_vol_dol"
        }
        four_dec = {"ease_of_move"}
        three_dec = {"efrang"}
        one_dec = set()
        if key in int_keys or isinstance(val, int):
            return 0
        if key in zero_dec_vols:
            return 0
        if key in four_dec:
            return 4
        if key in three_dec:
            return 3
        if key in one_dec:
            return 1
        return 2

    def is_percent_key(k):
        return k in {"buy_share_wdo", "sell_share_wdo"}

    def rowize(pairs, ncols=columns, width=colw):
        """Lista de (k, v) -> linhas com n colunas fixas, separadas por ' | '."""
        cells = [f"{k}: {v}" for k, v in pairs]
        lines = []
        for i in range(0, len(cells), ncols):
            chunk = cells[i:i+ncols]
            padded = []
            for c in chunk:
                if len(c) > width:
                    c = tw.shorten(c, width=width, placeholder="…")
                padded.append(f"{c:<{width}}")
            lines.append(" | ".join(padded).rstrip())
        return lines

    def print_block(title, pairs, ncols=columns, width=colw):
        if not pairs: return
        print(f"[{title}]")
        for ln in rowize(pairs, ncols, width):
            print(ln)
        print("")

    def print_kv_lines(title, kv_pairs, indent="  "):
        print(f"[{title}]")
        for k, v in kv_pairs:
            wrapped = tw.wrap(str(v), width=line_width - len(indent) - len(k) - 2)
            if not wrapped:
                print(f"{indent}{k}: -")
                continue
            print(f"{indent}{k}: {wrapped[0]}")
            for cont in wrapped[1:]:
                print(f"{indent}{'':{len(k)}}  {cont}")
        print("")

    # ---------- ordem sugerida ----------
    order = [
        "dt",
        "rng", "efrang",
        "dp", "dmx", "dmm", "dmt", "dmv",
        "dvwap", "ddayo", "d5m", "d30m",
        "pctx0", "pctx1", "pctx2", "pctx3", "pctx4",
        "b_vol", "s_vol", "t_vol", "d_vol", "avg_vol",
        "g_b", "g_s", "g_n", "rate_avg",
        "max_streak_buy", "max_streak_sell",
        "ema_vol_total", "ema_order_rate",
        "d_ticks_max", "buy_share_wdo", "sell_share_wdo",
        "ease_of_move",
        "top_buy_vol_wdo", "top_sell_vol_wdo",
        "top_buy_vol_dol", "top_sell_vol_dol",
        # longos (irão para DETAILS):
        "vol_per_tick_up_a", "vol_per_tick_dn_a",
        "vol_per_tick_up_b", "vol_per_tick_dn_b",
        "absorptions_buy", "absorptions_sell",
    ]

    # chaves realmente presentes
    present = [k for k in order if k in vec]
    # quaisquer extras não previstos (garantia futura)
    extras = [k for k in vec.keys() if k not in present]
    present += sorted(extras)

    # separe “longos” dos “curtos”
    long_keys = {
        "vol_per_tick_up_a", "vol_per_tick_dn_a",
        "vol_per_tick_up_b", "vol_per_tick_dn_b",
        "absorptions_buy", "absorptions_sell",
    }

    summary_pairs = []
    detail_pairs  = []

    for k in present:
        v = vec[k]
        if isinstance(v, dict):
            # stats → vão para detalhes por padrão
            detail_pairs.append((k, fmt_stats(v)))
            continue

        if is_percent_key(k):
            summary_pairs.append((k, fmt_pct(v, 1)))
        else:
            nd = decimals_for(k, v)
            summary_pairs.append((k, fmt_num(v, nd)))

    # move explicitamente os long_keys para DETAILS (se vieram como não-dict)
    moved = []
    for k in list(summary_pairs):
        pass  # (já tratado acima)

    # ---------- impressão ----------
    print(hbar("═"))
    print(f"[EVENT {label}]  VECTOR")
    print(hbar("─"))

    # sumário em colunas
    print_block("SUMMARY", summary_pairs)

    # detalhes longos (vpt_* e absorptions_*)
    if any(k in vec for k in long_keys):
        details_fmt = []
        for k in ["vol_per_tick_up_a","vol_per_tick_dn_a","vol_per_tick_up_b","vol_per_tick_dn_b",
                  "absorptions_buy","absorptions_sell"]:
            if k in vec:
                val = vec[k]
                if isinstance(val, dict):
                    details_fmt.append((k, fmt_stats(val)))
                else:
                    details_fmt.append((k, fmt_num(val, 2)))
        print_kv_lines("DETAILS", details_fmt)

    print(hbar("═"))
    print("")  # espaço entre eventos
