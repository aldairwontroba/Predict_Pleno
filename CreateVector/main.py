from __future__ import annotations

import argparse
from typing import List, Tuple
import sys
import numpy as np
import os
import sys
from pathlib import Path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from segmentation import SegParams, EventSegmenter

from data_processing import (
    process_day,
    lot_multiplier,
)
from plotting import (
    plot_event_distributions,
    plot_relationships,
    plot_counts_by_hour_and_reason,
    print_top_outliers,
    plot_tick_series,
    print_event,
)

def parse_pair(value: str) -> Tuple[str, str]:
    parts = value.split(",") if "," in value else value.split()
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("pair must consist of two comma or space separated symbols, e.g. 'wdo dol'")
    return parts[0], parts[1]

def realtime_process(pair: Tuple[str, str]):
    import ctypes
    import mmap
    import time
    import win32event
    import win32con
    import pywintypes

    TICK_SIZE = {"wdo":0.5, "dol":0.5, "win":5.0, "ind":5.0}
    sym_a, sym_b = pair[0].lower(), pair[1].lower()
    tick_a, tick_b = TICK_SIZE[sym_a], TICK_SIZE[sym_b]

    # === 3) INSTANCIA O SEGMENTER COM PARAMS DO PAR ===
    seg = EventSegmenter(SegParams(
        tick_a=tick_a, tick_b=tick_b,
        lot_mult_a=lot_multiplier(sym_a),
        lot_mult_b=lot_multiplier(sym_b),
    ))
    seg.reset()

    # === Definição da struct compartilhada ===
    class MyPyStruct(ctypes.Structure):
        _fields_ = [
            ("bookdol", ctypes.c_float * (10 * 4)),
            ("bookwdo", ctypes.c_float * (10 * 4)),

            ("n_trades_dol", ctypes.c_int),
            ("trades_dol", ctypes.c_float * (2000 * 5)),

            ("n_trades_wdo", ctypes.c_int),
            ("trades_wdo", ctypes.c_float * (2000 * 5)),

            ("cot", ctypes.c_float * 6),
        ]

    sizeof_struct = ctypes.sizeof(MyPyStruct)

    # === Conecta ao mapeamento de memória e ao evento ===
    MAP_NAME = "MyPythonConection"
    EVENT_NAME = "WorkEvent"

    mm = mmap.mmap(-1, sizeof_struct, tagname=MAP_NAME)
    data_ptr = MyPyStruct.from_buffer(mm)

    # Abre o handle do evento criado pelo C++
    max_wait_sec = 10
    event_handle = None
    for _ in range(max_wait_sec * 10):  # tenta por até 10 segundos
        try:
            event_handle = win32event.OpenEvent(win32con.EVENT_MODIFY_STATE | win32con.SYNCHRONIZE,
                                                False, EVENT_NAME)
            break
        except pywintypes.error:
            time.sleep(0.1)

    if event_handle is None:
        print("Não foi possível abrir o evento compartilhado (WorkEvent) após aguardar.")
        exit(1)

    print("✅ Aguardando sinal de novo pacote...")

    while True:
        # Aguarda o evento ser sinalizado (timeout opcional: INFINITE = espera indefinida)
        wait_result = win32event.WaitForSingleObject(event_handle, win32event.INFINITE)

        if wait_result == win32con.WAIT_OBJECT_0:
            # lê snapshot
            n_dol = int(data_ptr.n_trades_dol)
            n_wdo = int(data_ptr.n_trades_wdo)

            if n_dol > 0:
                arr_d = np.ctypeslib.as_array(data_ptr.trades_dol)[: n_dol * 5].reshape(-1, 5).astype(float)
            else:
                arr_d = np.empty((0,5), dtype=float)

            if n_wdo > 0:
                arr_w = np.ctypeslib.as_array(data_ptr.trades_wdo)[: n_wdo * 5].reshape(-1, 5).astype(float)
            else:
                arr_w = np.empty((0,5), dtype=float)

            # print(f"\n📦 Novo pacote: {n_dol} trades DOL, {n_wdo} trades WDO")
            # print(f"    Últimos trades DOL: {arr_d}")
            # print(f"    Últimos trades WDO: {arr_w}")

            # depois de ler n_dol/n_wdo e montar arr_d, arr_w (shape N×5)
            sec_now = int(time.time())  # se tiver timestamp do produtor, use ele aqui!

            # passa os trades CRUS; o segmenter aplica os multiplicadores de lote
            events_done = seg.step(sec_now,
                                arr_w.tolist(),   # WDO
                                arr_d.tolist())   # DOL

            # imprime qualquer evento finalizado neste passo (normalmente 0 ou 1)
            for ev in events_done:
                print_event(ev)
        
        win32event.ResetEvent(event_handle)

def process_all():
    pass

if __name__ == "__main__":
    realtime = False

    DATA_DIR  = Path(r"E:\Mercado BMF&BOVESPA\tryd\consolidados_npz")
    DAY       = "20201013"
    PAIR      = ("wdo", "dol")   # ou 

    eventos = process_day(DATA_DIR, DAY, PAIR)

    if realtime:
        realtime_process()

    else:
        process_all()