"""
Realtime processing for event segmentation (experimental).

This module provides a class for connecting to a shared memory segment
and event signal produced by an external trading system. It reads
trades from the shared memory, aggregates them into events using
``EventSegmenter``, and prints completed events. Due to platform
differences the implementation relies on Windows-specific APIs (via
pywin32) and therefore is not tested or enabled by default.

If run on an unsupported platform, attempting to instantiate or use
``RealTimeRunner`` will raise an informative error. Users should
primarily use the offline processing capabilities unless they have a
matching Windows environment and producer application.
"""

from __future__ import annotations

import sys
from typing import Optional, List, Dict
import time

try:
    import ctypes
    import mmap
    import win32event
    import win32con
    import pywintypes
except ImportError as e:
    # Real‑time capabilities require pywin32 on Windows
    ctypes = None  # type: ignore
    mmap = None  # type: ignore
    win32event = None  # type: ignore
    win32con = None  # type: ignore
    pywintypes = None  # type: ignore

from .segmentation import EventSegmenter, SegParams


class RealTimeRunner:
    """Run the event segmenter in realtime using shared memory input.

    Parameters
    ----------
    params : SegParams
        Configuration parameters for the underlying :class:`EventSegmenter`.

    Notes
    -----
    This class is experimental and only works on Windows with the
    matching producer application that populates a named shared memory
    region and signals an event when new data is available. The
    constructor lazily imports pywin32 modules and will raise if they
    cannot be loaded.
    """
    MAP_NAME: str = "MyPythonConection"
    EVENT_NAME: str = "WorkEvent"

    def __init__(self, params: Optional[SegParams] = None) -> None:
        if ctypes is None or win32event is None:
            raise RuntimeError("Realtime processing requires pywin32 on Windows; please install pywin32 and run on Windows.")
        self.seg = EventSegmenter(params or SegParams())
        self.seg.reset()
        self._connect_shared_memory()

    def _connect_shared_memory(self) -> None:
        # Define the C struct mapping
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
        self._struct_cls = MyPyStruct
        sizeof_struct = ctypes.sizeof(MyPyStruct)
        # Open memory map
        self.mm = mmap.mmap(-1, sizeof_struct, tagname=self.MAP_NAME)
        self.data_ptr = MyPyStruct.from_buffer(self.mm)
        # Open event handle created by producer
        event_handle = None
        max_wait_sec = 10
        for _ in range(max_wait_sec * 10):
            try:
                event_handle = win32event.OpenEvent(win32con.EVENT_MODIFY_STATE | win32con.SYNCHRONIZE, False, self.EVENT_NAME)
                break
            except pywintypes.error:
                time.sleep(0.1)
        if event_handle is None:
            raise RuntimeError("Não foi possível abrir o evento compartilhado (WorkEvent)")
        self.event_handle = event_handle

    def run(self) -> None:
        print("✅ Aguardando sinal de novo pacote...")
        while True:
            wait_result = win32event.WaitForSingleObject(self.event_handle, win32event.INFINITE)
            if wait_result == win32con.WAIT_OBJECT_0:
                n_dol = int(self.data_ptr.n_trades_dol)
                n_wdo = int(self.data_ptr.n_trades_wdo)
                if n_dol > 0:
                    arr_d = np.ctypeslib.as_array(self.data_ptr.trades_dol)[: n_dol * 5].reshape(-1, 5).astype(float)
                else:
                    arr_d = np.empty((0, 5), dtype=float)
                if n_wdo > 0:
                    arr_w = np.ctypeslib.as_array(self.data_ptr.trades_wdo)[: n_wdo * 5].reshape(-1, 5).astype(float)
                else:
                    arr_w = np.empty((0, 5), dtype=float)
                sec_now = int(time.time())
                events_done = self.seg.step(sec_now, arr_w.tolist(), arr_d.tolist())
                for ev in events_done:
                    # Simplified event print: show start/end, duration and reason
                    start_ts = ev.get("start")
                    end_ts = ev.get("end")
                    dur = ev.get("dur")
                    reason = ev.get("end_reason")
                    print(f"Evento: {start_ts}->{end_ts} dur={dur}s end_reason={reason}")
            win32event.ResetEvent(self.event_handle)