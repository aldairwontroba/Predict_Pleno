# -*- coding: utf-8 -*-
from pathlib import Path
import re
import numpy as np
import pandas as pd

from src.config import PATHS

ROOT = PATHS.tryd_root   # raiz dos dados
OUT  = PATHS.consolidados_npz              # pasta de saída
YEARS = range(2025, 2026)

start_date = "20250801"

# regex para identificar arquivos (aceita dol/wdo com sufixo extra)
FILE_RE = re.compile(r"(?P<date>\d{8})_(?P<ativo>dol|wdo)", re.IGNORECASE)
# FILE_RE = re.compile(r"(?P<date>\d{8})_(?P<ativo>ind|win)", re.IGNORECASE)
FOLDER_RE = re.compile(r"^(?P<y>\d{4})-?(?P<m>\d{2})-?(?P<d>\d{2})$")

def parse_folder_date(name: str) -> str | None:
    m = FOLDER_RE.match(name)
    if not m:
        return None
    return f"{m.group('y')}{m.group('m')}{m.group('d')}"

def read_one_file(path: Path):
    try:
        df = pd.read_csv(
            path,
            sep=";",
            header=None,
            names=["ordem", "preco", "lote", "data", "cor_b", "cor_v", "tipo"],
            dtype=str,
            engine="python",
        )
        if df.empty:
            return None, None

        # converte colunas numéricas
        for col in ["preco", "lote", "cor_b", "cor_v", "tipo"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # parse datetime
        dt = pd.to_datetime(df["data"], format="%Y%m%d%H%M%S", errors="coerce")
        mask = ~dt.isna()
        if not mask.all():
            df = df.loc[mask]
            dt = dt.loc[mask]

        if df.empty:
            return None, None

        t = dt.to_numpy(dtype="datetime64[ns]")
        X = df[["preco", "lote", "cor_b", "cor_v", "tipo"]].to_numpy(dtype=np.float32)
        return t, X

    except Exception as e:
        print(f"[ERRO] {path}: {e}")
        return None, None

def salvar_por_dia():
    OUT.mkdir(parents=True, exist_ok=True)
    total = 0

    for year in YEARS:
        year_dir = ROOT / str(year) / "Extraidos DOLWDO"
        # year_dir = ROOT / str(year) / "Extraidos INDWIN"
        if not year_dir.is_dir():
            continue

        for day_dir in year_dir.iterdir():
            if not day_dir.is_dir():
                continue

            folder_date = parse_folder_date(day_dir.name)
            if not folder_date:
                continue

            # ✅ filtro para continuar de uma data específica
            if start_date and folder_date < start_date:
                continue

            for f in day_dir.iterdir():
                if not f.is_file():
                    continue
                m = FILE_RE.match(f.name.lower())
                if not m:
                    continue

                ativo = m.group("ativo").lower()   # dol ou wdo
                t, X = read_one_file(f)
                if t is None:
                    continue

                out_file = OUT / f"{folder_date}_{ativo}.npz"

                # se já existe, pula
                if out_file.exists():
                    continue

                np.savez_compressed(out_file, t=t, X=X)
                total += len(t)

    print(f"✅ Finalizado. Total de registros salvos: {total}")
    print(f"📂 Arquivos em: {OUT}")

if __name__ == "__main__":
    salvar_por_dia()
