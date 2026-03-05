from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _env_path(key: str, default: Path) -> Path:
    val = os.getenv(key)
    return Path(val) if val else default


@dataclass(frozen=True)
class Paths:
    repo: Path = REPO_ROOT

    data_root: Path = _env_path("PLENO_DATA_ROOT", Path(r"C:\Users\Aldair\Desktop"))
    eventos_processados: Path = _env_path(
        "PLENO_EVENTOS_PROCESSADOS", data_root / "eventos_processados"
    )
    consolidados_npz: Path = _env_path(
        "PLENO_CONSOLIDADOS_NPZ", data_root / "consolidados_npz"
    )
    tryd_root: Path = _env_path(
        "PLENO_TRYD_ROOT", Path(r"E:\Mercado BMF&BOVESPA\tryd")
    )

    tokens_out: Path = _env_path("PLENO_TOKENS_OUT", repo / "tokens_out")

    artifacts: Path = repo / "artifacts"
    models: Path = _env_path("PLENO_MODELS_DIR", artifacts / "models")
    checkpoints: Path = _env_path("PLENO_CHECKPOINTS_DIR", artifacts / "checkpoints")
    scalers: Path = _env_path("PLENO_SCALERS_DIR", artifacts / "scalers")


PATHS = Paths()

