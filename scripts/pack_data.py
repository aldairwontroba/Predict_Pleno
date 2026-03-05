"""
Compacta os arquivos *_norm_norm.npy para upload no RunPod.

Roda no Windows antes de fazer o upload:
  python scripts/pack_data.py

Saída: dados_ct.tar.gz (~380 MB)

No RunPod, descompacta com:
  tar -xzf dados_ct.tar.gz -C /workspace/data/
"""
import os
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.continuous_transformer.config import _DEFAULT_DATA_DIR

# ── Configuração ────────────────────────────────────────────
DATA_DIR  = Path(os.getenv("PLENO_EVENTOS_PROCESSADOS", str(_DEFAULT_DATA_DIR)))
PATTERN   = "*_norm_norm.npy"
OUT_FILE  = Path(__file__).resolve().parents[1] / "dados_ct.tar.gz"
# ────────────────────────────────────────────────────────────

files = sorted(DATA_DIR.glob(PATTERN))
if not files:
    print(f"Nenhum arquivo encontrado em: {DATA_DIR} com padrão {PATTERN}")
    sys.exit(1)

total_bytes = sum(f.stat().st_size for f in files)
print(f"Compactando {len(files)} arquivos ({total_bytes/1024**3:.2f} GB)...")
print(f"Saída: {OUT_FILE}")

with tarfile.open(OUT_FILE, "w:gz", compresslevel=6) as tar:
    for i, f in enumerate(files, 1):
        tar.add(f, arcname=f.name)   # sem estrutura de diretório
        if i % 100 == 0 or i == len(files):
            pct = 100 * i / len(files)
            print(f"  {i}/{len(files)} ({pct:.0f}%)  {f.name}")

size_gz = OUT_FILE.stat().st_size
print(f"\nPronto! Tamanho compactado: {size_gz/1024**3:.2f} GB  ({size_gz/1024**2:.0f} MB)")
print(f"\nUpload para RunPod:")
print(f"  scp {OUT_FILE} root@<IP_DO_POD>:/workspace/")
print(f"  # No pod:")
print(f"  tar -xzf /workspace/dados_ct.tar.gz -C /workspace/data/")
