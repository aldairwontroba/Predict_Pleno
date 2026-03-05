#!/bin/bash
# =============================================================
# Setup do ambiente RunPod para treino do ContinuousEventTransformer
#
# Uso (no terminal do RunPod após clonar o repo):
#   chmod +x scripts/setup_runpod.sh
#   bash scripts/setup_runpod.sh
# =============================================================

set -e   # para imediatamente se qualquer comando falhar

echo "======================================================"
echo " Setup RunPod — Predict_Pleno"
echo "======================================================"

# ── 1. Diretórios ─────────────────────────────────────────
WORKSPACE=/workspace
DATA_DIR=$WORKSPACE/data
ARTIFACTS_DIR=$WORKSPACE/artifacts
CHECKPOINTS_DIR=$WORKSPACE/checkpoints

mkdir -p "$DATA_DIR"
mkdir -p "$ARTIFACTS_DIR"
mkdir -p "$CHECKPOINTS_DIR"

echo "[1/5] Diretórios criados: $DATA_DIR  $ARTIFACTS_DIR  $CHECKPOINTS_DIR"

# ── 2. Instala dependências Python ───────────────────────
echo "[2/5] Instalando dependências..."
pip install --quiet --upgrade pip
pip install --quiet numpy torch tqdm pathlib

echo "  torch: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA:  $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"sem GPU\")')"

# ── 3. Variável de ambiente ───────────────────────────────
echo "[3/5] Configurando PLENO_EVENTOS_PROCESSADOS=$DATA_DIR"
export PLENO_EVENTOS_PROCESSADOS="$DATA_DIR"

# Persiste em .bashrc para sessões futuras
if ! grep -q "PLENO_EVENTOS_PROCESSADOS" ~/.bashrc 2>/dev/null; then
    echo "export PLENO_EVENTOS_PROCESSADOS=$DATA_DIR" >> ~/.bashrc
fi

# ── 4. Symlinks de checkpoints e artifacts ───────────────
# Para que o código encontre os caminhos padrão (artifacts/, checkpoints/)
REPO_DIR=$(pwd)
echo "[4/5] Repo dir: $REPO_DIR"

# Cria symlinks se não existirem
if [ ! -e "$REPO_DIR/artifacts" ]; then
    ln -sf "$ARTIFACTS_DIR" "$REPO_DIR/artifacts"
    echo "  -> artifacts/ → $ARTIFACTS_DIR"
fi
if [ ! -e "$REPO_DIR/checkpoints" ]; then
    ln -sf "$CHECKPOINTS_DIR" "$REPO_DIR/checkpoints"
    echo "  -> checkpoints/ → $CHECKPOINTS_DIR"
fi

# ── 5. PYTHONPATH ─────────────────────────────────────────
echo "[5/5] Configurando PYTHONPATH"
export PYTHONPATH="$REPO_DIR:$PYTHONPATH"
if ! grep -q "PYTHONPATH.*$REPO_DIR" ~/.bashrc 2>/dev/null; then
    echo "export PYTHONPATH=$REPO_DIR:\$PYTHONPATH" >> ~/.bashrc
fi

echo ""
echo "======================================================"
echo " Setup concluído!"
echo ""
echo " Próximos passos:"
echo "   1. Fazer upload dos dados:"
echo "      scp dados_ct.tar.gz root@<ip>:/workspace/"
echo "      tar -xzf /workspace/dados_ct.tar.gz -C /workspace/data/"
echo ""
echo "   2. Fazer upload do ct_sft_sequences.pt:"
echo "      scp artifacts/ct_sft_sequences.pt root@<ip>:/workspace/artifacts/"
echo ""
echo "   3. Rodar pré-treino:"
echo "      cd $REPO_DIR"
echo "      python scripts/run_pretrain_ct.py 2>&1 | tee /workspace/pretrain.log"
echo ""
echo "   4. Rodar SFT:"
echo "      python scripts/run_sft_ct.py 2>&1 | tee /workspace/sft.log"
echo ""
echo "   5. Baixar checkpoints:"
echo "      scp -r root@<ip>:/workspace/checkpoints/ct_pretrain ./checkpoints/"
echo "      scp -r root@<ip>:/workspace/checkpoints/ct_sft ./checkpoints/"
echo "======================================================"
