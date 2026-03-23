# Predict_Pleno

Pipeline para preparar dados de mercado, treinar VQ-VAE + Transformer e rodar o agente em tempo real.

**Resumo do fluxo**
1. Extração/ingestão: replay Tryd → `.npz`
2. Segmentação: `.npz` → eventos `.npy`
3. Normalização: manual + scaler sklearn → `*_norm_norm.npy` + `scaler_norm2.pkl`
4. VQ-VAE: treino + checkpoints
5. Tokenização: eventos → `tokens_out/tokens_by_day.pt`
6. Janelas do agente: `agent_sequences.pt`
7. Treino SFT do agente
8. Execução em tempo real

**Estrutura**
- `src/`: código principal (segmentação, normalização, VQ-VAE, agente, realtime)
- `scripts/`: entry points
- `notebooks/`: notebooks auxiliares
- `artifacts/`: modelos, checkpoints, scalers
- `docs/`: documentação do pipeline e paths

**Docs**
- `docs/pipeline.md`
- `docs/paths_and_env.md`
- `docs/realtime.md`

**Configuração de paths**
Todos os caminhos padrões estão centralizados em `src/config.py`.
Você pode sobrescrever com variáveis de ambiente (ver `docs/paths_and_env.md`).

