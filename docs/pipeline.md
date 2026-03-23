# Pipeline

**1) Ingestão e extração**
- `scripts/automatizador_tryd.py`: automação do replay Tryd, extração de `.gz` e organização.
- `src/data_ingest/save_files_in_np.py`: consolida CSVs em `.npz` por dia/ativo.
- `scripts/replay_reverse_engineering/*`: utilitários de engenharia reversa do replay.

**2) Segmentação e vetorização**
- `scripts/process_events.py`: processamento em lote e modo realtime.
- `src/segmentation/segmentation.py`: regras e features.
- `src/segmentation/data_processing.py`: leitura `.npz`, agregação e geração de eventos.
- `src/segmentation/plotting.py`: estatísticas e plots.

**3) Normalização**
- `src/normalization/normalize_manual.py`: normalização manual por feature.
- `scripts/fit_scaler.py`: fit do scaler global e geração de `scaler_norm2.pkl`.
- `src/normalization/normalization_rt.py`: normalização em tempo real.

**4) VQ-VAE**
- `src/vqvae/train.py`: treino principal.
- `src/vqvae/train_legacy.py`: variante anterior.
- `scripts/eval_vqvae.py`: avaliação de reconstrução.

**5) Tokenização**
- `scripts/tokenize_events.py`: gera `tokens_out/tokens_by_day.pt`.

**6) Dataset do agente**
- `src/agent/build_sequences.py`: monta janelas com layout `[CTX][PAST][FUT][ANALYSIS][ACTION]`.

**7) Treino do agente**
- `src/agent/model.py`: Transformer base.
- `scripts/train_sft.py`: SFT usando `agent_sequences.pt`.
- `scripts/eval_sft.py`: validação + baselines.

**8) Realtime**
- `scripts/run_realtime_agent.py`: inferência real time.
- `src/realtime/vq_tokenizer_rt.py`: tokenização por evento em tempo real.
- `src/realtime/realtime_segmenter.py`: reader de memória compartilhada (pywin32).

