# Experimento CET v1 — Continuous Event Transformer

## Objetivo

Substituir o pipeline VQ-VAE + Transformer discreto por um Transformer contínuo (CET)
que recebe diretamente os vetores de eventos normalizados (59 features) sem tokenização,
eliminando a perda de informação do processo de quantização.

O objetivo final é um agente que tome decisões de mercado (buy/sell/hold) em tempo real.

---

## O que foi feito

### Arquitetura

- **ContinuousEventTransformer**: GPT-style causal Transformer com projeção linear de entrada
  - Input: `[B, T, 59]` float32 → Linear(59, d_model) + LayerNorm
  - Positional embedding aprendido
  - N blocos de atenção causal (GPT-like)
  - Duas cabeças de saída: `next_event` (pretrain) e `action` (SFT)

### Fase 1 — Pré-treino (next-event prediction)

- **Tarefa**: dado os últimos T eventos, prever o próximo evento (Huber loss)
- **Dados**: 1810 dias (Abr/2018–Out/2025), arquivos `*_norm_norm.npy`, 59 features
- **Config (RunPod RTX 5090)**: d=256, n_layers=6, n_heads=8, seq_len=512, stride=1, 10 épocas
- **Janelas**: 3.612.370 treino + 399.917 validação

**Resultados do pré-treino:**

| Época | train_loss | val_loss |
|-------|-----------|----------|
| 1     | 0.28156   | 0.32883  ← melhor |
| 2     | 0.24649   | 0.33690  |
| 3     | 0.23652   | 0.34082  |
| 4     | 0.23215   | 0.34247  |

Parado na época 4 por overfitting claro: val_loss subindo enquanto train_loss cai.
O `pretrain_best.pt` corresponde à época 1.

### Fase 2 — SFT (fine-tuning supervisionado)

- **Tarefa**: classificar ação (buy=0 / sell=1 / hold=2) com base nos últimos n_past eventos
- **Labels**: calculados a partir do movimento futuro de preço (5 pontos de threshold)
- **Distribuição**: buy≈26%, sell≈27%, hold≈47% → class weights aplicados
- **Config**: batch=128, accum=2, lr=1e-4, 10 épocas, inicializado do pretrain_best.pt (época 1)

**Resultados do SFT:**

| Época | train_acc | val_acc  | val_loss |
|-------|-----------|----------|----------|
| 1     | 0.458     | 0.449    | 1.046    |
| 2     | 0.498     | 0.481    | **1.010** ← sft_best.pt |
| 3     | 0.550     | 0.475    | 1.042    |
| 4     | 0.599     | 0.486    | 1.059    |
| 5     | 0.645     | 0.492    | 1.096    |
| 6     | 0.683     | 0.497    | 1.129    |
| 7     | 0.716     | 0.505    | 1.196    |
| 8     | 0.746     | **0.515** | 1.235   |
| 9     | 0.770     | 0.510    | 1.301    |
| 10    | 0.793     | 0.513    | 1.394    |

Baseline "sempre hold" = 47.3%
Melhor val_acc = 51.5% (época 8) → +4.2% acima do baseline

---

## Conclusão

### O que funcionou
- Pipeline end-to-end implementado e funcionando (pretrain → SFT → checkpoint)
- Resume automático de checkpoints (resiliente a quedas do pod)
- Dataset construído eficientemente (266K janelas, ~1.1GB em memória)
- A RTX 5090 rodou o SFT completo em ~21 minutos (2.1 min/época)

### O que não funcionou bem

**1. Overfitting severo no pretrain**
O val_loss do pretrain subiu desde a época 1 (0.328 → 0.342). Isso indica que o modelo
memorizou padrões do conjunto de treino que não generalizam. Com dropout=0.1 e 3.6M
janelas com shuffle, o modelo tem capacidade suficiente para memorizar.

**2. Overfitting severo no SFT**
train_acc=79% vs val_acc=51% na época 10. Lacuna enorme de generalização.
O modelo aprendeu a acertar o treino mas não transfere para dados não vistos.

**3. val_acc medíocre**
51.5% vs baseline 47.3% é uma melhoria pequena. Para um agente de trading útil,
seria necessário no mínimo 54-58% de acurácia com boa calibração de confiança.

**4. Checkpoint salvo por val_loss, não val_acc**
O `sft_best.pt` foi salvo na época 2 (val_acc=0.481) por ter o menor val_loss,
mas o melhor val_acc foi na época 8 (0.515). Para trading, val_acc é mais relevante.

### Causa raiz provável

O pretrain fraco é a causa principal. Se o modelo não aprendeu representações
suficientemente ricas e generalizáveis dos eventos de mercado, o SFT não tem
uma boa base para aprender a classificar ações.

O pretrain overfitou porque:
- dropout muito baixo (0.1) para o volume de dados
- Sem weight decay explícito além do padrão do AdamW
- Os dados de validação são cronologicamente separados (regimes diferentes)

---

## O que fazer diferente na v2

### Pretrain
1. **Dropout maior**: 0.3–0.4 (atual: 0.1)
2. **Weight decay explícito**: AdamW com weight_decay=0.1
3. **Learning rate schedule**: cosine decay com warmup (atual: lr fixo)
4. **Early stopping real**: parar quando val_loss não melhora por 2 épocas
5. **torch.compile + bf16**: ~2x speedup, permite testar mais variações
6. **Modelo maior**: d=512, n_layers=8 (~23M params) com regularização forte

### SFT
1. **Salvar por val_acc** (não val_loss)
2. **Dropout maior no fine-tuning**: 0.4–0.5 (congelar backbone parcialmente)
3. **Learning rate menor**: 1e-5 com warmup
4. **Label smoothing**: reduz overconfiança
5. **Congelar camadas inferiores** e treinar só as superiores + action_head

### Labels (repensar)
Os labels atuais usam threshold de 5 pontos em 64 eventos futuros (~5 min).
Considerar:
- Thresholds adaptativos por volatilidade do dia
- Janela futura maior (128–256 eventos)
- Labels baseados em retorno ajustado ao risco (sharpe do período)
- Remover janelas ambíguas (hold quando há movimento mas fraco)

### Arquitetura alternativa
- **Contrastive learning** no pretrain: forçar representações similares para
  situações de mercado parecidas (em vez de next-event prediction)
- **Cabeça de valor** além da cabeça de ação (estilo actor-critic)
- **Ensemble** de modelos treinados com seeds diferentes

---

## Custo do experimento
- RTX 4090 Spot ($0.20/hr): ~3h → ~$0.60 (testes iniciais, quedas frequentes)
- RTX 5090 On-Demand ($0.89/hr): ~6h → ~$5.34 (pretrain 4 épocas + SFT 10 épocas)
- **Total**: ~$6 de $20 disponíveis → $14 restantes para v2
