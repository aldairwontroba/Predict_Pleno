# Paths e Variáveis de Ambiente

Os caminhos padrão estão em `src/config.py`. Para sobrescrever, use variáveis de ambiente:

`PLENO_DATA_ROOT`  
`PLENO_EVENTOS_PROCESSADOS`  
`PLENO_CONSOLIDADOS_NPZ`  
`PLENO_TRYD_ROOT`  
`PLENO_TOKENS_OUT`  
`PLENO_MODELS_DIR`  
`PLENO_CHECKPOINTS_DIR`  
`PLENO_SCALERS_DIR`

**Defaults (se não setar env):**
- `PLENO_DATA_ROOT`: `C:\Users\Aldair\Desktop`
- `PLENO_TRYD_ROOT`: `E:\Mercado BMF&BOVESPA\tryd`
- `PLENO_EVENTOS_PROCESSADOS`: `${PLENO_DATA_ROOT}\eventos_processados`
- `PLENO_CONSOLIDADOS_NPZ`: `${PLENO_DATA_ROOT}\consolidados_npz`
- `PLENO_TOKENS_OUT`: `<repo>\tokens_out`
- `PLENO_MODELS_DIR`: `<repo>\artifacts\models`
- `PLENO_CHECKPOINTS_DIR`: `<repo>\artifacts\checkpoints`
- `PLENO_SCALERS_DIR`: `<repo>\artifacts\scalers`

**Exemplo (PowerShell)**
```powershell
$env:PLENO_DATA_ROOT="D:\dados"
$env:PLENO_TRYD_ROOT="E:\Mercado BMF&BOVESPA\tryd"
```

