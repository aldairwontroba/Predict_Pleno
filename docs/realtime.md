# Realtime

**Dependências**
- Windows
- `pywin32`
- Produtor C++ que preenche o shared memory `MyPythonConection` e sinaliza `WorkEvent`

**Fluxo**
1. `scripts/process_events.py` contém `realtime_process` (callback por evento).
2. `scripts/run_realtime_agent.py` pluga o callback para tokenizar + inferir.

**Execução**
```powershell
python scripts/run_realtime_agent.py
```

**Notas**
- Ajuste os paths no `src/config.py` ou via variáveis de ambiente.
- Ajuste o par de ativos no `scripts/run_realtime_agent.py`.

