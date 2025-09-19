# pip install pyautogui pygetwindow
import time
import pyautogui as gui
import pygetwindow as gw
import re
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
import tempfile
import sys
from zipfile import ZipFile, ZIP_DEFLATED

# ================== CONFIG GERAL ==================
YEAR = 2018
MARKET_PAIR = "INDWIN"   # "DOLWDO" ou "WININD"
SAVE_AND_ZIP_DAT = False  # True = copiar e compactar .dat | False = não salvar .dat

# Mapeia nomes de pastas e símbolos aceitos por par de mercado
MARKET_CFG = {
    "DOLWDO": {
        "import_folder": r"C:\Tryd7\workspace\replay\import\WDO&DOL",
        "dest_root":     fr"E:\Mercado BMF&BOVESPA\tryd\{YEAR}\Extraidos DOLWDO",
        "symbols": ("dol", "wdo")
    },
    "INDWIN": {
        "import_folder": r"C:\Tryd7\workspace\replay\import\WIN&IND",
        "dest_root":     fr"E:\Mercado BMF&BOVESPA\tryd\{YEAR}\Extraidos INDWIN",
        "symbols": ("win", "ind")
    }
}

if MARKET_PAIR not in MARKET_CFG:
    raise ValueError("MARKET_PAIR deve ser 'DOLWDO' ou 'INDWIN'.")

IMPORT_DIR = Path(MARKET_CFG[MARKET_PAIR]["import_folder"])
DEST_ROOT  = Path(MARKET_CFG[MARKET_PAIR]["dest_root"])
SYMS       = MARKET_CFG[MARKET_PAIR]["symbols"]

NEGOCIOS_DIR = IMPORT_DIR / "negocios"

SEVEN_ZIP_EXE = r"C:\Program Files\7-Zip\7z.exe"  # ou "7z" se estiver no PATH

# Controle de sobrescrita
OVERWRITE_COPY = False     # copiar por cima?
OVERWRITE_ZIP  = True      # recriar .zip se já existir?

# ================== CHECKS ==================
if not Path(SEVEN_ZIP_EXE).exists():
    print(f"❌ 7z.exe não encontrado em: {SEVEN_ZIP_EXE}")
    sys.exit(1)

DEST_ROOT.mkdir(parents=True, exist_ok=True)

# Regex (compila com símbolos do par selecionado)
# Ex.: 20240110_dolm18.gz, 20180430_wdom18.gz, 20240110_ind12.gz, etc.
pat_gz   = re.compile(rf"^(?P<ymd>\d{{8}})_(?P<kind>({'|'.join(SYMS)}).*)$", re.IGNORECASE)
pat_repl = re.compile(r"^tryd_replay_(?P<ymd>\d{8})\.0\.0\.dat$", re.IGNORECASE)

# ------------- CONFIGS DE AUTOMAÇÃO -------------
WINDOW_TITLE_CONTAINS = "Preferências"      # Ajuste conforme a janela do Tryd
PLAY_STOP_POS = (3327, 141)                 # coords do botão Play/Parar (x,y)
LIST_CLICK_POS = (2787, 121)                # ponto dentro da lista (x,y)
N_ITENS_PARA_SUBIR = 1                      # geralmente 1 (arquivo logo acima)
ESPERA_REPLAY_INICIAR = 20                  # segundos até mandar Enter no pop-up
FATOR_SCROLL = 0                            # 0 = não rola
REPETICOES = 150                            # quantos arquivos processar

gui.FAILSAFE = True     # mover mouse p/ canto sup-esq aborta
gui.PAUSE = 0.15        # intervalo entre ações

# ------------- FUNÇÕES -------------
def ymd_to_dash(ymd: str) -> str:
    return datetime.strptime(ymd, "%Y%m%d").strftime("%Y-%m-%d")

def copy2(src: Path, dst: Path) -> bool:
    """Copia src→dst. Retorna True se copiou, False se pulou."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not OVERWRITE_COPY:
        print(f"↩️  Pulando (já existe): {dst}")
        return False
    shutil.copy2(src, dst)
    print(f"✅ Copiado: {src} → {dst}")
    return True

def zip_then_cleanup(dat_path: Path):
    """Compacta dat_path p/ .zip na mesma pasta, remove o .dat e .gz da pasta."""
    if not dat_path.exists():
        print(f"⚠️  .dat não encontrado para compactar: {dat_path}")
        return

    zip_path = dat_path.with_suffix(".zip")
    if zip_path.exists():
        if OVERWRITE_ZIP:
            try:
                zip_path.unlink()
                print(f"🧹 Removendo zip existente para recriar: {zip_path}")
            except Exception as e:
                print(f"❌ Não foi possível remover {zip_path}: {e}")
                return
        else:
            print(f"↩️  Zip já existe e OVERWRITE_ZIP=False, mantendo: {zip_path}")
            return

    try:
        with ZipFile(zip_path, "w", compression=ZIP_DEFLATED, compresslevel=6) as zf:
            zf.write(dat_path, arcname=dat_path.name)
        print(f"🗜️  Compactado: {dat_path.name} → {zip_path.name}")

        dat_path.unlink()
        print(f"🧹 Removido .dat original: {dat_path}")

        for gz in dat_path.parent.glob("*.gz"):
            try:
                gz.unlink()
                print(f"🧹 Removido .gz: {gz.name}")
            except Exception as e:
                print(f"⚠️  Falha ao remover {gz.name}: {e}")

    except Exception as e:
        print(f"❌ Erro ao compactar {dat_path}: {e}")

def extract_gz(gz: Path, out_dir: Path):
    print(f"📦 Extraindo: {gz.name}")
    cmd = [SEVEN_ZIP_EXE, "e", "-y", f"-o{out_dir}", str(gz)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print("❌ Erro 7z:", proc.stderr or proc.stdout)
        raise RuntimeError(f"Falha 7z em {gz}")

def cleanup_dest_folder(folder: Path):
    """Remove .gz e .dat da pasta destino."""
    for ext in ("*.gz", "*.dat"):
        for f in folder.glob(ext):
            try:
                f.unlink()
                print(f"🧹 Removido {f.name}")
            except Exception as e:
                print(f"⚠️  Falha ao remover {f.name}: {e}")

def process_gz():
    gz_files = sorted(NEGOCIOS_DIR.glob("*.gz"))
    print(f"🔎 Encontrados {len(gz_files)} .gz em {NEGOCIOS_DIR}\n")

    with tempfile.TemporaryDirectory(prefix="negocios_extract_") as tmpdir:
        TMP = Path(tmpdir)
        print(f"📂 Pasta temporária: {TMP}\n")

        for gz in gz_files:
            m = pat_gz.match(gz.stem)  # sem .gz
            if not m:
                print(f"⚠️  Ignorando (fora do padrão {SYMS[0]}/{SYMS[1]}): {gz.name}")
                continue

            ymd = m.group("ymd")
            yyyy_mm_dd = ymd_to_dash(ymd)
            dest_folder = DEST_ROOT / yyyy_mm_dd

            # 1) extrair
            try:
                extract_gz(gz, TMP)
            except Exception as e:
                print(e)
                continue

            # 2) pegar arquivos extraídos que comecem com 'YYYYMMDD_'
            for f in TMP.iterdir():
                if not (f.is_file() and f.name.startswith(ymd + "_")):
                    continue

                # normaliza nome com base no símbolo (dol/wdo ou win/ind)
                suffix = f.name.split("_", 1)[1].lower()
                norm_sym = None
                for s in SYMS:
                    if suffix.startswith(s):
                        norm_sym = s
                        break
                if norm_sym is None:
                    # ex.: di1f21 ou outros — ignorar
                    continue

                norm = f"{ymd}_{norm_sym}"
                copy2(f, dest_folder / norm)

def process_replays():
    replay_files = sorted(IMPORT_DIR.glob("tryd_replay_*.0.0.dat"))
    if not replay_files:
        print(f"⚠️  Nenhum replay encontrado em {IMPORT_DIR}")
        return

    print(f"\n🎬 Replays encontrados: {len(replay_files)}")
    for rp in replay_files:
        m = pat_repl.match(rp.name)
        if not m:
            print(f"⚠️  Ignorando replay fora do padrão: {rp.name}")
            continue
        ymd = m.group("ymd")
        yyyy_mm_dd = ymd_to_dash(ymd)
        dest_folder = DEST_ROOT / yyyy_mm_dd
        dst_dat = dest_folder / rp.name

        if SAVE_AND_ZIP_DAT:
            did_copy = copy2(rp, dst_dat)
            if did_copy:
                zip_then_cleanup(dst_dat)
        else:
            # não salva, mas limpa qualquer .dat/.gz do destino
            cleanup_dest_folder(dest_folder)

def minha_funcao_de_processamento():
    """
    Chame aqui sua função que processa enquanto o replay está rodando.
    Pode ser bloqueante; quando retornar, vamos parar o replay.
    """
    process_gz()
    process_replays()
    print("\n🏁 Finalizado.")

# ------------- HELPER: focar janela do Tryd -------------
def focar_janela_tryd():
    wins = [w for w in gw.getAllWindows() if WINDOW_TITLE_CONTAINS.lower() in w.title.lower()]
    if not wins:
        raise RuntimeError(f"Janela com '{WINDOW_TITLE_CONTAINS}' não encontrada. Abra a tela de Replay/Preferências.")
    win = wins[0]
    if win.isMinimized:
        win.restore()
    win.activate()
    time.sleep(0.5)

# ------------- AÇÕES ATÔMICAS -------------
def clicar_play_ou_parar():
    gui.click(PLAY_STOP_POS)

def enviar_enter_para_popup():
    for _ in range(2):
        gui.press("enter")
        time.sleep(0.2)

def focar_lista():
    gui.click(LIST_CLICK_POS)

def selecionar_item_acima(n=1):
    focar_lista()
    if FATOR_SCROLL:
        gui.scroll(FATOR_SCROLL)  # positivo = rola para cima
    for _ in range(n):
        gui.press("up")

# ------------- LOOP PRINCIPAL -------------
def main():
    print(f"🧭 Modo: {MARKET_PAIR} | Símbolos: {SYMS}")
    print(f"📁 IMPORT_DIR: {IMPORT_DIR}")
    print(f"📁 NEGOCIOS_DIR: {NEGOCIOS_DIR}")
    print(f"📁 DEST_ROOT: {DEST_ROOT}")
    print(f"💾 SAVE_AND_ZIP_DAT: {SAVE_AND_ZIP_DAT}\n")

    focar_janela_tryd()

    for i in range(REPETICOES):
        # 1) PLAY
        clicar_play_ou_parar()
        print("PLAY")

        # 2) Espera iniciar e fecha pop-up com Enter
        time.sleep(ESPERA_REPLAY_INICIAR)
        enviar_enter_para_popup()

        # 3) Executa sua rotina
        minha_funcao_de_processamento()

        # 4) PARAR (mesmo botão)
        clicar_play_ou_parar()
        time.sleep(1)
        print("PARAR")

        # 5) Selecionar próximo arquivo (logo acima)
        selecionar_item_acima(N_ITENS_PARA_SUBIR)
        time.sleep(0.5)
        print(f"Próximo arquivo ({i+1}/{REPETICOES})")

    print("✔️ Automação concluída.")

if __name__ == "__main__":
    main()
