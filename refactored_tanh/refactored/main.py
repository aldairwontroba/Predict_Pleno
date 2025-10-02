"""
Command‑line interface for processing trade data into events and generating
visualisations.

This script provides a simple CLI for running the event segmentation
algorithm on historical NPZ files or (optionally) in real‑time using
shared memory. After processing a day of data users can interactively
explore basic statistics, generate plots, and normalise event features
for downstream modelling. The script uses modules from the
``refactored`` package to handle segmentation, data loading and
plotting.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple
import sys
import numpy as np

"""
The imports below use relative notation when the module is executed as
part of the ``refactored`` package (e.g. ``python -m refactored.main``).
To support running this file directly as a script (``python
refactored/main.py``), we fallback to adjusting ``sys.path`` and
performing absolute imports. This avoids the ``ImportError: attempted
relative import with no known parent package`` when executed outside a
package context.
"""
try:
    # Package‑relative imports (preferred when running via ``-m``)
    from .segmentation import SegParams, EventSegmenter
    from .data_processing import (
        process_day,
        events_to_df,
        normalize_events_to_vectors,
        lot_multiplier,
        load_npz,
        group_by_second_preserving_order,
        to_epoch_seconds,
    )
    from .plotting import (
        plot_event_distributions,
        plot_relationships,
        plot_counts_by_hour_and_reason,
        print_top_outliers,
        plot_tick_series,
    )
except ImportError:
    # Fallback for script execution: adjust sys.path and import using absolute names
    import os
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    from refactored.segmentation import SegParams, EventSegmenter
    from refactored.data_processing import (
        process_day,
        events_to_df,
        normalize_events_to_vectors,
        lot_multiplier,
        load_npz,
        group_by_second_preserving_order,
        to_epoch_seconds,
    )
    from refactored.plotting import (
        plot_event_distributions,
        plot_relationships,
        plot_counts_by_hour_and_reason,
        print_top_outliers,
        plot_tick_series,
    )

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore


def parse_pair(value: str) -> Tuple[str, str]:
    parts = value.split(",") if "," in value else value.split()
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("pair must consist of two comma or space separated symbols, e.g. 'wdo dol'")
    return parts[0], parts[1]


def interactive_menu(df, sec_price, eventos, pair_label):
    """Launch an interactive CLI for exploring event statistics and plots."""
    while True:
        print("\nSelecione uma ação:\n"
              "1. Mostrar distribuições básicas (histogramas, ECDFs)\n"
              "2. Mostrar relacionamentos entre variáveis (hexbin, boxplots)\n"
              "3. Mostrar contagens por hora e motivo\n"
              "4. Mostrar top outliers\n"
              "5. Plotar tick‑a‑tick colorido por evento\n"
              "6. Normalizar eventos em vetores e salvar (opcional)\n"
              "0. Sair")
        try:
            choice = input("Digite o número da opção desejada: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSaindo.")
            return
        if choice == "1":
            plot_event_distributions(df)
        elif choice == "2":
            plot_relationships(df)
        elif choice == "3":
            plot_counts_by_hour_and_reason(df)
        elif choice == "4":
            print_top_outliers(df, top=10)
        elif choice == "5":
            label = pair_label[0].upper() if pair_label[0].lower() == "wdo" or pair_label[1].lower() == "wdo" else pair_label[0].upper()
            plot_tick_series(sec_price, eventos, title=f"Tick-a-tick {label} — cores por evento", symbol_label=label)
        elif choice == "6":
            # Normalise and optionally save
            X, feature_names, scaler_meta = normalize_events_to_vectors(df)
            print(f"\nVetor normalizado com shape {X.shape}. {len(feature_names)} features.\n")
            save = input("Salvar X e metadados em arquivo NPZ? (s/n): ").strip().lower()
            if save == "s":
                out_path = input("Informe o caminho do arquivo NPZ (ex.: eventos_norm.npz): ").strip()
                try:
                    np.savez(out_path, X=X, feature_names=feature_names, scaler_meta=scaler_meta)
                    print(f"Dados salvos em {out_path}.")
                except Exception as e:
                    print(f"Erro ao salvar: {e}")
        elif choice == "0":
            return
        else:
            print("Opção inválida. Tente novamente.")


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Processa trades em eventos e gera estatísticas.")
    parser.add_argument("--mode", choices=["offline", "realtime"], default="offline", help="Modo de execução: offline ou realtime")
    parser.add_argument("--data-dir", type=Path, default=Path("."), help="Diretório contendo arquivos NPZ")
    parser.add_argument("--day", type=str, default="", help="Data no formato YYYYMMDD (para modo offline)")
    parser.add_argument("--pair", type=parse_pair, default=("wdo", "dol"), help="Símbolos A e B, separados por espaço ou vírgula")
    parser.add_argument("--interactive", action="store_true", help="Entrar em modo interativo após processamento")
    args = parser.parse_args(argv)
    if args.mode == "realtime":
        # Defer import of realtime components to avoid dependency issues
        print("Modo realtime ainda não suportado nesta versão.")
        print("Por favor, utilize o modo offline com --mode offline.")
        return
    # Offline processing
    if not args.day:
        parser.error("--day é obrigatório no modo offline")
    pair = args.pair
    try:
        eventos = process_day(args.data_dir, args.day, pair)
    except Exception as e:
        print(f"Erro ao processar dados: {e}")
        sys.exit(1)
    print(f"Dia {args.day} {pair[0].upper()} + {pair[1].upper()} → {len(eventos)} eventos")
    if len(eventos) == 0:
        return
    # Build DataFrame
    if pd is None:
        print("Pandas não está disponível. Instale pandas para análises adicionais.")
        return
    df = events_to_df(eventos)
    # Determine which second->price map to use for tick plotting (WDO by default)
    # Recreate sec_price map using group_by_second if needed
    try:
        t_a, TT_a = load_npz(args.data_dir / f"{args.day}_{pair[0]}.npz")  # type: ignore
        t_b, TT_b = load_npz(args.data_dir / f"{args.day}_{pair[1]}.npz")  # type: ignore
        sec_a = group_by_second_preserving_order(to_epoch_seconds(t_a), TT_a, pair[0])  # type: ignore
        sec_b = group_by_second_preserving_order(to_epoch_seconds(t_b), TT_b, pair[1])  # type: ignore
        if pair[0].lower() == "wdo":
            sec_price = sec_a
        elif pair[1].lower() == "wdo":
            sec_price = sec_b
        else:
            sec_price = sec_a
    except Exception:
        # Fallback: empty map
        sec_price = {}
    if args.interactive:
        interactive_menu(df, sec_price, eventos, pair)


if __name__ == "__main__":
    main(["--mode", "offline", "--data-dir", r"E:\Mercado BMF&BOVESPA\tryd\consolidados_npz", "--day", "20201013", "--pair", "wdo,dol", "--interactive"])