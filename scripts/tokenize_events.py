#!/usr/bin/env python
from src.tokenization.tokenize_events import tokenize_all_events
from src.config import PATHS


def main():
    tokenize_all_events(
        data_dir=str(PATHS.eventos_processados),
        checkpoint=str(PATHS.models / "vq_out" / "vqvae_best_1311.pt"),
        out=str(PATHS.tokens_out / "tokens_by_day.pt"),
        batch_size=4096,
        device="cuda",
    )


if __name__ == "__main__":
    main()
