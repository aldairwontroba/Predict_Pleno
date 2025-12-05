# vq_tokenizer_rt.py
import numpy as np
import torch
from normalization_rt import RealTimeNormalizer
import vqvae_train_new as vqmod  # ajuste o nome caso precise

class RealTimeVQTokenizer:
    def __init__(self, vq_ckpt, scaler_path, device="cuda"):
        self.device = torch.device(device)

        # normalizador completo
        self.norm = RealTimeNormalizer(scaler_path)

        # modelo VQ-VAE
        self.model = vqmod.VQVAE().to(device)
        state = torch.load(vq_ckpt, map_location="cpu")
        self.model.load_state_dict(state["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def event_to_token(self, vector_dict):
        """
        Recebe ev["vector"] → token VQ-VAE (int).
        """
        vec_raw = self.norm.extract_vec(vector_dict)
        vec_norm = self.norm.normalize(vec_raw)

        x = torch.from_numpy(vec_norm).float().to(self.device).unsqueeze(0)

        z_e = self.model.encoder(x)
        _, codes, _ = self.model.vq(z_e)

        return int(codes[0].view(-1).item())
