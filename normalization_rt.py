# normalization_rt.py
import numpy as np
import joblib

from CreateVector.normalize_manual import normalize_matrix, FEATURE_ORDER

# As 45 colunas usadas no treinamento:
DL = np.r_[0:4, 6:11, 12:23, 28, 31, 34, 37:59]

class RealTimeNormalizer:
    def __init__(self, scaler_path: str):
        self.scaler = joblib.load(scaler_path)

        # A ordem correta das features vindo do dict ev["vector"]
        self.feature_order = FEATURE_ORDER

        # Índices usados no treino
        self.idx_used = DL

    def extract_vec(self, vector_dict: dict) -> np.ndarray:
        """
        Converte o dict de um evento em um vetor ordenado (1D).
        Igual ao pipeline offline.
        """
        vec = np.zeros(len(self.feature_order), dtype=np.float32)

        for i, k in enumerate(self.feature_order):
            v = vector_dict.get(k, 0.0)
            try:
                vec[i] = float(v)
            except:
                vec[i] = 0.0

        return vec

    def normalize(self, vec_raw: np.ndarray) -> np.ndarray:
        """
        Aplica:
        - seleção de colunas DL
        - normalização manual
        - normalização sklearn (global)
        """
        # (59,) → (45,)  seleciona features usadas no treino
        vec_sel = vec_raw[self.idx_used]

        # manual normalization (igual treino)
        x = normalize_matrix(vec_sel[None, :])   # (1,45)

        # sklearn normalization
        x2 = self.scaler.transform(x)            # (1,45)

        return x2[0]   # volta para (45,)
