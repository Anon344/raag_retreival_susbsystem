
from typing import List
import torch
import numpy as np
from sentence_transformers import SentenceTransformer

class STEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = None):
        self.model = SentenceTransformer(model_name, device=device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dim = self.model.get_sentence_embedding_dimension()

    @torch.no_grad()
    def encode(self, texts: List[str], batch_size: int = 64, normalize: bool = True) -> np.ndarray:
        emb = self.model.encode(texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=normalize)
        return np.asarray(emb, dtype=np.float32)
