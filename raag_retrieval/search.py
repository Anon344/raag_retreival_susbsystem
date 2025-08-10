
from typing import List, Dict
import json
from .embedder import STEmbedder
try:
    import faiss  # type: ignore
except Exception:
    faiss = None

def load_mapping(mapping_path: str) -> List[Dict]:
    return [json.loads(x) for x in open(mapping_path, 'r', encoding='utf-8') if x.strip()]

def search(query: str, index_path: str, mapping_path: str, encoder_name: str, topk: int = 5) -> List[Dict]:
    assert faiss is not None, "faiss-cpu is required for search."
    idx = faiss.read_index(index_path)
    metas = load_mapping(mapping_path)
    emb = STEmbedder(encoder_name)
    q = emb.encode([query])
    # Clamp topk to index size
    topk = min(topk, idx.ntotal if hasattr(idx, "ntotal") else len(metas))
    sims, idxs = idx.search(q, topk)
    idxs = idxs[0].tolist(); sims = sims[0].tolist()
    out = []
    for rank, (i, s) in enumerate(zip(idxs, sims)):
        # Filter FAISS "empty" slots: invalid index or -inf-like scores
        if i < 0 or i >= len(metas) or s < -1e30:
            continue
        m = metas[i]
        out.append({**m, "rank": rank, "score": float(s), "chunk_global_id": i})
    return out
