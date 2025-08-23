# raag/retriever.py
import os, json, gzip, pickle, math
from typing import List, Dict, Tuple, Optional
import numpy as np
import faiss
from rank_bm25 import BM25Okapi

from raag.model import encode_texts, EncConfig

def _load_mapping(mapping_path: str) -> Dict[int, str]:
    texts = {}
    with open(mapping_path, "r", encoding="utf-8") as f:
        for ln in f:
            rec = json.loads(ln)
            texts[int(rec["chunk_global_id"])] = rec["text"]
    return texts

def _load_bm25(bm25_path: str):
    with gzip.open(bm25_path, "rb") as f:
        data = pickle.load(f)
    return data["bm25"], data["tokenized"]

def mmr(query_vec: np.ndarray, doc_vecs: np.ndarray, k: int, lambda_mult: float = 0.5) -> List[int]:
    """
    Maximal Marginal Relevance selection over dense vectors.
    Returns indices into doc_vecs.
    """
    if doc_vecs.shape[0] <= k:
        return list(range(doc_vecs.shape[0]))
    # similarities to query
    sim_q = (doc_vecs @ query_vec.reshape(-1,1)).ravel()  # IP
    selected = []
    candidate_idxs = list(range(doc_vecs.shape[0]))
    while len(selected) < k and candidate_idxs:
        if not selected:
            i = int(np.argmax(sim_q[candidate_idxs]))
            selected.append(candidate_idxs.pop(i))
            continue
        sel_mat = doc_vecs[selected]  # [s, d]
        cand_mat = doc_vecs[candidate_idxs]  # [c, d]
        # redundancy = max doc-doc sim per candidate
        red = np.max(cand_mat @ sel_mat.T, axis=1)
        util = lambda_mult * sim_q[candidate_idxs] - (1 - lambda_mult) * red
        i = int(np.argmax(util))
        selected.append(candidate_idxs.pop(i))
    return selected

class HybridRetriever:
    def __init__(self, index_path: str, mapping_path: str, encoder_name: str,
                 bm25_path: Optional[str] = None, nprobe: Optional[int] = None):
        self.index = faiss.read_index(index_path)
        if nprobe is not None and hasattr(self.index, "nprobe"):
            self.index.nprobe = nprobe
        self.texts = _load_mapping(mapping_path)
        self.enc_cfg = EncConfig(encoder_name=encoder_name)
        self.bm25 = None
        self.tokenized = None
        if bm25_path and os.path.exists(bm25_path):
            self.bm25, self.tokenized = _load_bm25(bm25_path)

    def dense_search(self, query: str, topn: int) -> List[Tuple[int, float]]:
        q = encode_texts([query], batch_size=1, cfg=self.enc_cfg)[0:1]  # [1,d], L2-normalized
        D, I = self.index.search(q, topn)  # IP on L2-norm -> cosine
        ids = I[0].tolist(); scores = D[0].tolist()
        return [(int(i), float(s)) for i, s in zip(ids, scores) if i != -1]

    def bm25_search(self, query: str, topn: int) -> List[Tuple[int, float]]:
        if self.bm25 is None:
            return []
        toks = query.lower().split()
        scores = self.bm25.get_scores(toks)
        # take topn indices
        idx = np.argpartition(-scores, kth=min(topn, len(scores)-1))[:topn]
        idx = idx[np.argsort(-scores[idx])]
        return [(int(i), float(scores[int(i)])) for i in idx]

    def search(self, query: str, topn_dense: int = 200, topn_bm25: int = 0,
               mmr_k: int = 50, mmr_lambda: float = 0.5) -> List[Dict]:
        dense = self.dense_search(query, topn_dense)
        cand_ids = [i for i,_ in dense]
        cand_vecs = encode_texts([self.texts[i] for i in cand_ids], cfg=self.enc_cfg)

        # Optional BM25 fusion (RRF)
        if topn_bm25 > 0 and self.bm25 is not None:
            bm = self.bm25_search(query, topn_bm25)
            # reciprocal rank fusion over union
            ranks = {}
            # dense ranks
            for r,(i,_) in enumerate(dense,1):
                ranks.setdefault(i, 0.0)
                ranks[i] += 1.0/(60 + r)
            # bm25 ranks
            for r,(i,_) in enumerate(bm,1):
                ranks.setdefault(i, 0.0)
                ranks[i] += 1.0/(60 + r)
            fused_ids = [i for i,_ in sorted(ranks.items(), key=lambda x:x[1], reverse=True)]
            # re-embed fused for MMR
            cand_ids = fused_ids[:max(mmr_k*4, 200)]
            cand_vecs = encode_texts([self.texts[i] for i in cand_ids], cfg=self.enc_cfg)

        # MMR down to mmr_k
        qvec = encode_texts([query], cfg=self.enc_cfg)[0]
        keep = mmr(qvec, cand_vecs, k=mmr_k, lambda_mult=mmr_lambda)
        out = []
        for rank, j in enumerate(keep):
            cid = int(cand_ids[j])
            out.append({
                "doc_id": str(cid),
                "chunk_global_id": cid,
                "text": self.texts[cid],
                "rank": rank,
            })
        return out
