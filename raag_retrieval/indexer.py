
from typing import Iterable, Dict, List
import os, json
from transformers import AutoTokenizer
from .chunker import chunk_by_generator_tokens
from .embedder import STEmbedder

try:
    import faiss  # type: ignore
except Exception:
    faiss = None

def build_index(docs: Iterable[Dict], generator_tokenizer_name: str, encoder_name: str,
                index_path: str, mapping_path: str, chunk_size: int = 256, chunk_overlap: int = 32,
                batch_size: int = 64) -> int:
    assert faiss is not None, "faiss-cpu is required to build the index."
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    os.makedirs(os.path.dirname(mapping_path), exist_ok=True)

    gtok = AutoTokenizer.from_pretrained(generator_tokenizer_name, use_fast=True)
    embedder = STEmbedder(encoder_name)

    all_chunks: List[str] = []
    metas: List[Dict] = []

    for doc in docs:
        chunks = chunk_by_generator_tokens(doc["text"], gtok, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for local_id, ch in enumerate(chunks):
            all_chunks.append(ch["text"])
            metas.append({
                "doc_id": doc["id"],
                "title": doc.get("title"),
                "chunk_local_id": local_id,
                "tok_start": ch["tok_start"],
                "tok_end": ch["tok_end"],
                "char_start": ch["char_start"],
                "char_end": ch["char_end"],
                "text": ch["text"],
            })

    vecs = embedder.encode(all_chunks, batch_size=batch_size, normalize=True)
    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)

    faiss.write_index(index, index_path)

    with open(mapping_path, "w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    return len(metas)
