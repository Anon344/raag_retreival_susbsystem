
# RAAG Retrieval Subsystem

- Generator-tokenizer-aware chunking (exact spans for later provenance)
- Dense retrieval with sentence-transformers (all-MiniLM-L6-v2)
- FAISS IP index build/search
- Rich chunk metadata written to index/chunk_meta.jsonl
- Prompt helper to map retrieved chunks into exact prompt token spans

See `scripts/build_index.py` and `scripts/search.py` for CLIs.
