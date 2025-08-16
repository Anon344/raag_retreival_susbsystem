#!/usr/bin/env python3
"""
Build a RAG prompt bundle for RAAG.

Retrieval:
  - FAISS index at --index_path
  - Chunk metadata JSONL at --mapping_path (one line per chunk; includes 'text' and optionally 'chunk_global_id')
  - Encoder: sentence-transformers (e.g., all-MiniLM-L6-v2)

Prompt:
  You are a helpful assistant. Use the retrieved evidence to answer.

  [DOC 0]
  <text0>

  [DOC 1]
  <text1>
  ...
  Question: <question>
  Answer:

Outputs (to --out):
  {
    "prompt": str,
    "input_ids": List[int],
    "chunk_spans": [
      {"chunk_global_id": int, "tok_start_in_prompt": int, "tok_end_in_prompt": int, "text": str}
    ],
    "retrieved": [<topk doc dicts with rank/score/...>],
    "question_span": [start_tok, end_tok],
    "question": str
  }

Usage:
  python -m scripts.make_prompt \
    --question "Who discovered penicillin?" \
    --index_path index/faiss.index \
    --mapping_path index/chunk_meta.jsonl \
    --encoder_name sentence-transformers/all-MiniLM-L6-v2 \
    --generator_tokenizer meta-llama/Meta-Llama-3-8B-Instruct \
    --topk 5 \
    --out outputs/nq1_bundle.json
"""
import argparse, json, os, sys
from typing import List, Dict, Any
import faiss
import numpy as np

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            rows.append(json.loads(line))
    return rows

def load_encoder(name: str):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(name)
    return model

def encode_query(encoder, text: str) -> np.ndarray:
    emb = encoder.encode([text], normalize_embeddings=True)
    return np.asarray(emb, dtype="float32")

def load_tokenizer(name: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name, use_fast=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    return tok

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", required=True)
    ap.add_argument("--index_path", required=True)
    ap.add_argument("--mapping_path", required=True)
    ap.add_argument("--encoder_name", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--generator_tokenizer", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--out", required=True, help="Path to write bundle JSON")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # 1) Load FAISS & mapping
    index = faiss.read_index(args.index_path)
    meta = read_jsonl(args.mapping_path)
    if len(meta) == 0:
        print(f"[ERROR] mapping file {args.mapping_path} is empty.", file=sys.stderr)
        sys.exit(2)

    # 2) Encode query & search
    encoder = load_encoder(args.encoder_name)
    qvec = encode_query(encoder, args.question)  # [1, d]
    scores, ids = index.search(qvec, args.topk)  # scores: [1,topk], ids: [1,topk]
    scores, ids = scores[0].tolist(), ids[0].tolist()

    # 3) Gather top-k metadata
    retrieved = []
    for rank, (idx, sc) in enumerate(zip(ids, scores)):
        if idx < 0 or idx >= len(meta):
            # FAISS can return -1 if not found
            continue
        m = dict(meta[idx])  # shallow copy
        m["rank"] = rank
        m["score"] = float(sc)
        # ensure chunk_global_id exists and matches row index if absent
        if "chunk_global_id" not in m:
            m["chunk_global_id"] = idx
        retrieved.append(m)

    # 4) Build prompt text
    header = "You are a helpful assistant. Answer the following question. The following information is provided as a resource. Only respond with your final answer in one line: do not include anything else in the response.\n\n"
    blocks = [header]
    for i, m in enumerate(retrieved):
        blocks.append(f"[DOC {i}]\n")
        blocks.append(m.get("text", "").strip() + "\n\n")
    qline = f"Question: {args.question}\n"
    aline = "Answer:"
    blocks.append(qline)
    blocks.append(aline)
    prompt = "".join(blocks)

    # 5) Tokenize and compute spans (prompt coordinates)
    tok = load_tokenizer(args.generator_tokenizer)

    def enc(txt: str) -> List[int]:
        return tok.encode(txt, add_special_tokens=False)

    input_ids: List[int] = []
    chunk_spans: List[Dict[str, Any]] = []

    # Encode header
    ids_header = enc(header)
    input_ids.extend(ids_header)
    cur = len(input_ids)

    # Encode each DOC block; record spans ONLY over the document text (exclude headers)
    for i, m in enumerate(retrieved):
        doc_header = f"[DOC {i}]\n"
        ids_doc_header = enc(doc_header)
        input_ids.extend(ids_doc_header)
        cur += len(ids_doc_header)

        doc_text = (m.get("text") or "").strip()
        ids_doc_text = enc(doc_text)
        start = cur
        end = cur + len(ids_doc_text)

        input_ids.extend(ids_doc_text)
        cur = end

        # trailing blank line
        ids_gap = enc("\n\n")
        input_ids.extend(ids_gap)
        cur += len(ids_gap)

        chunk_spans.append({
            "chunk_global_id": int(m.get("chunk_global_id", i)),
            "tok_start_in_prompt": start,
            "tok_end_in_prompt": end,
            "text": doc_text
        })

    # Encode question + answer cue
    ids_q = enc(qline)
    q_start = cur
    q_end = cur + len(ids_q)
    input_ids.extend(ids_q)
    cur = q_end

    ids_a = enc(aline)
    input_ids.extend(ids_a)
    cur += len(ids_a)

    # 6) Emit bundle
    bundle = {
        "prompt": prompt,
        "input_ids": input_ids,
        "chunk_spans": chunk_spans,
        "retrieved": retrieved,
        "question_span": [q_start, q_end],
        "question": args.question
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False, indent=2)
    print(f"[OK] wrote {args.out} (topk={len(retrieved)})")

if __name__ == "__main__":
    main()
