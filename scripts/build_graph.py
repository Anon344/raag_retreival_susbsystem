#!/usr/bin/env python3
"""
Build an interactive RAAG graph from validated edges.

- Aggregates validated edges to the CHUNK level (sum weights per (chunk, t))
- Optional TruncatedSVD on the chunk x token matrix
- Exports interactive HTML via PyVis (headless-safe)

Usage:
  python -m scripts.build_graph \
    --bundle_json tmp_bundle.json \
    --validated_json outputs/validated_edges.json \
    --svd_rank 8 \
    --similarity_thresh 0.9 \
    --html_out outputs/raag_graph.html \
    --json_out outputs/raag_graph.json \
    --model_name meta-llama/Meta-Llama-3-8B-Instruct
"""
import argparse, json
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
from sklearn.decomposition import TruncatedSVD
from pyvis.network import Network
from transformers import AutoTokenizer


def aggregate_chunk_token_matrix(bundle: Dict, validated: Dict[str, List[Dict]]):
    # Map chunk_id -> display text
    chunk_text = {}
    for ch in bundle.get("chunk_spans", []):
        cid = int(ch.get("chunk_global_id", -1))
        if cid >= 0:
            chunk_text[cid] = ch.get("text", "")

    # Build (chunk x token) matrix
    target_positions = sorted(int(t) for t in validated.keys())
    chunk_ids = sorted(chunk_text.keys())

    idx_c = {cid: i for i, cid in enumerate(chunk_ids)}
    idx_t = {t: j for j, t in enumerate(target_positions)}

    W = np.zeros((len(chunk_ids), len(target_positions)), dtype=np.float32)

    # Sum weights per (chunk, t)
    for t_str, edges in validated.items():
        t = int(t_str)
        if t not in idx_t:
            continue
        j = idx_t[t]
        for e in edges:
            cid = int(e.get("chunk_global_id", -1))
            i = idx_c.get(cid, None)
            if i is None:
                continue
            W[i, j] += float(e.get("weight", 0.0))

    return W, chunk_ids, target_positions, chunk_text


def build_graph_html(W,
                     chunk_ids,
                     target_positions,
                     chunk_text,
                     bundle,
                     model_name,
                     svd_rank,
                     similarity_thresh,
                     html_out,
                     json_out,
                     tf_input_ids=None,
                     prompt_len=None):
    # Optional SVD (kept for analysis / later merging)
    if svd_rank and svd_rank > 0 and min(W.shape) >= svd_rank:
        _ = TruncatedSVD(n_components=svd_rank, random_state=0).fit_transform(W)

    # Build PyVis graph
    net = Network(height="900px", width="100%", directed=True, notebook=False, cdn_resources="remote")
    net.barnes_hut(gravity=-8000, central_gravity=0.2, spring_length=150, spring_strength=0.005)

    # Tokenizer to label output tokens
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # Add chunk nodes
    for cid in chunk_ids:
        full_txt = (chunk_text.get(cid, "") or "")
        txt = full_txt[:80].replace("\n", " ")
        net.add_node(f"C:{cid}", label=f"[C{cid}] {txt}", color="#3182bd", title=full_txt)

    # Add output token nodes (robust to teacher-forced vs prompt-only indexing)
    prompt_ids = bundle.get("input_ids", [])
    T_prompt = len(prompt_ids)
    T_tf = len(tf_input_ids) if isinstance(tf_input_ids, list) else None

    for t in target_positions:
        label = None
        # Prefer teacher-forced ids (absolute positions)
        if T_tf is not None and 0 <= t < T_tf:
            piece = tok.decode([tf_input_ids[t]], skip_special_tokens=True).strip()
            label = piece if piece else f"y@{t}"
        # Else try prompt ids if within range
        elif 0 <= t < T_prompt:
            piece = tok.decode([prompt_ids[t]], skip_special_tokens=True).strip()
            label = piece if piece else f"y@{t}"
        else:
            label = f"y@{t}"

        net.add_node(f"Y:{t}", label=label, color="#31a354", title=f"Output token index {t}")

    # Add edges with thickness proportional to weight
    for i, cid in enumerate(chunk_ids):
        for j, t in enumerate(target_positions):
            w = float(W[i, j])
            if w <= 0:
                continue
            net.add_edge(f"C:{cid}", f"Y:{t}", value=max(1, int(4 * w)), title=f"w={w:.3f}")

    # --- Headless-safe HTML write ---
    net.write_html(html_out, open_browser=False, notebook=False)

    out_json = {
        "chunks": [{"id": cid, "text": chunk_text[cid]} for cid in chunk_ids],
        "tokens": target_positions,
        "weights": W.tolist(),
        "meta": {
            "model_name": model_name,
            "prompt_len": int(prompt_len) if prompt_len is not None else T_prompt,
            "has_tf_input_ids": bool(tf_input_ids is not None),
        }
    }
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2)

    print(f"[OK] wrote {html_out} and {json_out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle_json", required=True)
    ap.add_argument("--validated_json", required=True)
    ap.add_argument("--svd_rank", type=int, default=8)
    ap.add_argument("--similarity_thresh", type=float, default=0.9)  # reserved for future merging
    ap.add_argument("--html_out", default="outputs/raag_graph.html")
    ap.add_argument("--json_out", default="outputs/raag_graph.json")
    ap.add_argument("--model_name", default="meta-llama/Meta-Llama-3-8B-Instruct")
    args = ap.parse_args()

    bundle = json.loads(open(args.bundle_json, "r", encoding="utf-8").read())
    val = json.loads(open(args.validated_json, "r", encoding="utf-8").read())

    tf_input_ids = val.get("tf_input_ids", None)
    prompt_len = val.get("config", {}).get("prompt_len", len(bundle.get("input_ids", [])))

    W, chunk_ids, target_positions, chunk_text = aggregate_chunk_token_matrix(bundle, val["validated_edges"])
    build_graph_html(
        W, chunk_ids, target_positions, chunk_text, bundle, args.model_name,
        args.svd_rank, args.similarity_thresh, args.html_out, args.json_out,
        tf_input_ids=tf_input_ids, prompt_len=prompt_len
    )


if __name__ == "__main__":
    main()
