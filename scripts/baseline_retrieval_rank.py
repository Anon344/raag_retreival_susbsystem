#!/usr/bin/env python3
import argparse, json, math
from typing import Dict, List, Tuple, Optional
import numpy as np

def jload(p): return json.load(open(p, "r", encoding="utf-8"))
def jdump(o,p): json.dump(o, open(p,"w",encoding="utf-8"), ensure_ascii=False, indent=2)

def softmax(xs: List[float]) -> List[float]:
    if not xs: return []
    m = max(xs)
    ex = [math.exp(x - m) for x in xs]
    Z = sum(ex) or 1.0
    return [e / Z for e in ex]

def normalize_scores(scores: List[float], scheme: str) -> List[float]:
    if scheme == "softmax": return softmax(scores)
    # min-max (robust)
    lo, hi = min(scores), max(scores)
    if hi - lo <= 1e-9:
        return [1.0/len(scores)] * len(scores) if scores else []
    return [(s - lo) / (hi - lo) for s in scores]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle_json", required=True)
    ap.add_argument("--edges_json", required=True, help="Use RAAG tf_input_ids / positions for alignment")
    ap.add_argument("--k_edges", type=int, default=8)
    ap.add_argument("--norm", choices=["softmax","minmax"], default="softmax")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    bundle = jload(args.bundle_json)
    edges  = jload(args.edges_json)

    # Use teacher-forced positions so baseline aligns with RAAG
    tf_ids   = edges.get("tf_input_ids")
    prompt_len = int(edges.get("prompt_len", len(bundle.get("input_ids", []))))
    ans_positions = edges.get("answer_token_positions")
    if not isinstance(tf_ids, list) or not isinstance(ans_positions, list):
        raise RuntimeError("[retrieval_baseline] edges_json must contain tf_input_ids and answer_token_positions")

    # Pull top-k retrieved and normalize their scores
    retrieved = bundle.get("retrieved", []) or []
    if not retrieved:
        # graceful empty
        out = {
            "source": "retrieval_rank",
            "edges": {},
            "tf_input_ids": tf_ids,
            "prompt_len": prompt_len,
            "answer_token_positions": ans_positions,
            "chunk_scores": {},
        }
        jdump(out, args.out)
        print(f"[OK][retrieval_baseline] wrote {args.out} (no retrieved chunks)")
        return

    # expect each item to have chunk_global_id and a score
    rows = []
    for r in retrieved:
        cid = int(r.get("chunk_global_id", -1))
        sc  = float(r.get("score", 0.0))
        if cid >= 0:
            rows.append((cid, sc))
    # dedup by cid, keep the best score
    best = {}
    for cid, sc in rows:
        if (cid not in best) or (sc > best[cid]):
            best[cid] = sc
    # sort & top-k for edges
    sorted_items = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
    sorted_items = sorted_items[:args.k_edges]
    if not sorted_items:
        out = {
            "source": "retrieval_rank",
            "edges": {},
            "tf_input_ids": tf_ids,
            "prompt_len": prompt_len,
            "answer_token_positions": ans_positions,
            "chunk_scores": {},
        }
        jdump(out, args.out)
        print(f"[OK][retrieval_baseline] wrote {args.out} (no valid chunk ids)")
        return

    cids, scores = zip(*sorted_items)
    weights = normalize_scores(list(scores), args.norm)

    # Per-token edges: replicate the same top-k list to every answer token
    edges_out: Dict[str, List[Dict]] = {}
    for t in ans_positions:
        edges_out[str(int(t))] = [
            {"span": [-1, -1], "chunk_global_id": int(cid), "weight": float(w)}
            for cid, w in zip(cids, weights)
        ]

    out = {
        "source": "retrieval_rank",
        "edges": edges_out,
        "tf_input_ids": tf_ids,
        "prompt_len": prompt_len,
        "answer_token_positions": ans_positions,
        "chunk_scores": {int(cid): float(best[cid]) for cid in best.keys()},
        "config": {"k_edges": args.k_edges, "norm": args.norm}
    }
    jdump(out, args.out)
    print(f"[OK][retrieval_baseline] wrote {args.out}  |  k={args.k_edges}")

if __name__ == "__main__":
    main()
