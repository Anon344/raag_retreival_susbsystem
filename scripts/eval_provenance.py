#!/usr/bin/env python3
"""
Evaluate provenance at the CHUNK level from validated_edges.json.

Positives: chunks whose text contains the (case-insensitive) answer string.
Score per chunk: sum of validated weights across all answer tokens.
Metrics:
  - Precision@1, Precision@3 (and legacy Precision@K via --k)
  - nDCG@3
  - Average Precision (AP)
  - AUPRC (area under precision-recall)

Usage:
  python -m scripts.eval_provenance \
    --bundle_json tmp_bundle.json \
    --validated_json outputs/validated_edges.json \
    --answer "Alexander Fleming" \
    --k 3 \
    --out outputs/provenance_eval.json
"""
import argparse
import json
import math
from collections import defaultdict
from typing import List


def stable_ranking(scores: List[float]) -> List[int]:
    """
    Deterministic ranking: sort by score desc, then by index asc.
    Returns a list of indices.
    """
    return sorted(range(len(scores)), key=lambda i: (-scores[i], i))


def auprc_from_scores(scores, labels):
    """
    Compute AUPRC via step-wise interpolation along the ranked list.
    If there are no positives, returns NaN.
    """
    order = stable_ranking(scores)
    tp, fp, fn = 0, 0, sum(labels)
    if fn == 0:
        return float("nan")

    prev_recall = 0.0
    area = 0.0
    for i in order:
        if labels[i] == 1:
            tp += 1
            fn -= 1
        else:
            fp += 1
        recall = tp / (tp + fn)
        precision = tp / max(1, (tp + fp))
        area += (recall - prev_recall) * precision
        prev_recall = recall
    return area


def precision_at_k(order, labels, k: int) -> float:
    k = max(1, k)
    top = order[:k]
    return sum(labels[i] for i in top) / float(k)


def dcg_at_k(order, labels, k: int) -> float:
    k = max(1, k)
    dcg = 0.0
    for rank, i in enumerate(order[:k], start=1):
        rel = labels[i]
        if rel:
            dcg += 1.0 / math.log2(rank + 1)
    return dcg


def ndcg_at_k(order, labels, k: int) -> float:
    k = max(1, k)
    ideal_order = stable_ranking(labels)  # labels are 0/1, so this places all 1s first
    idcg = dcg_at_k(ideal_order, labels, k)
    if idcg == 0.0:
        return float("nan")
    return dcg_at_k(order, labels, k) / idcg


def average_precision(order, labels) -> float:
    """
    AP = mean of precision@rank over ranks where label==1.
    If no positives, returns NaN.
    """
    num_pos = sum(labels)
    if num_pos == 0:
        return float("nan")
    tp = 0
    ap = 0.0
    for rank, i in enumerate(order, start=1):
        if labels[i] == 1:
            tp += 1
            ap += tp / rank
    return ap / num_pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle_json", required=True)
    ap.add_argument("--validated_json", required=True)
    ap.add_argument("--answer", required=True)
    ap.add_argument("--k", type=int, default=3, help="Legacy Precision@K to report alongside detailed metrics")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    bundle = json.loads(open(args.bundle_json, "r", encoding="utf-8").read())
    val = json.loads(open(args.validated_json, "r", encoding="utf-8").read())
    validated = val.get("validated_edges", {})

    # Build chunk_id -> text map
    chunk_text = {}
    for ch in bundle.get("chunk_spans", []):
        chunk_text[int(ch["chunk_global_id"])] = ch["text"]

    # Aggregate validated mass per chunk
    per_chunk = defaultdict(float)
    for _, lst in validated.items():
        for e in lst:
            cid = int(e.get("chunk_global_id", -1))
            w = float(e.get("weight", 0.0))
            if cid >= 0:
                per_chunk[cid] += w

    # Gold labels by answer substring match (case-insensitive)
    ans_lc = args.answer.lower()
    labels = {}
    for cid, text in chunk_text.items():
        labels[cid] = 1 if (ans_lc in text.lower()) else 0

    # Align vectors
    all_cids = sorted(chunk_text.keys())
    scores = [per_chunk.get(cid, 0.0) for cid in all_cids]
    gold = [labels.get(cid, 0) for cid in all_cids]

    # Rankings and metrics
    order = stable_ranking(scores)

    p_at_1 = precision_at_k(order, gold, 1)
    p_at_3 = precision_at_k(order, gold, 3)
    p_at_k = precision_at_k(order, gold, args.k)  # legacy
    ndcg3 = ndcg_at_k(order, gold, 3)
    ap = average_precision(order, gold)
    prc = auprc_from_scores(scores, gold)

    # Prepare detailed chunk table (ranked)
    ranked = [
        {
            "rank": r + 1,
            "cid": all_cids[i],
            "score": scores[i],
            "label": gold[i],
            "preview": (chunk_text.get(all_cids[i], "")[:160].replace("\n", " "))
        }
        for r, i in enumerate(order)
    ]

    out = {
        "metrics": {
            "precision_at_1": p_at_1,
            "precision_at_3": p_at_3,
            "precision_at_k": p_at_k,   # legacy, uses --k
            "ndcg_at_3": ndcg3,
            "average_precision": ap,
            "auprc": prc
        },
        "ranked_chunks": ranked
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # Console summary
    def fmt(x):
        return "nan" if not (x == x) else f"{x:.3f}"
    print(
        "[OK] wrote {}  |  P@1={}  P@3={}  P@K={}  nDCG@3={}  AP={}  AUPRC={}".format(
            args.out, fmt(p_at_1), fmt(p_at_3), fmt(p_at_k), fmt(ndcg3), fmt(ap), fmt(prc)
        )
    )


if __name__ == "__main__":
    main()
