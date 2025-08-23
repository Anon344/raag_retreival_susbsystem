#!/usr/bin/env python3
import argparse, json, os, re, csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional

def jload(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def safe_jload(p: Path):
    try:
        return jload(p)
    except Exception:
        return None

def find_runs(roots: List[str]) -> Dict[str, Dict[str, Path]]:
    """
    Return mapping: id -> {
        'dataset': <dataset_dir_name>,
        'validated': Path,
        'edges': Path or None,
        'baseline_retrieval': Path or None,
        'baseline_shapley': Path or None,
    }
    """
    out: Dict[str, Dict[str, Path]] = {}
    for root in roots:
        rpath = Path(root)
        if not rpath.exists():
            continue
        for p in rpath.glob("*_validated.json"):
            stem = p.name.replace("_validated.json", "")
            ex = out.setdefault(stem, {})
            ex["dataset"] = rpath.name
            ex["validated"] = p
            cand_edges = p.with_name(stem + "_edges.json")
            if cand_edges.exists():
                ex["edges"] = cand_edges
            # baselines
            bretr = p.with_name(stem + "_baseline_retrieval.json")
            if bretr.exists():
                ex["baseline_retrieval"] = bretr
            bshape = p.with_name(stem + "_shapley_doc.json")
            if bshape.exists():
                ex["baseline_shapley"] = bshape
    return out

def aggregate_raags_mass(validated_json: Dict) -> Dict[int, float]:
    """Sum validated weights per chunk_global_id."""
    masses: Dict[int, float] = {}
    vedges = validated_json.get("validated_edges", {})
    for _t, lst in vedges.items():
        for e in lst:
            cid = int(e.get("chunk_global_id", -1))
            w = float(e.get("weight", 0.0))
            if cid >= 0:
                masses[cid] = masses.get(cid, 0.0) + w
    return masses

def load_retrieval_baseline(bjson: Dict) -> Dict[int, float]:
    """
    Try multiple shapes:
      - {"rank_by_chunk": {cid: rank}, ...}
      - {"score_by_chunk": {cid: score}, ...}
      - {"ranking":[cid,...]} (1-best first)
      - {"retrieved":[{"chunk_global_id":cid,"rank":r,"score":s},...]}
    Returns cid->score (higher is better). If only ranks are present, use -rank.
    """
    if "score_by_chunk" in bjson and isinstance(bjson["score_by_chunk"], dict):
        return {int(k): float(v) for k, v in bjson["score_by_chunk"].items()}

    if "rank_by_chunk" in bjson and isinstance(bjson["rank_by_chunk"], dict):
        return {int(k): -float(v) for k, v in bjson["rank_by_chunk"].items()}

    if "ranking" in bjson and isinstance(bjson["ranking"], list):
        # assign descending scores from position
        ranking = [int(x) for x in bjson["ranking"]]
        n = len(ranking)
        return {cid: float(n - i) for i, cid in enumerate(ranking)}  # 1..n

    if "retrieved" in bjson and isinstance(bjson["retrieved"], list):
        # prefer score if present; else -rank; else descending by order
        have_score = any("score" in r for r in bjson["retrieved"])
        if have_score:
            return {int(r["chunk_global_id"]): float(r.get("score", 0.0)) for r in bjson["retrieved"]}
        have_rank = any("rank" in r for r in bjson["retrieved"])
        if have_rank:
            return {int(r["chunk_global_id"]): -float(r.get("rank", 1e9)) for r in bjson["retrieved"]}
        # fall back to list order
        n = len(bjson["retrieved"])
        return {int(r["chunk_global_id"]): float(n - i) for i, r in enumerate(bjson["retrieved"])}

    return {}

def load_shapley_baseline(bjson: Dict) -> Dict[int, float]:
    """
    Expect {"phi_by_chunk": {cid: phi}, ...}
    """
    if "phi_by_chunk" in bjson and isinstance(bjson["phi_by_chunk"], dict):
        return {int(k): float(v) for k, v in bjson["phi_by_chunk"].items()}
    return {}

def ranks_from_scores(scores: Dict[int, float]) -> Dict[int, int]:
    """
    Convert scores (higher better) to ranks 1..n (1=best). Break ties by (score desc, cid asc).
    """
    order = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return {cid: i + 1 for i, (cid, _) in enumerate(order)}

def top_k_list(scores: Dict[int, float], k: int) -> List[int]:
    return [cid for cid, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]]

def pearson_corr(x: List[float], y: List[float]) -> Optional[float]:
    n = len(x)
    if n < 2:
        return None
    mx = sum(x) / n
    my = sum(y) / n
    vx = sum((xi - mx) ** 2 for xi in x)
    vy = sum((yi - my) ** 2 for yi in y)
    if vx <= 0 or vy <= 0:
        return None
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    return cov / (vx ** 0.5) / (vy ** 0.5)

def spearman_from_rankmaps(r1: Dict[int, int], r2: Dict[int, int]) -> Optional[float]:
    common = sorted(set(r1.keys()) & set(r2.keys()))
    if len(common) < 2:
        return None
    a = [float(r1[c]) for c in common]
    b = [float(r2[c]) for c in common]
    return pearson_corr(a, b)

def jaccard(a: List[int], b: List[int]) -> Optional[float]:
    A, B = set(a), set(b)
    U = len(A | B)
    if U == 0:
        return None
    return len(A & B) / U

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", default=["outputs/natural_questions", "outputs/hotpot_qa"],
                    help="One or more directories to scan for *_validated.json and baselines")
    ap.add_argument("--out_csv", default="outputs/baseline_compare.csv")
    ap.add_argument("--k_jaccard", type=int, default=3)
    args = ap.parse_args()

    runs = find_runs(args.roots)
    print(f"[INFO] scanned {len(args.roots)} roots; found {len(runs)} RAAG validated runs")

    rows = []
    n_pair_retr, n_pair_shap = 0, 0
    n_missing_retr, n_missing_shap = 0, 0
    n_empty_raags = 0

    for rid, paths in sorted(runs.items()):
        validated_p: Optional[Path] = paths.get("validated")
        if validated_p is None or not validated_p.exists():
            continue
        vjson = safe_jload(validated_p)
        if vjson is None:
            continue

        # RAAG per-chunk mass
        raag_scores = aggregate_raags_mass(vjson)
        if not raag_scores:
            n_empty_raags += 1

        dataset = paths.get("dataset", Path(validated_p).parent.name)

        # Retrieval baseline
        r_spearman = r_top1 = r_j3 = None
        r_n_common = None
        if "baseline_retrieval" in paths and paths["baseline_retrieval"].exists():
            bjson = safe_jload(paths["baseline_retrieval"])
            if bjson is not None and raag_scores:
                retr_scores = load_retrieval_baseline(bjson)
                # intersect on cids with scores on both sides
                common = sorted(set(raag_scores.keys()) & set(retr_scores.keys()))
                r_n_common = len(common)
                if r_n_common and r_n_common >= 2:
                    r1 = ranks_from_scores({c: raag_scores[c] for c in common})
                    r2 = ranks_from_scores({c: retr_scores[c] for c in common})
                    r_spearman = spearman_from_rankmaps(r1, r2)
                # top-1 match (use global top1 from each side, not just intersection)
                raag_top1 = top_k_list(raag_scores, 1)
                retr_top1 = top_k_list(retr_scores, 1)
                if raag_top1 and retr_top1:
                    r_top1 = 1.0 if raag_top1[0] == retr_top1[0] else 0.0
                # Jaccard@k
                raag_topk = top_k_list(raag_scores, args.k_jaccard)
                retr_topk = top_k_list(retr_scores, args.k_jaccard)
                r_j3 = jaccard(raag_topk, retr_topk)
                n_pair_retr += 1
        else:
            n_missing_retr += 1

        # Doc-Shapley baseline
        s_spearman = s_top1 = s_j3 = None
        s_n_common = None
        if "baseline_shapley" in paths and paths["baseline_shapley"].exists():
            bjson = safe_jload(paths["baseline_shapley"])
            if bjson is not None and raag_scores:
                shap_scores = load_shapley_baseline(bjson)
                common = sorted(set(raag_scores.keys()) & set(shap_scores.keys()))
                s_n_common = len(common)
                if s_n_common and s_n_common >= 2:
                    r1 = ranks_from_scores({c: raag_scores[c] for c in common})
                    r2 = ranks_from_scores({c: shap_scores[c] for c in common})
                    s_spearman = spearman_from_rankmaps(r1, r2)
                raag_top1 = top_k_list(raag_scores, 1)
                shap_top1 = top_k_list(shap_scores, 1)
                if raag_top1 and shap_top1:
                    s_top1 = 1.0 if raag_top1[0] == shap_top1[0] else 0.0
                raag_topk = top_k_list(raag_scores, args.k_jaccard)
                shap_topk = top_k_list(shap_scores, args.k_jaccard)
                s_j3 = jaccard(raag_topk, shap_topk)
                n_pair_shap += 1
        else:
            n_missing_shap += 1

        rows.append({
            "id": rid,
            "dataset": dataset,
            # retrieval baseline metrics
            "retrieval_spearman": "" if r_spearman is None else f"{r_spearman:.6f}",
            "retrieval_top1_match": "" if r_top1 is None else f"{r_top1:.6f}",
            "retrieval_jaccard_at{K}".format(K=args.k_jaccard): "" if r_j3 is None else f"{r_j3:.6f}",
            "retrieval_n_common": "" if r_n_common is None else r_n_common,
            # shapley baseline metrics
            "shapley_spearman": "" if s_spearman is None else f"{s_spearman:.6f}",
            "shapley_top1_match": "" if s_top1 is None else f"{s_top1:.6f}",
            "shapley_jaccard_at{K}".format(K=args.k_jaccard): "" if s_j3 is None else f"{s_j3:.6f}",
            "shapley_n_common": "" if s_n_common is None else s_n_common,
        })

    outp = Path(args.out_csv)
    outp.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id", "dataset",
        "retrieval_spearman", "retrieval_top1_match", f"retrieval_jaccard_at{args.k_jaccard}", "retrieval_n_common",
        "shapley_spearman", "shapley_top1_match", f"shapley_jaccard_at{args.k_jaccard}", "shapley_n_common",
    ]
    with open(outp, "w", newline="", encoding="utf-8") as cf:
        w = csv.DictWriter(cf, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[OK] wrote {outp} with {len(rows)} rows.")
    print(f"[INFO] paired with retrieval baseline: {n_pair_retr} | missing files: {n_missing_retr}")
    print(f"[INFO] paired with shapley baseline:   {n_pair_shap} | missing files: {n_missing_shap}")
    print(f"[INFO] RAAG validated with zero mass (skip-worthy): {n_empty_raags}")

if __name__ == "__main__":
    main()
