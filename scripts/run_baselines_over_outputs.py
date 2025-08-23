#!/usr/bin/env python3
import argparse, json, os, sys, csv, glob, time, math
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import subprocess
import numpy as np

def jload(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def jdump(o,p):
    with open(p, "w", encoding="utf-8") as f:
        json.dump(o, f, ensure_ascii=False, indent=2)

def run(cmd: List[str]):
    print("[RUN]", " ".join(cmd), flush=True)
    t0 = time.time()
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    dt = time.time() - t0
    print(res.stdout)
    if res.returncode != 0:
        raise SystemExit(f"Command failed ({dt:.2f}s): {' '.join(cmd)}")
    print(f"[OK] done in {dt:.2f}s\n", flush=True)
    return res

def spearman(a: Dict[int,float], b: Dict[int,float]) -> Optional[float]:
    # align over union of keys; zeros if missing
    keys = sorted(set(a.keys()) | set(b.keys()))
    if len(keys) < 2: return None
    xa = np.array([a.get(k, 0.0) for k in keys], dtype=float)
    xb = np.array([b.get(k, 0.0) for k in keys], dtype=float)

    # rank
    ra = np.argsort(np.argsort(xa))
    rb = np.argsort(np.argsort(xb))
    ra = ra.astype(float); rb = rb.astype(float)
    # handle ties by assigning average ranks (simple tie-correction)
    # quick tie-fix: replace equal values with average rank within tie group
    def avg_rank(x):
        order = np.argsort(x)
        ranks = np.empty_like(x, dtype=float)
        i = 0
        while i < len(x):
            j = i
            while j+1 < len(x) and x[order[j+1]] == x[order[i]]:
                j += 1
            r = (i + j) / 2.0
            for k in range(i, j+1):
                ranks[order[k]] = r
            i = j+1
        return ranks
    ra = avg_rank(xa)
    rb = avg_rank(xb)
    # pearson over ranks
    if np.std(ra) < 1e-12 or np.std(rb) < 1e-12: return None
    rho = np.corrcoef(ra, rb)[0,1]
    return float(rho)

def topk_ids(d: Dict[int,float], k: int) -> List[int]:
    return [int(k) for k,_ in sorted(d.items(), key=lambda kv: kv[1], reverse=True)[:k]]

def jaccard(a: List[int], b: List[int]) -> Optional[float]:
    if not a and not b: return None
    return float(len(set(a) & set(b)) / max(1, len(set(a) | set(b))))

def aggregate_raags(validated: Dict[str, List[Dict]]) -> Dict[int,float]:
    agg: Dict[int,float] = {}
    for _t, edges in validated.items():
        for e in edges:
            cid = int(e.get("chunk_global_id", -1))
            w   = float(e.get("weight", 0.0))
            if cid >= 0:
                agg[cid] = agg.get(cid, 0.0) + w
    return agg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_dir", default="outputs")
    ap.add_argument("--model_name", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--layers", default="last8")
    ap.add_argument("--permutations", type=int, default=16, help="doc-Shapley permutations")
    ap.add_argument("--k_edges", type=int, default=8, help="Retrieval baseline per-token list")
    ap.add_argument("--norm", choices=["softmax","minmax"], default="softmax")
    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--max_questions", type=int, default=0, help="0 = all")
    ap.add_argument("--out_csv", default="outputs/baseline_compare.csv")
    args = ap.parse_args()

    wd = Path(args.work_dir)
    wd.mkdir(parents=True, exist_ok=True)

    # discover examples we’ve already run RAAG on
    bundles = sorted(glob.glob(str(wd / "*_bundle.json")))
    items = []
    for b in bundles:
        stem = os.path.basename(b).replace("_bundle.json","")
        e = wd / f"{stem}_edges.json"
        v = wd / f"{stem}_validated.json"
        if os.path.exists(e) and os.path.exists(v):
            items.append((stem, b, str(e), str(v)))
    if args.max_questions > 0:
        items = items[:args.max_questions]

    if not items:
        print(f"[WARN] no items with bundle+edges+validated found in {wd}")
        sys.exit(0)

    rows = []
    for stem, bundle_path, edges_path, validated_path in items:
        print(f"=== {stem} ===")
        bundle   = jload(bundle_path)
        validated = jload(validated_path).get("validated_edges", {})
        raag_chunk = aggregate_raags(validated)

        # ----- baseline A: retrieval-rank -----
        retr_out = str(wd / f"{stem}_baseline_retrieval.json")
        run([
            sys.executable, "-m", "scripts.baseline_retrieval_rank",
            "--bundle_json", bundle_path,
            "--edges_json",  edges_path,
            "--k_edges",     str(args.k_edges),
            "--norm",        args.norm,
            "--out",         retr_out
        ])
        retr_blob = jload(retr_out)
        # turn retrieval scores (or edge weights) into a chunk-importance vector
        if retr_blob.get("chunk_scores"):
            retr_chunk = {int(k): float(v) for k,v in retr_blob["chunk_scores"].items()}
        else:
            retr_chunk = {}
            for _t, lst in retr_blob.get("edges", {}).items():
                for e in lst:
                    cid = int(e.get("chunk_global_id", -1))
                    w   = float(e.get("weight", 0.0))
                    if cid >= 0:
                        retr_chunk[cid] = retr_chunk.get(cid, 0.0) + w

        # ----- baseline B: document Shapley -----
        shap_out = str(wd / f"{stem}_shapley_doc.json")
        cmd = [
            sys.executable, "-m", "scripts.baseline_shapley",
            "--bundle_json", bundle_path,
            "--edges_json",  edges_path,
            "--model_name",  args.model_name,
            "--layers",      args.layers,
            "--mode",        "doc",
            "--permutations", str(args.permutations),
            "--out",         shap_out
        ]
        if args.load_in_4bit:
            cmd.insert(5, "--load_in_4bit")
        run(cmd)
        shap_blob = jload(shap_out)
        shap_chunk = {int(k): float(v) for k,v in shap_blob.get("phi_by_chunk", {}).items()}

        # --- comparisons ---
        rho_retr = spearman(raag_chunk, retr_chunk)
        rho_shap = spearman(raag_chunk, shap_chunk)

        raag_top1 = topk_ids(raag_chunk, 1)
        retr_top1 = topk_ids(retr_chunk, 1)
        shap_top1 = topk_ids(shap_chunk, 1)

        top1_match_retr = int(bool(raag_top1 and retr_top1 and raag_top1[0] == retr_top1[0]))
        top1_match_shap = int(bool(raag_top1 and shap_top1 and raag_top1[0] == shap_top1[0]))

        raag_top3 = topk_ids(raag_chunk, 3)
        retr_top3 = topk_ids(retr_chunk, 3)
        shap_top3 = topk_ids(shap_chunk, 3)

        jacc_retr = jaccard(raag_top3, retr_top3)
        jacc_shap = jaccard(raag_top3, shap_top3)

        rows.append({
            "id": stem,
            "rho_retrieval": "" if rho_retr is None else rho_retr,
            "rho_shapley_doc": "" if rho_shap is None else rho_shap,
            "top1_match_retrieval": top1_match_retr,
            "top1_match_shapley_doc": top1_match_shap,
            "jaccard_top3_retrieval": "" if jacc_retr is None else jacc_retr,
            "jaccard_top3_shapley_doc": "" if jacc_shap is None else jacc_shap,
            "raag_top1": "" if not raag_top1 else raag_top1[0],
            "retr_top1": "" if not retr_top1 else retr_top1[0],
            "shap_top1": "" if not shap_top1 else shap_top1[0],
        })

    # write CSV
    outp = Path(args.out_csv)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", newline="", encoding="utf-8") as cf:
        fieldnames = list(rows[0].keys()) if rows else [
            "id","rho_retrieval","rho_shapley_doc",
            "top1_match_retrieval","top1_match_shapley_doc",
            "jaccard_top3_retrieval","jaccard_top3_shapley_doc",
            "raag_top1","retr_top1","shap_top1"
        ]
        w = csv.DictWriter(cf, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"[OK] wrote {outp} with {len(rows)} rows.")

    # Simple aggregate printout
    def _avg(vals):
        xs = [float(v) for v in vals if str(v) != ""]
        return (sum(xs)/len(xs)) if xs else float("nan")

    avg_rho_retr = _avg([r["rho_retrieval"] for r in rows])
    avg_rho_shap = _avg([r["rho_shapley_doc"] for r in rows])
    avg_t1_retr  = _avg([r["top1_match_retrieval"] for r in rows])
    avg_t1_shap  = _avg([r["top1_match_shapley_doc"] for r in rows])
    avg_j3_retr  = _avg([r["jaccard_top3_retrieval"] for r in rows])
    avg_j3_shap  = _avg([r["jaccard_top3_shapley_doc"] for r in rows])

    print("\n===== Baseline vs RAAG (aggregate) =====")
    print(f"Spearman ρ  (retrieval)    : {avg_rho_retr:.3f}")
    print(f"Spearman ρ  (shapley-doc)  : {avg_rho_shap:.3f}")
    print(f"Top-1 match (retrieval)    : {avg_t1_retr:.3f}")
    print(f"Top-1 match (shapley-doc)  : {avg_t1_shap:.3f}")
    print(f"Jaccard@3  (retrieval)     : {avg_j3_retr:.3f}")
    print(f"Jaccard@3  (shapley-doc)   : {avg_j3_shap:.3f}")

if __name__ == "__main__":
    main()


