#!/usr/bin/env python3
"""
Batch runner for RAAG on a QA JSONL.

For each example (id, question, answer):
  1) make_prompt.py          -> bundle (prompt + retrieved spans)
  2) score_edges.py          -> edges  (teacher-forced Δ log p with control + length-norm)
  3) validate_edges.py       -> validated edges (per-layer activation ablation)
  4) eval_provenance.py      -> provenance metrics (P@1, P@3, nDCG@3, AP, AUPRC)
  5) eval_hallucination.py   -> hallucination metrics (HRI etc.)

Outputs:
  - Per-item artifacts in <out_dir>/
  - Aggregate CSV at --out_csv
  - Summary JSON at <out_dir>/batch_summary.json
"""

import argparse
import csv
import json
import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, Any

def run(cmd):
    print("[RUN]", " ".join(cmd), flush=True)
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(res.stdout)
    if res.returncode != 0:
        raise SystemExit(f"Command failed: {' '.join(cmd)}")
    return res

def safe_read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except Exception:
            return {}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa_jsonl", required=True, help="JSONL with fields: id, question, answer")
    ap.add_argument("--index_path", required=True)
    ap.add_argument("--mapping_path", required=True)
    ap.add_argument("--encoder_name", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--generator_tokenizer", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--model_name", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--gen_max_new_tokens", type=int, default=32)

    # RAAG causal settings
    ap.add_argument("--layers", default="last12", help='Layer spec for score/validate (e.g., "last12" or "28,29,30,31")')
    ap.add_argument("--k_edges", type=int, default=5)
    ap.add_argument("--epsilon", type=float, default=0.10, help="Validation threshold (Δ log p)")

    # Outputs
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--out_csv", default="outputs/batch_metrics.csv")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(args.qa_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            qid = str(ex.get("id"))
            question = ex["question"]
            answer = ex["answer"]

            # Per-item artifact paths
            bundle_path     = out_dir / f"{qid}_bundle.json"
            edges_path      = out_dir / f"{qid}_edges.json"
            validated_path  = out_dir / f"{qid}_validated.json"
            prov_path       = out_dir / f"{qid}_prov.json"
            halluc_path     = out_dir / f"{qid}_hallucination.json"

            # 1) make prompt
            cmd = [
                sys.executable, "-m", "scripts.make_prompt",
                "--question", question,
                "--index_path", args.index_path,
                "--mapping_path", args.mapping_path,
                "--encoder_name", args.encoder_name,
                "--generator_tokenizer", args.generator_tokenizer,
                "--topk", str(args.topk),
                "--out", str(bundle_path)
            ]
            run(cmd)

            # 2) score edges (teacher-forced patched scorer)
            cmd = [
                sys.executable, "-m", "scripts.score_edges",
                "--bundle_json", str(bundle_path),
                "--model_name", args.model_name,
                "--gen_max_new_tokens", str(args.gen_max_new_tokens),
                "--layers", args.layers,
                "--k_edges", str(args.k_edges),
                "--length_norm",
                "--control_subtract",
                "--control_trials", "16",
                "--answer_string", str(answer),
                "--out", str(edges_path)
            ]
            if args.load_in_4bit:
                cmd.insert(5, "--load_in_4bit")  # after -m module
            run(cmd)

            # 3) validate edges
            cmd = [
                sys.executable, "-m", "scripts.validate_edges",
                "--bundle_json", str(bundle_path),
                "--edges_json", str(edges_path),
                "--model_name", args.model_name,
                "--layers", args.layers,
                "--epsilon", str(args.epsilon),
                "--out", str(validated_path)
            ]
            if args.load_in_4bit:
                cmd.insert(5, "--load_in_4bit")
            run(cmd)

            # 4) provenance metrics
            cmd = [
                sys.executable, "-m", "scripts.eval_provenance",
                "--bundle_json", str(bundle_path),
                "--validated_json", str(validated_path),
                "--answer", str(answer),
                "--k", "3",
                "--out", str(prov_path)
            ]
            run(cmd)

            # 5) hallucination metrics (entity-aware, uses answer substring for support;
            #    contradictions optional—script should default to none on toy)
            cmd = [
                sys.executable, "-m", "scripts.eval_hallucination",
                "--bundle_json", str(bundle_path),
                "--validated_json", str(validated_path),
                "--answer", str(answer),
                "--out", str(halluc_path)
            ]
            run(cmd)

            # Collect metrics
            prov = safe_read_json(prov_path) or {}
            hall = safe_read_json(halluc_path) or {}

            prov_m = (prov.get("metrics") or {})
            rows.append({
                "id": qid,
                "question": question,
                "answer": answer,
                "precision_at_1": prov_m.get("precision_at_1"),
                "precision_at_3": prov_m.get("precision_at_3"),
                "ndcg_at_3": prov_m.get("ndcg_at_3"),
                "average_precision": prov_m.get("average_precision"),
                "auprc": prov_m.get("auprc"),
                "halluc_correct": hall.get("correct"),
                "halluc_support_mass": hall.get("support_mass"),
                "halluc_contradiction_mass": hall.get("contradiction_mass"),
                "halluc_unlabeled_mass": hall.get("unlabeled_mass"),
                "halluc_total_mass": hall.get("total_mass"),
                "HRI": hall.get("HRI"),
            })

    # Write CSV
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as cf:
        fieldnames = list(rows[0].keys()) if rows else [
            "id","question","answer",
            "precision_at_1","precision_at_3","ndcg_at_3","average_precision","auprc",
            "halluc_correct","halluc_support_mass","halluc_contradiction_mass",
            "halluc_unlabeled_mass","halluc_total_mass","HRI"
        ]
        w = csv.DictWriter(cf, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[OK] wrote {str(out_csv)} with {len(rows)} rows.")

    # Summary JSON
    def mean(values):
        vals = [v for v in values if isinstance(v, (int, float))]
        return (sum(vals) / len(vals)) if vals else None

    summary = {
        "num_examples": len(rows),
        "provenance": {
            "precision_at_1": mean([r.get("precision_at_1") for r in rows]),
            "precision_at_3": mean([r.get("precision_at_3") for r in rows]),
            "ndcg_at_3":      mean([r.get("ndcg_at_3")      for r in rows]),
            "average_precision": mean([r.get("average_precision") for r in rows]),
            "auprc": mean([r.get("auprc") for r in rows])
        },
        "hallucination": {
            "accuracy": mean([1.0 if r.get("halluc_correct") is True else 0.0 for r in rows]),
            "HRI": mean([r.get("HRI") for r in rows]),
            "support_mass": mean([r.get("halluc_support_mass") for r in rows]),
            "contradiction_mass": mean([r.get("halluc_contradiction_mass") for r in rows]),
            "unlabeled_mass": mean([r.get("halluc_unlabeled_mass") for r in rows]),
            "total_mass": mean([r.get("halluc_total_mass") for r in rows]),
        },
        "config": {
            "model_name": args.model_name,
            "layers": args.layers,
            "epsilon": args.epsilon,
            "k_edges": args.k_edges,
            "gen_max_new_tokens": args.gen_max_new_tokens,
            "topk_retrieval": args.topk,
            "encoder_name": args.encoder_name,
            "index_path": args.index_path,
            "mapping_path": args.mapping_path,
            "load_in_4bit": bool(args.load_in_4bit)
        }
    }
    summary_path = out_dir / "batch_summary.json"
    with open(summary_path, "w", encoding="utf-8") as sf:
        json.dump(summary, sf, ensure_ascii=False, indent=2)
    print(f"[OK] wrote {str(summary_path)}")

if __name__ == "__main__":
    main()
