#!/usr/bin/env python3
import argparse, json, os, sys, csv, subprocess
from pathlib import Path

def run(cmd):
    print("[RUN]", " ".join(cmd), flush=True)
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(res.stdout)
    if res.returncode != 0:
        raise SystemExit(f"Command failed: {' '.join(cmd)}")
    return res

def safe_load_json(p):
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    # Data + index
    ap.add_argument("--qa_jsonl", required=True, help="JSONL with fields: id, question, answer")
    ap.add_argument("--index_path", required=True)
    ap.add_argument("--mapping_path", required=True)
    # Retrieval/generation config
    ap.add_argument("--encoder_name", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--generator_tokenizer", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--model_name", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--gen_max_new_tokens", type=int, default=32)
    # RAAG scoring/validation
    ap.add_argument("--layers", default="last8")
    ap.add_argument("--k_edges", type=int, default=5)
    ap.add_argument("--epsilon", type=float, default=0.1)
    ap.add_argument("--length_norm", action="store_true")
    ap.add_argument("--control_subtract", action="store_true")
    ap.add_argument("--control_trials", type=int, default=16)
    # Provenance eval
    ap.add_argument("--k_prov", type=int, default=3)
    # (MNLI) hallucination options (optional; script has sensible defaults)
    ap.add_argument("--hallucination_nli_model", default="sileod/mdeberta-v3-base-tasksource-nli")
    ap.add_argument("--hallucination_tau_entail", type=float, default=0.50)
    ap.add_argument("--hallucination_tau_contra", type=float, default=0.50)
    ap.add_argument("--hallucination_batch_size", type=int, default=16)
    # Output
    ap.add_argument("--out_csv", default="outputs/batch_metrics.csv")
    ap.add_argument("--work_dir", default="outputs", help="Where to write intermediate JSONs")
    args = ap.parse_args()

    Path(args.work_dir).mkdir(parents=True, exist_ok=True)

    rows = []
    with open(args.qa_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            qid = str(ex.get("id"))
            question = ex["question"]
            answer = ex["answer"]

            # Paths
            bundle_path     = f"{args.work_dir}/{qid}_bundle.json"
            edges_path      = f"{args.work_dir}/{qid}_edges.json"
            validated_path  = f"{args.work_dir}/{qid}_validated.json"
            prov_path       = f"{args.work_dir}/{qid}_prov.json"
            halluc_path     = f"{args.work_dir}/{qid}_hallucination.json"
            rars_path       = f"{args.work_dir}/{qid}_rars.json"

            # 1) make_prompt
            run([
                sys.executable, "-m", "scripts.make_prompt",
                "--question", question,
                "--index_path", args.index_path,
                "--mapping_path", args.mapping_path,
                "--encoder_name", args.encoder_name,
                "--generator_tokenizer", args.generator_tokenizer,
                "--topk", str(args.topk),
                "--out", bundle_path
            ])

            # 2) score_edges
            cmd = [
                sys.executable, "-m", "scripts.score_edges",
                "--bundle_json", bundle_path,
                "--model_name", args.model_name,
                "--gen_max_new_tokens", str(args.gen_max_new_tokens),
                "--layers", args.layers,
                "--k_edges", str(args.k_edges),
                "--answer_string", answer,
                "--out", edges_path
            ]
            if args.load_in_4bit: cmd.insert(5, "--load_in_4bit")
            if args.length_norm: cmd.append("--length_norm")
            if args.control_subtract:
                cmd.append("--control_subtract")
                cmd.extend(["--control_trials", str(args.control_trials)])
            run(cmd)

            # 3) validate_edges
            cmd = [
                sys.executable, "-m", "scripts.validate_edges",
                "--bundle_json", bundle_path,
                "--edges_json", edges_path,
                "--model_name", args.model_name,
                "--layers", args.layers,
                "--epsilon", str(args.epsilon),
                "--out", validated_path
            ]
            if args.load_in_4bit: cmd.insert(5, "--load_in_4bit")
            run(cmd)

            # 4) provenance eval (P@K etc.)
            run([
                sys.executable, "-m", "scripts.eval_provenance",
                "--bundle_json", bundle_path,
                "--validated_json", validated_path,
                "--answer", answer,
                "--k", str(args.k_prov),
                "--out", prov_path
            ])
            prov = safe_load_json(prov_path) or {}
            metrics = prov.get("metrics", {})
            p_at_1  = metrics.get("precision_at_1")
            p_at_3  = metrics.get("precision_at_3")
            ndcg3   = metrics.get("ndcg_at_3")
            ap      = metrics.get("average_precision")
            auprc   = metrics.get("auprc")

            # 5) Hallucination (MNLI HRI) – supply gold answer explicitly
            cmd = [
                sys.executable, "-m", "scripts.eval_hallucination",
                "--bundle_json", bundle_path,
                "--validated_json", validated_path,
                "--edges_json", edges_path,               # for TF decode
                "--gold_answer", answer,                  # <- provide gold
                "--nli_model", args.hallucination_nli_model,
                "--tau_entail", str(args.hallucination_tau_entail),
                "--tau_contra", str(args.hallucination_tau_contra),
                "--batch_size", str(args.hallucination_batch_size),
                "--gen_max_new_tokens", str(args.gen_max_new_tokens),
                "--out", halluc_path
            ]
            # fallback generation if TF decode not present
            if args.model_name:
                cmd.extend(["--gen_model", args.model_name])
                if args.load_in_4bit: cmd.append("--gen_4bit")
            run(cmd)
            hall = safe_load_json(halluc_path) or {}
            HRI_g = (hall.get("HRI_generated") or {}).get("HRI")
            HRI_y = (hall.get("HRI_gold") or {}).get("HRI")
            generated = hall.get("generated")

            # For backward-compatible columns, also expose masses from generated-view if present
            mg = hall.get("HRI_generated") or {}
            support_mass = mg.get("support_mass")
            contra_mass  = mg.get("contradiction_mass")
            unlabeled    = mg.get("unlabeled_mass")
            total_mass   = mg.get("total_mass")

            # 6) RARS
            cmd = [
                sys.executable, "-m", "scripts.compute_rars",
                "--bundle_json", bundle_path,
                "--edges_json", edges_path,
                "--validated_json", validated_path,
                "--model_name", args.model_name,
                "--layers", args.layers,
                "--out", rars_path
            ]
            if args.load_in_4bit: cmd.insert(5, "--load_in_4bit")
            run(cmd)
            rars = safe_load_json(rars_path) or {}
            RARS_union     = rars.get("RARS_union")
            RARS_validated = rars.get("RARS_validated")
            gain_sum       = rars.get("gain_sum")
            union_sum      = rars.get("union_sum")
            total_val_mass = rars.get("total_validated_mass")

            # 7) Collect row
            rows.append({
                "id": qid,
                "question": question,
                "answer": answer,
                "generated": "" if generated is None else generated,
                # provenance metrics
                "precision_at_1": p_at_1,
                "precision_at_3": p_at_3,
                "ndcg_at_3": ndcg3,
                "average_precision": ap,
                "auprc": auprc,
                # hallucination (generated-view)
                "HRI_generated": HRI_g,
                "HRI_gold": HRI_y,
                "support_mass_gen": support_mass,
                "contradiction_mass_gen": contra_mass,
                "unlabeled_mass_gen": unlabeled,
                "total_mass_gen": total_mass,
                # RARS
                "gain_sum": gain_sum,
                "union_sum": union_sum,
                "total_validated_mass": total_val_mass,
                "RARS_union": RARS_union,
                "RARS_validated": RARS_validated,
            })

    # ---- Write CSV ----
    outp = Path(args.out_csv)
    outp.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fieldnames = list(rows[0].keys())
    else:
        fieldnames = [
            "id","question","answer","generated",
            "precision_at_1","precision_at_3","ndcg_at_3","average_precision","auprc",
            "HRI_generated","HRI_gold","support_mass_gen","contradiction_mass_gen","unlabeled_mass_gen","total_mass_gen",
            "gain_sum","union_sum","total_validated_mass","RARS_union","RARS_validated"
        ]
    with open(outp, "w", newline="", encoding="utf-8") as cf:
        w = csv.DictWriter(cf, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[OK] wrote {outp} with {len(rows)} rows.")

if __name__ == "__main__":
    main()
