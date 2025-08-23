#!/usr/bin/env python3
import argparse, json, os, math, glob
from pathlib import Path

def mean_ci(xs):
    xs = [x for x in xs if x is not None and (not isinstance(x, float) or (x == x))]
    n = len(xs)
    if n == 0: return None, None
    m = sum(xs)/n
    if n == 1: return m, float('nan')
    var = sum((x - m)**2 for x in xs)/(n-1)
    se = math.sqrt(var/n)
    return m, 1.96*se

def fmt_pm(m, ci):
    if m is None: return "—"
    if ci is None or (isinstance(ci, float) and ci != ci):  # NaN
        return f"{m:.3f} $\\pm$ —"
    return f"{m:.3f} $\\pm$ {ci:.3f}"

def load_json(p):
    try:
        return json.loads(open(p, "r", encoding="utf-8").read())
    except Exception:
        return None

def collect_all(roots):
    hri_gen, hri_gold = [], []
    rars_union, rars_validated = [], []

    for root in roots:
        # HRI
        for hpath in glob.glob(os.path.join(root, "**", "*_hallucination.json"), recursive=True):
            H = load_json(hpath)
            if not H: continue
            # New NLI-style fields
            if isinstance(H.get("HRI_generated"), dict):
                val = H["HRI_generated"].get("HRI", None)
                if val is not None: hri_gen.append(val)
            elif "HRI" in H:  # legacy single-HRI fallback
                hri_gen.append(H["HRI"])
            if isinstance(H.get("HRI_gold"), dict):
                val = H["HRI_gold"].get("HRI", None)
                if val is not None: hri_gold.append(val)

        # RARS
        for rpath in glob.glob(os.path.join(root, "**", "*_rars.json"), recursive=True):
            R = load_json(rpath)
            if not R: continue
            if "RARS_union" in R and R["RARS_union"] is not None:
                rars_union.append(R["RARS_union"])
            if "RARS_validated" in R and R["RARS_validated"] is not None:
                rars_validated.append(R["RARS_validated"])

    return hri_gen, hri_gold, rars_union, rars_validated

def write_combined_table(hri_gen, hri_gold, rars_union, rars_validated, out_tex):
    mg, cig = mean_ci(hri_gen)
    my, ciy = mean_ci(hri_gold)
    mu, ciu = mean_ci(rars_union)
    mv, civ = mean_ci(rars_validated)

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("  \\caption{hallucination and retrieval attribution robustness (All runs): mean $\\pm$ 95\\% CI. HRI is computed for the generated claim (HRI\\textsubscript{gen}) and, when available, for the gold claim (HRI\\textsubscript{gold}). RARS reports union ablation and validated mass variants.}")
    lines.append("  \\label{tab:a3-hri-rars}")
    lines.append("  \\centering")
    lines.append("  \\small")
    lines.append("  \\begin{tabular}{lcccc}")
    lines.append("    \\toprule")
    lines.append("    & \\textbf{HRI\\textsubscript{gen}} & \\textbf{HRI\\textsubscript{gold}} & \\textbf{RARS\\textsubscript{union}} & \\textbf{RARS\\textsubscript{validated}} \\\\")
    lines.append("    \\midrule")
    lines.append(f"    All & {fmt_pm(mg, cig)} & {fmt_pm(my, ciy)} & {fmt_pm(mu, ciu)} & {fmt_pm(mv, civ)} \\\\")
    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("\\end{table}")

    Path(out_tex).parent.mkdir(parents=True, exist_ok=True)
    open(out_tex, "w", encoding="utf-8").write("\n".join(lines))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True,
                    help="Directories to scan (e.g., outputs/natural_questions outputs/hotpot_qa)")
    ap.add_argument("--out_tex", default="outputs/TableA3_HRI_RARS.tex")
    args = ap.parse_args()

    hri_gen, hri_gold, r_union, r_valid = collect_all(args.roots)
    write_combined_table(hri_gen, hri_gold, r_union, r_valid, args.out_tex)
    print(f"[OK] wrote {args.out_tex}")

if __name__ == "__main__":
    main()
