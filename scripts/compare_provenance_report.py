#!/usr/bin/env python3
import argparse, csv, math, statistics, json
from pathlib import Path

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

def agg(metric_list):
    vals = [v for v in metric_list if v is not None and math.isfinite(v)]
    if not vals: return {"mean": None, "std": None, "n": 0}
    return {"mean": sum(vals)/len(vals), "std": statistics.pstdev(vals) if len(vals)>1 else 0.0, "n": len(vals)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True, help="baseline_compare.csv from run_baselines_over_outputs.py")
    ap.add_argument("--out_md", default="outputs/baseline_compare.md")
    args = ap.parse_args()

    rows = []
    with open(args.in_csv, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            rows.append(r)

    # Collect per-metric arrays
    keys = [
        ("p@1", "RAAG_p@1", "RET_p@1", "ATT_p@1"),
        ("p@3", "RAAG_p@3", "RET_p@3", "ATT_p@3"),
        ("nDCG@3", "RAAG_nDCG@3", "RET_nDCG@3", "ATT_nDCG@3"),
        ("AP", "RAAG_AP", "RET_AP", "ATT_AP"),
        ("AUPRC", "RAAG_AUPRC", "RET_AUPRC", "ATT_AUPRC"),
    ]

    report_lines = []
    report_lines.append("# Baseline Comparison (Provenance Metrics)\n")
    report_lines.append(f"Source: `{args.in_csv}`\n")
    report_lines.append("| Metric | RAAG (mean±std) | Retrieval (mean±std) | Attention (mean±std) | RAAG−RET Δ | RAAG−ATT Δ |")
    report_lines.append("|---|---:|---:|---:|---:|---:|")

    for name, k_r, k_ret, k_att in keys:
        raag = [safe_float(r.get(k_r)) for r in rows]
        ret  = [safe_float(r.get(k_ret)) for r in rows]
        att  = [safe_float(r.get(k_att)) for r in rows]
        aR, aT, aA = agg(raag), agg(ret), agg(att)

        # pairwise deltas for shared rows
        deltas_ret = []
        deltas_att = []
        for r in rows:
            xr = safe_float(r.get(k_r))
            xt = safe_float(r.get(k_ret))
            xa = safe_float(r.get(k_att))
            if xr is not None and xt is not None:
                deltas_ret.append(xr - xt)
            if xr is not None and xa is not None:
                deltas_att.append(xr - xa)
        dR = agg(deltas_ret)
        dA = agg(deltas_att)

        def fmt(a):
            if a["mean"] is None: return "–"
            return f"{a['mean']:.3f}±{a['std']:.3f} (n={a['n']})"

        report_lines.append(
            f"| {name} | {fmt(aR)} | {fmt(aT)} | {fmt(aA)} | "
            f"{'–' if dR['mean'] is None else f'{dR['mean']:.3f}'} | "
            f"{'–' if dA['mean'] is None else f'{dA['mean']:.3f}'} |"
        )

    # wins/losses @P@1 for quick headline
    wins_ret = losses_ret = ties_ret = 0
    wins_att = losses_att = ties_att = 0
    for r in rows:
        ra = safe_float(r.get("RAAG_p@1"))
        rt = safe_float(r.get("RET_p@1"))
        at = safe_float(r.get("ATT_p@1"))
        if ra is not None and rt is not None:
            if ra > rt: wins_ret += 1
            elif ra < rt: losses_ret += 1
            else: ties_ret += 1
        if ra is not None and at is not None:
            if ra > at: wins_att += 1
            elif ra < at: losses_att += 1
            else: ties_att += 1

    report_lines.append("\n## Win/Loss (P@1)\n")
    report_lines.append(f"- RAAG vs Retrieval: **{wins_ret} / {losses_ret} / {ties_ret}** (win/loss/tie)")
    report_lines.append(f"- RAAG vs Attention: **{wins_att} / {losses_att} / {ties_att}** (win/loss/tie)\n")

    outp = Path(args.out_md)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"[OK] wrote {outp}")

if __name__ == "__main__":
    main()
