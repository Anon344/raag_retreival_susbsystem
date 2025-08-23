#!/usr/bin/env python3
import argparse, os, json, glob, math, csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt

# ------------------- helpers -------------------
DATASET_LABEL = {"hotpot_qa":"HotpotQA","natural_questions":"NaturalQuestions"}

def jload(p):
    with open(p,"r",encoding="utf-8") as f: return json.load(f)

def wcsv(rows, header, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path,"w",newline="",encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header); w.writeheader(); w.writerows(rows)

def mean_ci95(xs: List[float]) -> Tuple[float,float]:
    clean=[]
    for x in xs:
        try:
            fx=float(x)
            if not (math.isnan(fx) or math.isinf(fx)): clean.append(fx)
        except: pass
    if not clean: return float("nan"), float("nan")
    m=float(np.mean(clean)); s=float(np.std(clean, ddof=1)) if len(clean)>1 else 0.0
    ci=1.96*s/max(1, math.sqrt(len(clean)))
    return m, ci

def latex_mean_ci(m,c): return f"{m:.3f} $\\pm$ {c:.3f}"

def discover_items_in_dir(dir_path: str, label: str):
    out=[]
    for bundle in glob.glob(os.path.join(dir_path,"*_bundle.json")):
        stem=os.path.basename(bundle).replace("_bundle.json","")
        base=dir_path
        edges=os.path.join(base,f"{stem}_edges.json")
        valid=os.path.join(base,f"{stem}_validated.json")
        prov =os.path.join(base,f"{stem}_prov.json")
        if os.path.exists(edges) and os.path.exists(valid) and os.path.exists(prov):
            out.append(dict(stem=stem, base=base, bundle=bundle, edges=edges,
                            validated=valid, prov=prov, group=label))
    return sorted(out, key=lambda d:d["stem"])

def discover_all(root: str, subdirs: List[str]):
    items=[]
    for sd in subdirs:
        d=os.path.join(root, sd)
        if os.path.isdir(d):
            items.extend(discover_items_in_dir(d, DATASET_LABEL.get(sd,sd)))
        else:
            print(f"[WARN] missing dir: {d}")
    return items

# ------------------- A1 -------------------
A1_KEYS=["precision_at_1","precision_at_3","ndcg_at_3","average_precision","auprc"]

def summarize_provenance(items, out_csv):
    groups={"All":{k:[] for k in A1_KEYS}}
    for it in items:
        g=it["group"]; groups.setdefault(g,{k:[] for k in A1_KEYS})
        prov=jload(it["prov"]).get("metrics",{})
        for k in A1_KEYS:
            groups[g][k].append(prov.get(k))
            groups["All"][k].append(prov.get(k))
    rows=[]
    for g in ["All"]+sorted([x for x in groups if x!="All"]):
        row={"group":g}
        for k in A1_KEYS:
            m,ci=mean_ci95(groups[g][k]); row[k]=m; row[k+"_ci95"]=ci
        rows.append(row)
    header=["group"]+sum([[k,k+"_ci95"] for k in A1_KEYS],[])
    wcsv(rows, header, out_csv)
    print("\nTable A1 (mean ± 95% CI):")
    for r in rows:
        print(r["group"]+" & "+
              latex_mean_ci(r["precision_at_1"],r["precision_at_1_ci95"])+" & "+
              latex_mean_ci(r["precision_at_3"],r["precision_at_3_ci95"])+" & "+
              latex_mean_ci(r["ndcg_at_3"],r["ndcg_at_3_ci95"])+" & "+
              latex_mean_ci(r["average_precision"],r["average_precision_ci95"])+" & "+
              latex_mean_ci(r["auprc"],r["auprc_ci95"])+" \\\\")
    return rows

# ------------------- utilities for A2 -------------------
def aggregate_raag(validated_json: str) -> Dict[int,float]:
    v=jload(validated_json).get("validated_edges",{})
    mass={}
    for _t,edges in v.items():
        for e in edges:
            cid=int(e.get("chunk_global_id",-1)); w=float(e.get("weight",0.0))
            if cid<0: continue
            mass[cid]=mass.get(cid,0.0)+w
    return mass

def spearman(a: Dict[int,float], b: Dict[int,float]) -> Optional[float]:
    common=sorted(set(a.keys()) & set(b.keys()))
    if len(common)<2: return None
    ax=np.array([a[i] for i in common])
    bx=np.array([b[i] for i in common])
    # rank with average ties
    def rankavg(x):
        order=np.argsort(x, kind="mergesort")  # stable
        ranks=np.empty_like(order,dtype=float)
        ranks[order]=np.arange(1,len(x)+1)
        # ties -> average
        vals={}
        for idx,val in enumerate(x):
            vals.setdefault(val,[]).append(idx)
        for v,idxs in vals.items():
            if len(idxs)>1:
                r=np.mean([ranks[i] for i in idxs])
                for i in idxs: ranks[i]=r
        return ranks
    ra=rankavg(ax); rb=rankavg(bx)
    if np.std(ra)==0 or np.std(rb)==0: return None
    rho=float(np.corrcoef(ra,rb)[0,1])
    return rho

def topk_ids(d: Dict[int,float], k:int=3) -> List[int]:
    return [cid for cid,_ in sorted(d.items(), key=lambda kv:(-kv[1],kv[0]))[:k]]

def jaccard_at3(a: Dict[int,float], b: Dict[int,float]) -> Optional[float]:
    A=set(topk_ids(a,3)); B=set(topk_ids(b,3))
    if not A and not B: return None
    return len(A&B)/len(A|B) if len(A|B)>0 else None

def argmax_match(a: Dict[int,float], b: Dict[int,float]) -> Optional[float]:
    if not a or not b: return None
    return 1.0 if topk_ids(a,1)==topk_ids(b,1) else 0.0

# ---- baseline readers (robust to variants) ----
def read_retrieval_baseline(base_dir: str, stem: str) -> Optional[Dict[int,float]]:
    cand = [
        os.path.join(base_dir, f"{stem}_baseline_retrieval.json"),
        os.path.join(base_dir, "baseline_retrieval", f"{stem}.json"),
        os.path.join(base_dir, "baseline_retrieval", f"{stem}_baseline_retrieval.json"),
        os.path.join(base_dir, f"{stem}_baseline_retrieval.csv"),
        os.path.join(base_dir, "baseline_retrieval", f"{stem}.csv"),
    ]
    path=None
    for p in cand:
        if os.path.exists(p): path=p; break
    if not path: return None
    # JSON
    if path.endswith(".json"):
        js=jload(path)
        # accept a few shapes
        if "score_by_chunk" in js and isinstance(js["score_by_chunk"], dict):
            d=js["score_by_chunk"]
        elif "rank_by_chunk" in js and isinstance(js["rank_by_chunk"], dict):
            # convert rank -> score (higher better)
            d={int(k): -float(v) for k,v in js["rank_by_chunk"].items()}
        elif "scores" in js and isinstance(js["scores"], dict):
            d=js["scores"]
        elif "ranked_cids" in js and isinstance(js["ranked_cids"], list):
            # highest first
            lst=js["ranked_cids"]; d={int(cid): float(len(lst)-i) for i,cid in enumerate(lst)}
        else:
            return None
        return {int(k): float(v) for k,v in d.items()}
    # CSV
    d={}
    with open(path,"r",encoding="utf-8") as f:
        r=csv.DictReader(f)
        # expect columns like cid,score or cid,rank
        for row in r:
            if "cid" not in row: continue
            cid=int(row["cid"])
            if "score" in row and row["score"]!="":
                d[cid]=float(row["score"])
            elif "rank" in row and row["rank"]!="":
                d[cid]=-float(row["rank"])
    return d or None

def read_shapley_doc(base_dir: str, stem: str) -> Optional[Dict[int,float]]:
    cand = [
        os.path.join(base_dir, f"{stem}_shapley_doc.json"),
        os.path.join(base_dir, "shapley", f"{stem}_doc.json"),
    ]
    path=None
    for p in cand:
        if os.path.exists(p): path=p; break
    if not path: return None
    js=jload(path)
    if "phi_by_chunk" in js and isinstance(js["phi_by_chunk"], dict):
        return {int(k): float(v) for k,v in js["phi_by_chunk"].items()}
    return None

# ------------------- A2 -------------------
A2_KEYS=[
    "rho_retrieval","rho_shapley_doc",
    "top1_match_retrieval","top1_match_shapley_doc",
    "jaccard_top3_retrieval","jaccard_top3_shapley_doc"
]

def summarize_baselines_from_files(items, out_csv):
    rows_item=[]
    for it in items:
        base=it["base"]; stem=it["stem"]; grp=it["group"]
        raag = aggregate_raag(it["validated"])
        ret  = read_retrieval_baseline(base, stem)
        shp  = read_shapley_doc(base, stem)
        if ret is None and shp is None:  # skip if no baselines available
            continue
        row={"id":stem,"group":grp}
        # Spearman
        row["rho_retrieval"]        = spearman(raag, ret) if ret else None
        row["rho_shapley_doc"]      = spearman(raag, shp) if shp else None
        # Top-1 match
        row["top1_match_retrieval"] = argmax_match(raag, ret) if ret else None
        row["top1_match_shapley_doc"]=argmax_match(raag, shp) if shp else None
        # Jaccard@3
        row["jaccard_top3_retrieval"]= jaccard_at3(raag, ret) if ret else None
        row["jaccard_top3_shapley_doc"]= jaccard_at3(raag, shp) if shp else None
        rows_item.append(row)

    if not rows_item:
        print("[WARN] No items with available baselines were found; skipping A2.")
        return []

    # write per-item detail (optional but handy)
    wcsv(rows_item, ["id","group"]+A2_KEYS, out_csv.replace(".csv","_per_item.csv"))

    # aggregate
    groups={"All":{k:[] for k in A2_KEYS}}
    for r in rows_item:
        g=r["group"]; groups.setdefault(g,{k:[] for k in A2_KEYS})
        for k in A2_KEYS:
            groups[g][k].append(r.get(k))
            groups["All"][k].append(r.get(k))

    rows=[]
    for g in ["All"]+sorted([x for x in groups if x!="All"]):
        row={"group":g}
        for k in A2_KEYS:
            m,ci=mean_ci95(groups[g][k]); row[k]=m; row[k+"_ci95"]=ci
        rows.append(row)

    header=["group"]+sum([[k,k+"_ci95"] for k in A2_KEYS],[])
    wcsv(rows, header, out_csv)

    print("\nTable A2 (mean ± 95% CI):")
    for r in rows:
        print(r["group"]+" & "+
              latex_mean_ci(r["rho_retrieval"],r["rho_retrieval_ci95"])+" & "+
              latex_mean_ci(r["rho_shapley_doc"],r["rho_shapley_doc_ci95"])+" & "+
              latex_mean_ci(r["top1_match_retrieval"],r["top1_match_retrieval_ci95"])+" & "+
              latex_mean_ci(r["top1_match_shapley_doc"],r["top1_match_shapley_doc_ci95"])+" & "+
              latex_mean_ci(r["jaccard_top3_retrieval"],r["jaccard_top3_retrieval_ci95"])+" & "+
              latex_mean_ci(r["jaccard_top3_shapley_doc"],r["jaccard_top3_shapley_doc_ci95"])+" \\\\")
    return rows

# ------------------- A3 -------------------
def collect_halluc_rars(items):
    stats={"All":{"H":[], "Ru":[], "Rv":[]}}
    for it in items:
        g=it["group"]; stats.setdefault(g,{"H":[], "Ru":[], "Rv":[]})
        base=it["base"]; stem=it["stem"]
        hall=os.path.join(base,f"{stem}_hallucination.json")
        rars=os.path.join(base,f"{stem}_rars.json")
        H=Ru=Rv=None
        if os.path.exists(hall):
            hj=jload(hall)
            Hg=hj.get("HRI_generated",{})
            if isinstance(Hg,dict) and "HRI" in Hg and Hg["HRI"] is not None: H=Hg["HRI"]
            elif "HRI" in hj: H=hj["HRI"]
        if os.path.exists(rars):
            rj=jload(rars); Ru=rj.get("RARS_union",None); Rv=rj.get("RARS_validated",None)
        stats[g]["H"].append(H); stats[g]["Ru"].append(Ru); stats[g]["Rv"].append(Rv)
        stats["All"]["H"].append(H); stats["All"]["Ru"].append(Ru); stats["All"]["Rv"].append(Rv)
    return stats

def plot_halluc_rars(stats, out_png):
    groups=["All"]
    K=len(groups); labels=["HRI_gen","RARS_union","RARS_validated"]
    means=np.zeros((K,3)); cis=np.zeros((K,3))
    for i,g in enumerate(groups):
        m,c=mean_ci95([x for x in stats[g]["H"] if x is not None]);  means[i,0]=m; cis[i,0]=c
        m,c=mean_ci95([x for x in stats[g]["Ru"] if x is not None]); means[i,1]=m; cis[i,1]=c
        m,c=mean_ci95([x for x in stats[g]["Rv"] if x is not None]); means[i,2]=m; cis[i,2]=c
    plt.figure(figsize=(9,4.8)); x=np.arange(K); w=0.25
    for j in range(3):
        plt.bar(x+(j-1)*w, means[:,j], width=w, yerr=cis[:,j], capsize=4, label=labels[j])
    plt.xticks(x, groups); plt.ylabel("Score (mean ± 95% CI)"); plt.title("Hallucination (HRI) and RARS")
    plt.legend(); Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(out_png, dpi=180); plt.close()
    print(f"[OK] wrote {out_png}")

# ------------------- CLI -------------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs")
    ap.add_argument("--subdirs", nargs="+", default=["natural_questions","hotpot_qa"])
    ap.add_argument("--out_table_a1", default="tables/table_A1_provenance.csv")
    ap.add_argument("--out_table_a2", default="tables/table_A2_baselines.csv")
    ap.add_argument("--out_fig_a3",  default="figures/fig_A3_halluc_rars.png")
    args=ap.parse_args()

    items=discover_all(args.root, args.subdirs)
    if not items:
        print(f"[WARN] No completed RAAG runs found under {args.root}/{args.subdirs}")
        return
    print(f"[INFO] Found {len(items)} completed RAAG items.")

    # A1
    summarize_provenance(items, args.out_table_a1)
    # A2 (from actual baseline files)
    summarize_baselines_from_files(items, args.out_table_a2)
    # A3
    stats=collect_halluc_rars(items); plot_halluc_rars(stats, args.out_fig_a3)

if __name__=="__main__":
    main()
