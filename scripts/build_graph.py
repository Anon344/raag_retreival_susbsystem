#!/usr/bin/env python3
"""
Interactive RAAG graph with adjacency tensor + (optional) SVD communities,
per-layer edge attributions, NLI-aware coloring, and edge filtering.

Default behavior is *chunk-level nodes* (no merging) and *all answer tokens*.

Usage (chunk-level, all tokens, no pruning):
  python -m scripts.build_graph_interactive \
    --bundle_json outputs/nq1_bundle.json \
    --validated_json outputs/nq1_validated.json \
    --edges_json outputs/nq1_edges.json \
    --hallucination_json outputs/nq1_hri.json \
    --group_mode none --include_all_answer_tokens \
    --token_topm 0 --min_edge 0 \
    --html_out outputs/nq1_raag.html \
    --json_out outputs/nq1_raag.json \
    --model_name meta-llama/Llama-3.2-3B-Instruct
"""
import argparse, json, html
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from pyvis.network import Network
from transformers import AutoTokenizer

# ---------- I/O ----------
def jload(p): return json.load(open(p, "r", encoding="utf-8"))
def jdump(o, p): json.dump(o, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

# ---------- Colors/legend ----------
COLORS = {
    "token": "#2ca25f",
    "group_support": "#2ca25f",
    "group_neutral": "#3182bd",
    "group_contra": "#e6550d",
    "group_mixed": "#9467bd",
}
def color_for_group(label: str) -> str:
    return {
        "support": COLORS["group_support"],
        "neutral": COLORS["group_neutral"],
        "contradiction": COLORS["group_contra"],
        "mixed": COLORS["group_mixed"],
    }.get(label, COLORS["group_neutral"])

# ---------- Aggregation: adjacency tensor ----------
def discover_layers(validated: Dict[str, List[Dict]], config_layers: Optional[List[int]]) -> List[int]:
    if config_layers and len(config_layers) > 0:
        return sorted({int(x) for x in config_layers})
    s = set()
    for edges in validated.values():
        for e in edges:
            if "best_layer" in e and e["best_layer"] is not None:
                s.add(int(e["best_layer"]))
            elif "layer" in e and e["layer"] is not None:
                s.add(int(e["layer"]))
    return sorted(s) or [0]

def tensor_from_validated(validated: Dict[str, List[Dict]], layers: List[int]):
    layer2idx = {ell:i for i, ell in enumerate(layers)}
    wmap = defaultdict(float)
    chunks, toks = set(), set()
    for t_str, edges in validated.items():
        t = int(t_str); toks.add(t)
        for e in edges:
            cid = int(e.get("chunk_global_id", -1))
            if cid < 0: continue
            w = float(e.get("weight", 0.0))
            if w <= 0: continue
            if "best_layer" in e and e["best_layer"] is not None:
                ell = int(e["best_layer"])
            elif "layer" in e and e["layer"] is not None:
                ell = int(e["layer"])
            else:
                ell = layers[0]
            if ell not in layer2idx:
                layer2idx[ell] = len(layer2idx); layers.append(ell)
            L = layer2idx[ell]
            wmap[(L, cid, t)] += w
            chunks.add(cid)
    chunk_ids = sorted(chunks)
    token_positions = sorted(toks)
    return wmap, chunk_ids, token_positions, layer2idx, layers

def dense_tensor(wmap, chunk_ids, token_positions, layer_count):
    idx_c = {cid:i for i, cid in enumerate(chunk_ids)}
    idx_t = {t:j for j, t in enumerate(token_positions)}
    W = np.zeros((layer_count, len(chunk_ids), len(token_positions)), dtype=np.float32)
    for (L, cid, t), w in wmap.items():
        i = idx_c.get(cid); j = idx_t.get(t)
        if i is None or j is None or not (0 <= L < layer_count): continue
        W[L, i, j] += float(w)
    return W, idx_c, idx_t

# ---------- SVD utilities (optional) ----------
def union_find_groups(X: np.ndarray, ids: List[int], thresh: float) -> List[List[int]]:
    if len(ids) == 0 or X.ndim != 2 or X.shape[0] != len(ids) or X.shape[1] == 0:
        return [[i] for i in ids]
    Xn = normalize(X, axis=1)
    parent = list(range(len(ids)))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    def union(a,b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[rb] = ra
    C = len(ids)
    for i in range(C):
        xi = Xn[i]
        for j in range(i+1, C):
            cos = float(np.dot(xi, Xn[j]))
            if cos >= thresh: union(i,j)
    buckets = defaultdict(list)
    for i in range(C):
        buckets[find(i)].append(i)
    return [[ids[k] for k in v] for v in buckets.values()]

def chunk_groups_from_W(W_total: np.ndarray, chunk_ids: List[int], svd_rank: int, merge_thresh: float) -> List[List[int]]:
    if W_total.size == 0 or len(chunk_ids) == 0: return [[cid] for cid in chunk_ids]
    r = max(1, min(svd_rank, min(W_total.shape)))
    svd = TruncatedSVD(n_components=r, random_state=0)
    U = svd.fit_transform(W_total)  # [C, r]
    return union_find_groups(U, chunk_ids, merge_thresh)

def token_embeddings_from_tensor(W: np.ndarray, svd_rank: int) -> np.ndarray:
    L, C, T = W.shape
    if T == 0: return np.zeros((0, 0), dtype=np.float32)
    A = np.zeros((T, T), dtype=np.float32)
    for Lidx in range(L):
        M = W[Lidx]       # C × T
        A += M.T @ M      # T × T
    r = max(1, min(svd_rank, T))
    svd = TruncatedSVD(n_components=r, random_state=0)
    Z = svd.fit_transform(A)  # [T, r]
    return Z

# ---------- NLI label integration ----------
def choose_group_label(members: List[int], support: set, contra: set) -> str:
    cnt_s = sum(1 for cid in members if cid in support)
    cnt_c = sum(1 for cid in members if cid in contra)
    if cnt_s > 0 and cnt_c == 0: return "support"
    if cnt_c > 0 and cnt_s == 0: return "contradiction"
    if cnt_s == 0 and cnt_c == 0: return "neutral"
    return "mixed"

# ---------- Token labels ----------
def decode_token_label(tok: AutoTokenizer, position: int,
                       prompt_ids: List[int],
                       tf_input_ids: Optional[List[int]],
                       prompt_len: Optional[int]) -> str:
    if isinstance(tf_input_ids, list) and 0 <= position < len(tf_input_ids):
        s = tok.decode([tf_input_ids[position]], skip_special_tokens=True).strip()
        return s or f"y@{position}"
    if 0 <= position < len(prompt_ids):
        s = tok.decode([prompt_ids[position]], skip_special_tokens=True).strip()
        return s or f"y@{position}"
    return f"y@{position}"

# ---------- Core builder ----------
def build_graph(bundle, validated_blob, edges_blob, halluc_blob,
                svd_rank: int, merge_thresh: float, group_mode: str,
                include_all_answer_tokens: bool,
                chunk_max: int, token_max: int, token_topm: int, min_edge: float,
                html_out: str, json_out: str, model_name: str):

    validated = validated_blob.get("validated_edges", {})
    config_layers = validated_blob.get("config", {}).get("layers", [])

    if not validated:
        net = Network(height="780px", width="100%", directed=True, notebook=False, cdn_resources="remote")
        net.add_node("NOTE", label="No validated edges", color="#e6550d")
        net.write_html(html_out, open_browser=False, notebook=False)
        jdump({"error": "no validated edges"}, json_out)
        print(f"[WARN] no validated edges; wrote {html_out} and {json_out}")
        return

    # ---- Token universe: union(validated targets, answer positions, TF continuation) ----
    token_positions_all = {int(t) for t in validated.keys()}
    if include_all_answer_tokens and isinstance(edges_blob, dict):
        if isinstance(edges_blob.get("answer_token_positions"), list):
            token_positions_all |= set(int(x) for x in edges_blob["answer_token_positions"])
        tf = edges_blob.get("tf_input_ids"); pl = edges_blob.get("prompt_len")
        if isinstance(tf, list) and isinstance(pl, int) and pl < len(tf):
            token_positions_all |= set(range(pl, len(tf)))

    token_positions_all = sorted(token_positions_all)

    # ---- Build tensor W[L,C,T] on full token set ----
    layers = discover_layers(validated, config_layers)
    wmap, chunk_ids_all, tokens_from_validated, layer2idx, layers = tensor_from_validated(validated, layers)

    # Make sure tokens in union set exist in index maps
    # We'll create dense tensor using the full list we computed:
    W, idx_c, idx_t = dense_tensor(
        wmap,
        chunk_ids_all,
        token_positions_all,
        layer_count=len(layers)
    )

    # ---- Optional pruning (defaults effectively "off") ----
    mass_per_chunk = W.sum(axis=(0,2)) if W.size else np.array([])
    mass_per_token = W.sum(axis=(0,1)) if W.size else np.array([])

    keep_chunk_order = np.argsort(-mass_per_chunk)
    keep_token_order = np.argsort(-mass_per_token)

    # keep everything by default (or cap if user asked)
    keep_chunks = [chunk_ids_all[i] for i in keep_chunk_order[:min(chunk_max or len(keep_chunk_order), len(keep_chunk_order))]]
    keep_tokens = [token_positions_all[j] for j in keep_token_order[:min(token_max or len(keep_token_order), len(keep_token_order))]]

    kc = [idx_c[cid] for cid in keep_chunks]
    kt = [idx_t[t] for t in keep_tokens]
    Wk = W[:, kc, :][:, :, kt]   # [L, Ck, Tk]
    Lc, Ck, Tk = Wk.shape

    # ---- Token embeddings (layout cue) ----
    token_Z = token_embeddings_from_tensor(Wk, svd_rank=max(2, svd_rank))

    # ---- Groups (by default: one chunk per group) ----
    if group_mode.lower() == "svd":
        W_total = Wk.sum(axis=0)  # [Ck, Tk]
        groups = chunk_groups_from_W(W_total, keep_chunks, svd_rank=max(2, svd_rank), merge_thresh=merge_thresh)
    else:  # "none" -> identity groups
        groups = [[cid] for cid in keep_chunks]

    cid2gid = {}
    for gid, members in enumerate(groups):
        for cid in members: cid2gid[cid] = gid

    # ---- Build group->token edges, per-layer breakdown ----
    gw = defaultdict(float)                      # total weight
    gl = defaultdict(lambda: defaultdict(float)) # per-layer
    for Lidx in range(Lc):
        for ic, cid in enumerate(keep_chunks):
            gid = cid2gid[cid]
            for jt, t in enumerate(keep_tokens):
                w = float(Wk[Lidx, ic, jt])
                if w <= 0: continue
                gw[(gid, t)] += w
                layer = layers[Lidx]
                gl[(gid, t)][layer] += w

    # ---- Filter edges ----
    edges_all = [((gid, t), w) for (gid, t), w in gw.items() if w >= float(min_edge)]
    if token_topm and token_topm > 0:
        per_token = defaultdict(list)
        for (gid_t, w) in edges_all:
            gid, t = gid_t
            per_token[t].append((gid_t, w))
        edges = []
        for t, lst in per_token.items():
            lst.sort(key=lambda x: -x[1])
            edges.extend(lst[:token_topm])
    else:
        edges = edges_all

    # ---- Labels (NLI) ----
    support_ids = set(halluc_blob.get("gen_support_cids", []) or halluc_blob.get("gold_support_cids", []) or [])
    contra_ids  = set(halluc_blob.get("gen_contradiction_cids", []) or halluc_blob.get("gold_contradiction_cids", []) or [])

    cid2text = {int(ch["chunk_global_id"]): ch.get("text","") for ch in bundle.get("chunk_spans", [])}

    group_nodes = []
    for gid, members in enumerate(groups):
        label = choose_group_label(members, support_ids, contra_ids)
        full_texts = [cid2text.get(cid, "") for cid in members]
        preview = " | ".join([txt.replace("\n"," ")[:160] for txt in full_texts[:2]])
        title = html.escape(("\n\n---\n").join(full_texts[:6]))
        out_mass = float(sum(gw.get((gid, t), 0.0) for t in keep_tokens))
        group_nodes.append({
            "gid": gid, "members": members, "label": label,
            "preview": preview, "title": title, "mass": out_mass
        })

    # ---- Token nodes (show ALL kept tokens; some may be isolated) ----
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    prompt_ids = bundle.get("input_ids", [])
    tf_input_ids = edges_blob.get("tf_input_ids") if isinstance(edges_blob, dict) else None
    prompt_len   = edges_blob.get("prompt_len")   if isinstance(edges_blob, dict) else None
    token_nodes = [{
        "t": t,
        "label": decode_token_label(tok, t, prompt_ids, tf_input_ids, prompt_len),
        "title": f"Token @{t}"
    } for t in keep_tokens]

    # ---- PyVis graph ----
    net = Network(height="850px", width="100%", directed=True, notebook=False, cdn_resources="remote")
    net.set_options(json.dumps({
        "physics": {"barnesHut": {"gravitationalConstant": -9000, "centralGravity": 0.25, "springLength": 160, "springConstant": 0.012}},
        "nodes": {"shape": "dot", "scaling": {"min": 10, "max": 42}},
        "edges": {"smooth": True, "arrows": {"to": {"enabled": True}}}
    }))

    # Legend
    net.add_node("LEG_T", label="Token", color=COLORS["token"], x=-1000, y=-520, physics=False, size=18)
    net.add_node("LEG_S", label="Support chunk", color=COLORS["group_support"], x=-1000, y=-470, physics=False, size=18)
    net.add_node("LEG_N", label="Neutral chunk", color=COLORS["group_neutral"], x=-1000, y=-420, physics=False, size=18)
    net.add_node("LEG_C", label="Contradiction", color=COLORS["group_contra"], x=-1000, y=-370, physics=False, size=18)
    net.add_node("LEG_M", label="Mixed", color=COLORS["group_mixed"], x=-1000, y=-320, physics=False, size=18)

    # Group (chunk) nodes
    for gn in group_nodes:
        size = min(42, 12 + 6*np.log1p(max(1.0, gn["mass"])))
        net.add_node(
            f"G:{gn['gid']}",
            label=f"C{gn['members'][0]}" if len(gn["members"]) == 1 else f"G{gn['gid']}",
            title=f"<b>{'Chunk' if len(gn['members'])==1 else 'Group'} {gn['gid']}</b> ({gn['label']})<br/>{gn['preview']}<hr/>{gn['title']}",
            color=color_for_group(gn["label"]),
            size=float(size),
        )

    # Token nodes
    for tn in token_nodes:
        net.add_node(f"Y:{tn['t']}", label=tn["label"] or f"y@{tn['t']}", title=tn["title"], color=COLORS["token"], size=14)

    # Edges (hover shows total + per-layer)
    w_max = max((w for (_e, w) in edges), default=1.0)
    for ((gid, t), w) in edges:
        by_layer = gl.get((gid, t), {})
        layer_parts = [f"L{ell}: {by_layer[ell]:.3f}" for ell in sorted(by_layer.keys())]
        layer_txt = "<br/>".join(layer_parts) if layer_parts else "n/a"
        title = f"<b>Δ log p</b>: {w:.4f}<br/><b>Per-layer</b><br/>{layer_txt}"
        width = 1.0 + 9.0*(w/(w_max+1e-8))
        net.add_edge(f"G:{gid}", f"Y:{t}", value=max(1, int(width*2)), title=title)

    net.write_html(html_out, open_browser=False, notebook=False)

    # JSON audit
    audit = {
        "question": bundle.get("question"),
        "layers": layers,
        "shapes": {"W": [int(x) for x in W.shape], "W_kept": [int(x) for x in Wk.shape]},
        "kept": {"chunks": keep_chunks, "tokens": keep_tokens},
        "groups": [{
            "gid": gn["gid"], "members": gn["members"], "label": gn["label"],
            "mass": gn["mass"], "preview": gn["preview"]
        } for gn in group_nodes],
        "token_nodes": [{"t": t["t"], "label": t["label"]} for t in token_nodes],
        "edges": [{
            "src_gid": int(gid), "t": int(t), "weight": float(w),
            "per_layer": {str(ell): float(gl[(gid,t)].get(ell, 0.0)) for ell in layers}
        } for ((gid, t), w) in sorted(edges, key=lambda kv: -kv[1])],
        "meta": {
            "svd_rank": svd_rank, "merge_thresh": merge_thresh,
            "group_mode": group_mode,
            "include_all_answer_tokens": bool(include_all_answer_tokens),
            "chunk_max": chunk_max, "token_max": token_max,
            "token_topm": token_topm, "min_edge": min_edge
        }
    }
    jdump(audit, json_out)
    print(f"[OK] wrote {html_out} and {json_out} | groups={len(groups)} tokens={len(keep_tokens)} edges={len(edges)}")

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle_json", required=True)
    ap.add_argument("--validated_json", required=True)
    ap.add_argument("--edges_json", default=None)
    ap.add_argument("--hallucination_json", default=None, help="Optional NLI labels for coloring (gen_* or gold_*)")
    ap.add_argument("--svd_rank", type=int, default=8)
    ap.add_argument("--merge_thresh", type=float, default=0.90)
    ap.add_argument("--group_mode", choices=["none","svd"], default="none")
    ap.add_argument("--include_all_answer_tokens", action="store_true")
    ap.add_argument("--chunk_max", type=int, default=0, help="0 means no cap")
    ap.add_argument("--token_max", type=int, default=0, help="0 means no cap")
    ap.add_argument("--token_topm", type=int, default=0, help="0 means keep all")
    ap.add_argument("--min_edge", type=float, default=0.0)
    ap.add_argument("--html_out", default="outputs/raag_graph.html")
    ap.add_argument("--json_out", default="outputs/raag_graph.json")
    ap.add_argument("--model_name", default="meta-llama/Llama-3.2-3B-Instruct")
    args = ap.parse_args()

    bundle = jload(args.bundle_json)
    validated_blob = jload(args.validated_json)
    edges_blob = jload(args.edges_json) if args.edges_json else {}
    halluc_blob = jload(args.hallucination_json) if args.hallucination_json else {}

    build_graph(bundle, validated_blob, edges_blob, halluc_blob,
                svd_rank=args.svd_rank, merge_thresh=args.merge_thresh, group_mode=args.group_mode,
                include_all_answer_tokens=bool(args.include_all_answer_tokens),
                chunk_max=args.chunk_max, token_max=args.token_max,
                token_topm=args.token_topm, min_edge=args.min_edge,
                html_out=args.html_out, json_out=args.json_out, model_name=args.model_name)

if __name__ == "__main__":
    main()
