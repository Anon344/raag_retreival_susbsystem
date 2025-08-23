#!/usr/bin/env python3
"""
Convert a vis-network RAAG HTML export into a publication-ready PNG
with layer-highlighted edges and a bipartite layout.

- Parses `nodes = new vis.DataSet([...])` and `edges = new vis.DataSet([...])`
  directly from the HTML you provided.
- Identifies group/chunk nodes by id prefix "G:" and token nodes by "Y:".
- Infers NLI labels (support/neutral/contradiction/mixed) from each group
  node's title "(...)" or falls back to color.
- Extracts per-edge Δ log p and per-layer attributions from the edge title,
  colors each edge by its dominant layer, and sets linewidth ∝ Δ log p.
- Produces a clean bipartite figure with legends.

Usage:
  python html_graph_to_png.py \
    --html_in outputs/toy_raag.html \
    --png_out outputs/toy_raag_layers.png \
    --token_topm 3 --min_edge 0.05 \
    --dpi 300 --fig_w 16 --fig_h 9
"""
import argparse, json, re, math
from collections import defaultdict
from typing import List, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

# --------- helpers ---------
COLORS = {
    "token": "#2ca25f",
    "group_support": "#2ca25f",
    "group_neutral": "#3182bd",
    "group_contra": "#e6550d",
    "group_mixed": "#9467bd",
    "outline": "#222222",
}

def color_for_group(label: str) -> str:
    return {
        "support": COLORS["group_support"],
        "neutral": COLORS["group_neutral"],
        "contradiction": COLORS["group_contra"],
        "mixed": COLORS["group_mixed"],
    }.get(label, COLORS["group_neutral"])

def extract_dataset_arrays(html_text: str) -> Tuple[List[Dict], List[Dict]]:
    # capture JSON inside new vis.DataSet([...])
    nodes_m = re.search(r"nodes\s*=\s*new\s+vis\.DataSet\(\s*(\[[\s\S]*?\])\s*\);", html_text)
    edges_m = re.search(r"edges\s*=\s*new\s+vis\.DataSet\(\s*(\[[\s\S]*?\])\s*\);", html_text)
    if not nodes_m or not edges_m:
        raise RuntimeError("Could not find nodes/edges vis.DataSet arrays in HTML.")
    nodes_json = nodes_m.group(1)
    edges_json = edges_m.group(1)
    # normalize JS booleans/null if present (here they look JSON-valid already)
    nodes = json.loads(nodes_json)
    edges = json.loads(edges_json)
    return nodes, edges

def strip_html(s: str) -> str:
    s = re.sub(r"<br\s*/?>", "\n", s or "", flags=re.I)
    s = re.sub(r"<[^>]+>", "", s)
    return s

def infer_group_label_from_title(title: str, fallback_color: str) -> str:
    t = strip_html(title or "")
    m = re.search(r"\((support|neutral|contradiction|mixed)\)", t, flags=re.I)
    if m:
        return m.group(1).lower()
    # fallback: map known colors → labels
    col = (fallback_color or "").lower()
    if col == COLORS["group_support"].lower(): return "support"
    if col == COLORS["group_contra"].lower():  return "contradiction"
    if col == COLORS["group_mixed"].lower():   return "mixed"
    return "neutral"

def parse_edge_metrics(title: str, value_fallback: float) -> Tuple[float, Dict[int, float]]:
    """
    Returns (delta_logp, per_layer_dict). If Δ not found, uses 'value' fallback scaled.
    """
    t = strip_html(title or "")
    # Δ log p: <number>
    delta = None
    m = re.search(r"Δ\s*log\s*p\s*:\s*([0-9]*\.?[0-9]+)", t)
    if m:
        try:
            delta = float(m.group(1))
        except Exception:
            delta = None
    # Per-layer lines: L<idx>: <num>
    per = {}
    for L, v in re.findall(r"[Ll]\s*([0-9]+)\s*:\s*([0-9]*\.?[0-9]+)", t):
        try:
            per[int(L)] = float(v)
        except Exception:
            pass
    if delta is None:
        # fall back to vis 'value' scaled modestly
        delta = float(value_fallback) / 10.0
    return float(delta), per

def bipartite_positions(n_left: int, n_right: int, y_margin: float=0.08):
    def lin(n):
        if n <= 1: return np.array([0.5])
        return np.linspace(y_margin, 1.0-y_margin, n)
    yl = lin(n_left); yr = lin(n_right)
    left  = {i: (0.05, float(1-yl[i])) for i in range(n_left)}
    right = {j: (0.95, float(1-yr[j])) for j in range(n_right)}
    return left, right

def shorten(s: str, max_chars: int=14) -> str:
    s = (s or "").replace("\n"," ")
    return s if len(s) <= max_chars else s[:max_chars-1] + "…"

# --------- main render ---------
def render_png_from_html(html_in: str, png_out: str,
                         token_topm: int = 3, min_edge: float = 0.0,
                         dpi: int = 300, fig_w: float = 16.0, fig_h: float = 9.0):
    text = open(html_in, "r", encoding="utf-8").read()
    nodes, edges = extract_dataset_arrays(text)

    # Split nodes
    groups = []   # list of dict: {id,label,title,color,size}
    tokens = []   # list of dict: {id,label,title}
    for n in nodes:
        nid = str(n.get("id",""))
        if nid.startswith("LEG_"):
            continue
        if nid.startswith("G:"):
            label = infer_group_label_from_title(n.get("title","") or n.get("label",""), n.get("color",""))
            groups.append({
                "id": nid,
                "label": label,
                "title": strip_html(n.get("title","")),
                "color": n.get("color", COLORS["group_neutral"]),
                "size": float(n.get("size", 12.0))
            })
        elif nid.startswith("Y:"):
            tokens.append({
                "id": nid,
                "label": n.get("label",""),
                "title": strip_html(n.get("title","")),
            })

    if not groups or not tokens:
        fig, ax = plt.subplots(figsize=(8,3), dpi=dpi)
        ax.text(0.5, 0.5, "No group/token nodes found", ha="center", va="center")
        ax.axis("off")
        fig.savefig(png_out, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"[WARN] empty graph; wrote {png_out}")
        return

    # Parse edges G: -> Y:
    gy_edges = []  # ((g_idx, t_idx), delta, per_layer)
    id2g = {g["id"]: i for i, g in enumerate(groups)}
    id2t = {t["id"]: j for j, t in enumerate(tokens)}
    for e in edges:
        u = str(e.get("from","")); v = str(e.get("to",""))
        if not (u.startswith("G:") and v.startswith("Y:")):
            continue
        if u not in id2g or v not in id2t:
            continue
        delta, per = parse_edge_metrics(e.get("title",""), e.get("value", 1))
        if delta < float(min_edge):
            continue
        gy_edges.append(((id2g[u], id2t[v]), delta, per))

    # Top-m by token
    if token_topm and token_topm > 0:
        bucket = defaultdict(list)
        for (idx_pair, d, per) in gy_edges:
            _, t_idx = idx_pair
            bucket[t_idx].append((idx_pair, d, per))
        kept = []
        for t_idx, lst in bucket.items():
            lst.sort(key=lambda x: -x[1])
            kept.extend(lst[:token_topm])
        gy_edges = kept

    if not gy_edges:
        fig, ax = plt.subplots(figsize=(8,3), dpi=dpi)
        ax.text(0.5, 0.5, "No edges after filtering", ha="center", va="center")
        ax.axis("off")
        fig.savefig(png_out, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"[WARN] no edges; wrote {png_out}")
        return

    # Mass per group for sizing
    mass = np.zeros(len(groups), dtype=float)
    for ((gi, _tj), d, _per) in gy_edges:
        mass[gi] += d

    # layer palette from tab20
    layer_vals = sorted({L for (_gt, _d, per) in gy_edges for L in per.keys()}) or [0]
    cmap = plt.get_cmap("tab20")
    layer_color = {L: cmap(i % 20) for i, L in enumerate(layer_vals)}

    # layout
    # order groups by mass desc (neater)
    g_order = np.argsort(-mass)
    t_order = np.arange(len(tokens))
    left_pos, right_pos = bipartite_positions(len(groups), len(tokens))
    gpos = {int(i): left_pos[k] for k, i in enumerate(g_order)}
    tpos = {int(j): right_pos[k] for k, j in enumerate(t_order)}

    # figure & axes
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    ax = fig.add_axes([0.02, 0.02, 0.78, 0.96]); ax.set_axis_off()

    # edges
    w_max = max(d for (_gt, d, _per) in gy_edges) if gy_edges else 1.0
    for ((gi, tj), d, per) in sorted(gy_edges, key=lambda x: -x[1]):
        dom_layer = max(per.items(), key=lambda kv: kv[1])[0] if per else layer_vals[0]
        color = layer_color.get(dom_layer, "#999999")
        lw = 0.5 + 5.0 * (d / (w_max + 1e-8))
        x1, y1 = gpos[int(gi)]
        x2, y2 = tpos[int(tj)]
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw, alpha=0.9, solid_capstyle="round", zorder=1)

    # nodes: groups
    for k, gi in enumerate(g_order):
        g = groups[int(gi)]
        x, y = gpos[int(gi)]
        radius = 0.018 + 0.012 * math.log1p(max(1.0, mass[int(gi)]))
        facecolor = color_for_group(g["label"])
        circ = mpatches.Circle((x, y), radius=radius, facecolor=facecolor,
                               edgecolor=COLORS["outline"], linewidth=0.8, zorder=2)
        ax.add_patch(circ)
        lbl = g["id"].split(":",1)[-1]
        ax.text(x, y - radius - 0.012, f"C{lbl}" if lbl.isdigit() else lbl,
                ha="center", va="top", fontsize=9, color="#111111")

    # nodes: tokens
    for tj in t_order:
        t = tokens[int(tj)]
        x, y = tpos[int(tj)]
        circ = mpatches.Circle((x, y), radius=0.016, facecolor=COLORS["token"],
                               edgecolor=COLORS["outline"], linewidth=0.8, zorder=3)
        ax.add_patch(circ)
        ax.text(x, y - 0.020, shorten(t["label"], 14), ha="center", va="top",
                fontsize=9, color="#111111")

    # legends on right
    leg_ax = fig.add_axes([0.82, 0.05, 0.16, 0.90]); leg_ax.set_axis_off()

    node_handles = [
        mpatches.Patch(facecolor=COLORS["token"], edgecolor=COLORS["outline"], label="Token"),
        mpatches.Patch(facecolor=COLORS["group_support"], edgecolor=COLORS["outline"], label="Support chunk"),
        mpatches.Patch(facecolor=COLORS["group_neutral"], edgecolor=COLORS["outline"], label="Neutral chunk"),
        mpatches.Patch(facecolor=COLORS["group_contra"], edgecolor=COLORS["outline"], label="Contradiction"),
        mpatches.Patch(facecolor=COLORS["group_mixed"], edgecolor=COLORS["outline"], label="Mixed"),
    ]
    leg_ax.legend(handles=node_handles, loc="upper left", fontsize=9, frameon=True, title="Nodes", title_fontsize=10)

    layer_handles = [mlines.Line2D([], [], color=layer_color[L], lw=3, label=f"L{L}") for L in layer_vals]
    leg_ax.legend(handles=layer_handles, loc="upper left", bbox_to_anchor=(0.0, 0.5),
                  fontsize=9, frameon=True, title="Edges by layer", title_fontsize=10)

    fig.savefig(png_out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] wrote {png_out} | groups={len(groups)} tokens={len(tokens)} edges={len(gy_edges)}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--html_in", required=True)
    ap.add_argument("--png_out", default="outputs/raag_from_html.png")
    ap.add_argument("--token_topm", type=int, default=3, help="Top-m edges per token (0=keep all)")
    ap.add_argument("--min_edge", type=float, default=0.0, help="Minimum Δ log p to draw")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--fig_w", type=float, default=16.0)
    ap.add_argument("--fig_h", type=float, default=9.0)
    args = ap.parse_args()

    render_png_from_html(
        html_in=args.html_in,
        png_out=args.png_out,
        token_topm=args.token_topm,
        min_edge=args.min_edge,
        dpi=args.dpi,
        fig_w=args.fig_w,
        fig_h=args.fig_h,
    )

if __name__ == "__main__":
    main()
