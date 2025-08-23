#!/usr/bin/env python3
"""
Shapley baselines for RAAG experiments.

Two modes:
  --mode doc    : each retrieved CHUNK (chunk_global_id) is a player
  --mode token  : subdivide each chunk span into smaller windows as players (capped by --max_players)

Coalition value v(S):
  1) Build teacher-forced sequence from edges_json["tf_input_ids"] (full context).
  2) Compute question-only teacher-forced sequence (strip docs) once -> base_qonly.
  3) For coalition S, mask the COMPLEMENT spans (players not in S) across selected layers
     and compute base_ctx_S; value(S) = sum_t max(0, base_ctx_S[t] - base_qonly[t]).
  4) Shapley(i) ≈ average_{permutations} [ v(P ∪ {i}) - v(P) ], where P is i's predecessors in a random permutation.

Outputs:
  {
    "mode": "doc" | "token",
    "permutations": int,
    "players": [...],              # list of player descriptors
    "phi_by_chunk": {cid: phi},    # summed per-chunk shapley (token mode folds windows back to chunk ids)
    "phi_by_player": {pid: phi}    # raw player scores (pid is index into players list)
  }

Usage:
  python -m scripts.baseline_shapley \
    --bundle_json outputs/nq1_bundle.json \
    --edges_json outputs/nq1_edges.json \
    --model_name meta-llama/Llama-3.2-3B-Instruct \
    --load_in_4bit \
    --layers last8 \
    --mode doc \
    --permutations 32 \
    --gen_max_new_tokens 32 \
    --out outputs/nq1_shapley_doc.json
"""
import argparse, json, random, re, math
from typing import Dict, List, Tuple, Optional, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------- I/O ---------------- #

def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

# ------------- Model utils ----------- #

def load_lm(model_name: str, load_in_4bit: bool):
    from transformers import BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16) if load_in_4bit else None
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=qcfg,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model.eval()
    return tok, model

@torch.no_grad()
def logprobs(model, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    return torch.log_softmax(out.logits, dim=-1)  # [B,T,V]

def logp_for_positions(lp: torch.Tensor, ids: torch.Tensor, positions: List[int]) -> List[float]:
    seq = ids[0]
    vals: List[float] = []
    for t in positions:
        if t <= 0: vals.append(float("nan"))
        else:
            tok_id = seq[t].item()
            vals.append(lp[0, t - 1, tok_id].item())
    return vals

# ---------- Layers, hooks ---------- #

def _get_layers_module(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"): return model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        return model.model.decoder.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"): return model.transformer.h
    raise AttributeError("Could not locate decoder layers on model.")

def parse_layers(spec: str, num_layers: int) -> List[int]:
    s = (spec or "").strip().lower()
    if s == "all":
        out = list(range(num_layers))
    elif s.startswith("last") and s[4:].isdigit():
        n = int(s[4:])
        out = list(range(max(0, num_layers - n), num_layers))
    elif s == "last4":
        out = list(range(max(0, num_layers - 4), num_layers))
    elif s == "last8":
        out = list(range(max(0, num_layers - 8), num_layers))
    else:
        out = []
        for tok in s.split(","):
            tok = tok.strip()
            if tok.isdigit():
                i = int(tok)
                if 0 <= i < num_layers: out.append(i)
        out = sorted(set(out))
    if not out:
        out = list(range(max(0, num_layers - 4), num_layers))
    return out

def _make_attn_prehook(spans_eff: List[Tuple[int,int]]):
    """
    Zero hidden_states[:, s:e, :] for each span in spans_eff (already clamped).
    """
    def pre_hook(module, args, kwargs):
        if kwargs and "hidden_states" in kwargs and kwargs["hidden_states"] is not None:
            hs = kwargs["hidden_states"]; B, T, H = hs.shape
            hs = hs.clone()
            for s, e in spans_eff:
                s2 = max(0, min(T, s)); e2 = max(0, min(T, e))
                if s2 < e2: hs[:, s2:e2, :] = 0
            kwargs["hidden_states"] = hs
            return args, kwargs
        if args and args[0] is not None:
            hs = args[0]; B, T, H = hs.shape
            hs = hs.clone()
            for s, e in spans_eff:
                s2 = max(0, min(T, s)); e2 = max(0, min(T, e))
                if s2 < e2: hs[:, s2:e2, :] = 0
            args = list(args); args[0] = hs; args = tuple(args)
            return args, kwargs
        return args, kwargs
    return pre_hook

# --------- Prompt surgery (q-only) --------- #

_DOC_BLOCK = re.compile(r"\[DOC\s+\d+\]\s*.*?(?=\n\[DOC\s+\d+\]|\nQuestion:|\Z)", flags=re.S)

def strip_docs_from_prompt(prompt: str) -> str:
    no_docs = re.sub(_DOC_BLOCK, "", prompt)
    no_docs = re.sub(r"\n{3,}", "\n\n", no_docs).strip()
    return no_docs

# -------------- Players -------------- #

def build_doc_players(bundle: Dict, prompt_len: int) -> List[Dict[str, Any]]:
    """
    Each player is a CHUNK (chunk_global_id) with possibly multiple spans.
    """
    spans_by_cid: Dict[int, List[Tuple[int,int]]] = {}
    for ch in bundle.get("chunk_spans", []):
        cid = int(ch["chunk_global_id"])
        s = int(ch["tok_start_in_prompt"]); e = int(ch["tok_end_in_prompt"])
        s_eff = max(0, min(prompt_len, s)); e_eff = max(0, min(prompt_len, e))
        if s_eff < e_eff:
            spans_by_cid.setdefault(cid, []).append((s_eff, e_eff))
    players = []
    for cid, spans in spans_by_cid.items():
        players.append({"type":"chunk", "cid": cid, "spans": spans})
    return players

def build_token_players(bundle: Dict, prompt_len: int, max_players: int) -> List[Dict[str, Any]]:
    """
    Split each chunk span into windows to cap total players <= max_players.
    """
    raw = []
    for ch in bundle.get("chunk_spans", []):
        cid = int(ch["chunk_global_id"])
        s = int(ch["tok_start_in_prompt"]); e = int(ch["tok_end_in_prompt"])
        s_eff = max(0, min(prompt_len, s)); e_eff = max(0, min(prompt_len, e))
        if s_eff < e_eff:
            raw.append((cid, s_eff, e_eff))
    if not raw: return []

    total_len = sum(e - s for _, s, e in raw)
    if total_len <= 0: return []

    # decide window size
    target_players = max(1, max_players)
    avg_window = max(1, math.ceil(total_len / target_players))

    players = []
    for cid, s, e in raw:
        start = s
        while start < e:
            stop = min(e, start + avg_window)
            players.append({"type":"window", "cid": cid, "spans":[(start, stop)]})
            start = stop
    # cap if overshoot
    if len(players) > max_players:
        players = players[:max_players]
    return players

# ------------- Coalition value ------------- #

@torch.no_grad()
def compute_value_for_coalition(model, layers_mod, layer_ids,
                                tf_ids_ctx: torch.Tensor, mask_ctx: torch.Tensor,
                                base_qonly: List[float], ans_positions: List[int],
                                players: List[Dict], S: set) -> float:
    """
    Value(S) = sum_t max(0, logp_ctx_S[t] - base_qonly[t]),
    where ctx_S masks the complement players (NOT in S).
    """
    # union of spans to mask = complement players
    spans_to_mask: List[Tuple[int,int]] = []
    for pid, P in enumerate(players):
        if pid in S:  # keep these
            continue
        for s, e in P["spans"]:
            spans_to_mask.append((s, e))

    if spans_to_mask and layer_ids:
        delta_best = [0.0] * len(ans_positions)
        # compute masked ctx across layers, keep best per-token (max drop)
        for L in layer_ids:
            hook = layers_mod[L].self_attn.register_forward_pre_hook(_make_attn_prehook(spans_to_mask), with_kwargs=True)
            lp_masked = logprobs(model, tf_ids_ctx, mask_ctx)
            hook.remove()
            masked_ctx = logp_for_positions(lp_masked, tf_ids_ctx, ans_positions)
            # convert to actual ctx logp (not delta) for value
            # we can't recover unmasked base here cheaply; but we don't need it:
            # we want logp with only S active; masking complement gives that directly.
            if "acc_ctx" not in locals():
                acc_ctx = masked_ctx
            else:
                # keep max over layers per token (upper bound for ctx with S active)
                acc_ctx = [max(a, b) for a, b in zip(acc_ctx, masked_ctx)]
    else:
        # no masking -> full context active
        lp = logprobs(model, tf_ids_ctx, mask_ctx)
        acc_ctx = logp_for_positions(lp, tf_ids_ctx, ans_positions)

    gains = [max(0.0, (c - q)) for c, q in zip(acc_ctx, base_qonly)]
    return float(sum(gains))

# ------------- Shapley estimation ------------- #

def estimate_shapley(model, tok, bundle: Dict, edges: Dict,
                     layers: List[int], mode: str, permutations: int,
                     max_players: int) -> Dict:
    device = model.device
    layers_mod = _get_layers_module(model)

    if "tf_input_ids" not in edges or "prompt_len" not in edges:
        raise RuntimeError("edges_json must contain 'tf_input_ids' and 'prompt_len' (from score_edges).")
    tf_ids_ctx_list = edges["tf_input_ids"]
    prompt_len_ctx  = int(edges["prompt_len"])
    ans_positions   = list(range(prompt_len_ctx, len(tf_ids_ctx_list)))

    tf_ids_ctx = torch.tensor([tf_ids_ctx_list], dtype=torch.long, device=device)
    mask_ctx   = torch.ones_like(tf_ids_ctx)

    # question-only base logp
    prompt_qonly_text = strip_docs_from_prompt(bundle.get("prompt",""))
    ids_qonly_prompt  = tok(prompt_qonly_text, return_tensors="pt", add_special_tokens=False).input_ids[0].tolist()
    ans_ids           = tf_ids_ctx_list[prompt_len_ctx:]
    tf_ids_qonly_list = ids_qonly_prompt + ans_ids
    tf_ids_q          = torch.tensor([tf_ids_qonly_list], dtype=torch.long, device=device)
    mask_q            = torch.ones_like(tf_ids_q)
    base_q_lp         = logprobs(model, tf_ids_q, mask_q)
    base_qonly        = logp_for_positions(base_q_lp, tf_ids_q, list(range(len(ids_qonly_prompt), len(tf_ids_qonly_list))))

    # players
    if mode == "doc":
        players = build_doc_players(bundle, prompt_len_ctx)
    elif mode == "token":
        players = build_token_players(bundle, prompt_len_ctx, max_players=max_players)
    else:
        raise ValueError("--mode must be 'doc' or 'token'")
    if not players:
        return {"mode": mode, "permutations": permutations, "players": [], "phi_by_chunk": {}, "phi_by_player": {}}

    num_players = len(players)
    # initialize φ
    phi = [0.0] * num_players

    for _ in range(max(1, permutations)):
        order = list(range(num_players))
        random.shuffle(order)
        S = set()
        v_S = compute_value_for_coalition(model, layers_mod, layers, tf_ids_ctx, mask_ctx, base_qonly, ans_positions, players, S)
        for pid in order:
            S2 = set(S); S2.add(pid)
            v_S2 = compute_value_for_coalition(model, layers_mod, layers, tf_ids_ctx, mask_ctx, base_qonly, ans_positions, players, S2)
            phi[pid] += (v_S2 - v_S)
            S = S2
            v_S = v_S2

    # average
    phi = [p / max(1, permutations) for p in phi]

    # fold to chunk ids
    phi_by_player = {i: float(phi[i]) for i in range(num_players)}
    phi_by_chunk: Dict[int, float] = {}
    for i, P in enumerate(players):
        cid = int(P["cid"])
        phi_by_chunk[cid] = phi_by_chunk.get(cid, 0.0) + float(phi[i])

    return {
        "mode": mode,
        "permutations": int(permutations),
        "players": players,
        "phi_by_chunk": phi_by_chunk,
        "phi_by_player": phi_by_player
    }

# ------------------ CLI ------------------ #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle_json", required=True)
    ap.add_argument("--edges_json", required=True)
    ap.add_argument("--model_name", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--layers", default="last8")
    ap.add_argument("--mode", choices=["doc","token"], default="doc")
    ap.add_argument("--permutations", type=int, default=32)
    ap.add_argument("--max_players", type=int, default=256)      # token mode cap
    ap.add_argument("--gen_max_new_tokens", type=int, default=32) # unused here but kept for symmetry
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    random.seed(args.seed)

    bundle = load_json(args.bundle_json)
    edges  = load_json(args.edges_json)

    tok, model = load_lm(args.model_name, args.load_in_4bit)
    layer_ids = parse_layers(args.layers, len(_get_layers_module(model)))

    result = estimate_shapley(
        model=model, tok=tok, bundle=bundle, edges=edges,
        layers=layer_ids, mode=args.mode, permutations=args.permutations,
        max_players=args.max_players
    )
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # Friendly summary line (no args.config bug)
    nplayers = len(result.get("players", []))
    print(f"[OK] wrote {args.out}  |  mode={args.mode}  players={nplayers}  perms={args.permutations}")

if __name__ == "__main__":
    main()