#!/usr/bin/env python3
"""
Compute RARS for a single example.

- Builds teacher-forced sequences for:
    (a) full context prompt (question + docs) using edges_json["tf_input_ids"]
    (b) question-only prompt (docs stripped) + same answer tokens
- Computes context gain per answer token: g(t) = max(0, logp_ctx - logp_qonly)
- Computes union-ablation retrieval effect per token by masking the union of all retrieval spans
  at selected layers and taking the per-token max drop across layers: Δ∪(t)
- Returns:
    RARS_union     = sum Δ∪(t) / sum g(t)
    RARS_validated = total_validated_mass / sum g(t)

Usage:
  python -m scripts.compute_rars \
    --bundle_json outputs/nq1_bundle.json \
    --edges_json outputs/nq1_edges.json \
    --validated_json outputs/nq1_validated.json \
    --model_name meta-llama/Meta-Llama-3-8B-Instruct \
    --load_in_4bit \
    --layers last8 \
    --eps 1e-8 \
    --out outputs/nq1_rars.json
"""
import argparse, json, re
from typing import Dict, List, Optional, Tuple

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

# ---------- Layers, hooks, spans ---------- #

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
        # kwargs path
        if kwargs and "hidden_states" in kwargs and kwargs["hidden_states"] is not None:
            hs = kwargs["hidden_states"]
            B, T, H = hs.shape
            hs = hs.clone()
            for s, e in spans_eff:
                s2 = max(0, min(T, s)); e2 = max(0, min(T, e))
                if s2 < e2: hs[:, s2:e2, :] = 0
            kwargs["hidden_states"] = hs
            return args, kwargs
        # positional path
        if args and args[0] is not None:
            hs = args[0]
            B, T, H = hs.shape
            hs = hs.clone()
            for s, e in spans_eff:
                s2 = max(0, min(T, s)); e2 = max(0, min(T, e))
                if s2 < e2: hs[:, s2:e2, :] = 0
            args = list(args); args[0] = hs; args = tuple(args)
            return args, kwargs
        return args, kwargs
    return pre_hook

# --------- Prompt surgery (strip docs) --------- #

_DOC_BLOCK = re.compile(r"\[DOC\s+\d+\]\s*.*?(?=\n\[DOC\s+\d+\]|\nQuestion:|\Z)", flags=re.S)

def strip_docs_from_prompt(prompt: str) -> str:
    # keep the system prefix and the Question/Answer scaffold; remove [DOC i] blocks
    no_docs = re.sub(_DOC_BLOCK, "", prompt)
    # collapse extra blank lines
    no_docs = re.sub(r"\n{3,}", "\n\n", no_docs).strip()
    return no_docs

# ------------------ Main ------------------ #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle_json", required=True)
    ap.add_argument("--edges_json", required=True)
    ap.add_argument("--validated_json", required=True)
    ap.add_argument("--model_name", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--layers", default="last8")
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    bundle = load_json(args.bundle_json)
    edges  = load_json(args.edges_json)
    val    = load_json(args.validated_json)
    validated = val.get("validated_edges", {})

    tok, model = load_lm(args.model_name, args.load_in_4bit)
    device = model.device

    # --- Build teacher-forced sequences ---
    if "tf_input_ids" not in edges or "prompt_len" not in edges:
        raise RuntimeError("edges_json must contain 'tf_input_ids' and 'prompt_len' from score_edges.")
    tf_ids_ctx_list = edges["tf_input_ids"]
    prompt_len_ctx  = int(edges["prompt_len"])
    ans_positions_ctx = list(range(prompt_len_ctx, len(tf_ids_ctx_list)))

    # Question-only TF ids = tokenize stripped prompt, then append the same answer ids
    prompt_qonly_text = strip_docs_from_prompt(bundle.get("prompt",""))
    ids_qonly_prompt  = tok(prompt_qonly_text, return_tensors="pt", add_special_tokens=False).input_ids[0].tolist()
    ans_ids           = tf_ids_ctx_list[prompt_len_ctx:]  # reuse the same answer ids
    tf_ids_qonly_list = ids_qonly_prompt + ans_ids
    prompt_len_qonly  = len(ids_qonly_prompt)
    ans_positions_q   = list(range(prompt_len_qonly, len(tf_ids_qonly_list)))

    # --- Tensors ---
    tf_ids_ctx = torch.tensor([tf_ids_ctx_list], dtype=torch.long, device=device)
    tf_ids_q   = torch.tensor([tf_ids_qonly_list], dtype=torch.long, device=device)
    mask_ctx   = torch.ones_like(tf_ids_ctx)
    mask_q     = torch.ones_like(tf_ids_q)

    # --- Base logp (ctx & q-only) ---
    lp_ctx = logprobs(model, tf_ids_ctx, mask_ctx)
    lp_q   = logprobs(model, tf_ids_q,   mask_q)

    # align lengths: answer positions lists are same size
    if len(ans_positions_ctx) != len(ans_positions_q):
        raise RuntimeError("Answer token counts differ between ctx and q-only TF sequences.")

    base_ctx = logp_for_positions(lp_ctx, tf_ids_ctx, ans_positions_ctx)
    base_q   = logp_for_positions(lp_q,   tf_ids_q,   ans_positions_q)

    # context gain per token
    gains = [max(0.0, (c - q)) for c, q in zip(base_ctx, base_q)]
    gain_sum = sum(gains)

    # --- Union-ablation Δ across layers (ctx seq) ---
    # union of retrieval spans from bundle["chunk_spans"] clamped to prompt
    spans_union = []
    for ch in bundle.get("chunk_spans", []):
        s = int(ch["tok_start_in_prompt"]); e = int(ch["tok_end_in_prompt"])
        s_eff = max(0, min(prompt_len_ctx, s)); e_eff = max(0, min(prompt_len_ctx, e))
        if s_eff < e_eff:
            spans_union.append((s_eff, e_eff))

    layers_mod = _get_layers_module(model)
    layer_ids  = parse_layers(args.layers, len(layers_mod))

    # compute per-token max drop across layers
    delta_union = [0.0] * len(ans_positions_ctx)
    if spans_union and layer_ids:
        for L in layer_ids:
            hook = layers_mod[L].self_attn.register_forward_pre_hook(_make_attn_prehook(spans_union), with_kwargs=True)
            lp_masked = logprobs(model, tf_ids_ctx, mask_ctx)
            hook.remove()
            masked_ctx = logp_for_positions(lp_masked, tf_ids_ctx, ans_positions_ctx)
            for i in range(len(delta_union)):
                d = base_ctx[i] - masked_ctx[i]
                if d > delta_union[i]:
                    delta_union[i] = float(max(0.0, d))

    union_sum = sum(delta_union)

    # --- Validated mass (already computed in your pipeline) ---
    total_validated_mass = 0.0
    for _, lst in validated.items():
        for e in lst:
            total_validated_mass += float(e.get("weight", 0.0))

    # --- RARS ---
    if gain_sum <= args.eps:
        rars_union = None
        rars_validated = None
        note = "No positive context gain (answer likely came from parametric knowledge or retrieval hurt)."
    else:
        rars_union = union_sum / max(args.eps, gain_sum)
        rars_validated = total_validated_mass / max(args.eps, gain_sum)
        note = None

    out = {
        "question": val.get("question", bundle.get("question")),
        "prompt_len_ctx": prompt_len_ctx,
        "prompt_len_qonly": prompt_len_qonly,
        "gain_sum": gain_sum,
        "union_sum": union_sum,
        "total_validated_mass": total_validated_mass,
        "RARS_union": rars_union,
        "RARS_validated": rars_validated,
        "layers_used": layer_ids,
        "note": note
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[OK] wrote {args.out} | gain_sum={gain_sum:.4f} union_sum={union_sum:.4f} "
          f"RARS_union={'nan' if rars_union is None else f'{rars_union:.3f}'} "
          f"RARS_validated={'nan' if rars_validated is None else f'{rars_validated:.3f}'}")

if __name__ == "__main__":
    main()
