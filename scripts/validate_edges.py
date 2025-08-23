#!/usr/bin/env python3
"""
Validate RAAG edges via activation patching on Llama-3–style HF models,
aligned with teacher-forced scoring over generated answer tokens.

Method (per candidate edge: retrieval span [s,e) -> answer token at abs pos t):
  1) Build the evaluation sequence:
       - Prefer edges_json["tf_input_ids"] (teacher-forced [prompt || Y]).
       - Else fall back to bundle_json["input_ids"] (prompt only).
  2) Compute base log p(y_t) for each answer position t using logits at t-1.
  3) For each chosen layer L, register a self-attention forward pre-hook that
     zeros the attention input hidden_states over [s:e) (clamped to prompt_len).
  4) Recompute log p(y_t); keep the max Δ over layers.
  5) (Optional) Restrict validation to answer tokens that match --answer_string.
  6) Accept the edge if Δ >= epsilon; record best layer and validated weight.

Usage (example):
  python -m scripts.validate_edges \
    --bundle_json outputs/nq2_bundle.json \
    --edges_json outputs/nq2_edges.json \
    --model_name meta-llama/Meta-Llama-3-8B-Instruct \
    --load_in_4bit \
    --layers last12 \
    --epsilon 0.05 \
    --answer_string "Paris" \
    --verbose \
    --out outputs/nq2_validated.json
"""

import argparse
import json
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------- Model loading ---------------------------- #

def load_lm(model_name: str, load_in_4bit: bool):
    """
    Load tokenizer + model (optionally 4-bit). Sets pad_token to eos if missing.
    """
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    qcfg = None
    if load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
            qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        except Exception:
            qcfg = None

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
    """
    Return log-softmax over vocabulary for each position. Shape: [B, T, V]
    """
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    return torch.log_softmax(out.logits, dim=-1)

# ---------------------------- Layer selection ---------------------------- #

def _get_layers_module(model):
    """
    Try common paths to the decoder block list across Llama/Qwen variants.
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        return model.model.decoder.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise AttributeError("Could not locate decoder layers on model.")

def parse_layers(spec: str, num_layers: int) -> List[int]:
    s = (spec or "").strip().lower()
    if s == "all":
        out = list(range(num_layers))
    elif s.startswith("last") and s[4:].isdigit():
        n = int(s[4:])
        out = list(range(max(0, num_layers - n), num_layers))
    else:
        out = []
        for tok in s.split(","):
            tok = tok.strip()
            if not tok: continue
            try:
                i = int(tok)
                if 0 <= i < num_layers:
                    out.append(i)
            except ValueError:
                pass
        out = sorted(set(out))
    if not out:
        out = list(range(max(0, num_layers - 12), num_layers))  # default: last12
    return out

# ---------------------------- Hooks & helpers ---------------------------- #

def _make_attn_prehook(s_eff: int, e_eff: int):
    """
    Return a kwargs-aware pre-hook for Llama-style self_attn that zeros
    hidden_states[:, s_eff:e_eff, :] safely. Works with positional or kw args.
    """
    def pre_hook(module, args, kwargs):
        # kwargs path (modern HF)
        if kwargs and "hidden_states" in kwargs and kwargs["hidden_states"] is not None:
            hs = kwargs["hidden_states"]
            B, T, H = hs.shape
            s = max(0, min(T, s_eff))
            e = max(0, min(T, e_eff))
            if s < e:
                hs = hs.clone()
                hs[:, s:e, :] = 0
                kwargs["hidden_states"] = hs
            return args, kwargs

        # positional path (fallback)
        if args and args[0] is not None:
            hs = args[0]
            B, T, H = hs.shape
            s = max(0, min(T, s_eff))
            e = max(0, min(T, e_eff))
            if s < e:
                hs = hs.clone()
                hs[:, s:e, :] = 0
                args = list(args); args[0] = hs; args = tuple(args)
            return args, kwargs

        return args, kwargs
    return pre_hook

def _logp_at_positions(lp: torch.Tensor, ids: torch.Tensor, positions: List[int]) -> List[float]:
    """
    Teacher-forced: log p(token at pos t) is taken from logits at t-1.
    lp: [1, T, V], ids: [1, T]
    """
    outs: List[float] = []
    seq = ids[0]
    for t in positions:
        if t <= 0:
            outs.append(float("nan"))
        else:
            tok_id = seq[t].item()
            outs.append(lp[0, t - 1, tok_id].item())
    return outs

def _normalize_token(s: str) -> str:
    return "".join(ch for ch in s.strip() if ch.isalnum()).lower()

def _select_entity_positions(tok: AutoTokenizer, tf_ids_list: List[int], prompt_len: int, answer_string: str) -> List[int]:
    target = [_normalize_token(w) for w in answer_string.split() if w.strip()]
    if not target:
        return list(range(prompt_len, len(tf_ids_list)))
    sel: List[int] = []
    for pos in range(prompt_len, len(tf_ids_list)):
        txt = tok.decode([tf_ids_list[pos]], skip_special_tokens=True)
        if _normalize_token(txt) in target:
            sel.append(pos)
    return sel or list(range(prompt_len, len(tf_ids_list)))

# ---------------------------- Validation core ---------------------------- #

def validate_edges(
    tok,
    model,
    bundle: Dict,
    edges_blob: Dict,
    layer_ids: List[int],
    epsilon: float,
    answer_string: str = "",
    verbose: bool = False
) -> Dict[str, List[Dict]]:
    """
    Validate candidate edges by per-layer span ablation at self-attn input.
    Input edges are keyed by absolute answer token positions (teacher-forced).
    Returns: { str(t) : [ {span, chunk_global_id, weight, best_layer}, ... ] }
    """
    device = model.device

    # Prefer teacher-forced ids if present
    if "tf_input_ids" in edges_blob and isinstance(edges_blob["tf_input_ids"], list):
        eval_ids_list = edges_blob["tf_input_ids"]
        prompt_len = int(edges_blob.get("prompt_len", len(bundle["input_ids"])))
    else:
        eval_ids_list = bundle["input_ids"]
        prompt_len = len(bundle["input_ids"])

    eval_ids = torch.tensor([eval_ids_list], dtype=torch.long, device=device)
    attn_mask = torch.ones_like(eval_ids)

    # Base log-probs
    base_lp = logprobs(model, eval_ids, attn_mask)  # [1, T, V]
    T = eval_ids.shape[1]

    edges = edges_blob.get("edges", {})
    # Candidate targets present in scored edges
    candidate_positions = sorted(int(t) for t in edges.keys() if 1 <= int(t) < T)

    # Optional entity focus
    if answer_string.strip():
        entity_positions = _select_entity_positions(tok, eval_ids_list, prompt_len, answer_string)
        target_positions = [t for t in candidate_positions if t in set(entity_positions)]
        if verbose:
            print(f"[validate] focusing on {len(target_positions)} entity positions (of {len(candidate_positions)} candidates)")
        if not target_positions:
            target_positions = candidate_positions
    else:
        target_positions = candidate_positions

    if verbose:
        print(f"[validate] T={T} prompt_len={prompt_len} targets={len(target_positions)} layers={layer_ids}")

    validated: Dict[str, List[Dict]] = {str(t): [] for t in target_positions}

    # Layers
    layers_mod = _get_layers_module(model)
    num_layers = len(layers_mod)
    layer_ids = [L for L in layer_ids if 0 <= L < num_layers]

    # Precompute base log p(y_t) for all targets
    base_logp_targets = _logp_at_positions(base_lp, eval_ids, target_positions)

    # For each target and candidate span, search best validating layer
    for idx_t, t in enumerate(target_positions):
        base_lp_t = base_logp_targets[idx_t]
        if not (base_lp_t == base_lp_t):  # NaN
            continue

        cands = edges.get(str(t), [])
        if verbose:
            print(f"[validate] t={t} #cands={len(cands)}")
        if not cands:
            continue

        for cand in cands:
            s_raw, e_raw = int(cand["span"][0]), int(cand["span"][1])
            # Clamp to PROMPT so we never ablate generated tokens
            s_eff = max(0, min(prompt_len, s_raw))
            e_eff = max(0, min(prompt_len, e_raw))
            if s_eff >= e_eff:
                if verbose:
                    print(f"  - skip span [{s_raw},{e_raw}) -> [{s_eff},{e_eff})")
                continue

            best_delta = 0.0
            best_layer = None

            for L in layer_ids:
                hook = layers_mod[L].self_attn.register_forward_pre_hook(
                    _make_attn_prehook(s_eff, e_eff),
                    with_kwargs=True
                )
                lp2 = logprobs(model, eval_ids, attn_mask)
                hook.remove()

                masked_logp_t = _logp_at_positions(lp2, eval_ids, [t])[0]
                if masked_logp_t == masked_logp_t:  # not NaN
                    delta = base_lp_t - masked_logp_t
                    if delta > best_delta:
                        best_delta = delta
                        best_layer = int(L)

            if best_delta >= epsilon:
                validated[str(t)].append({
                    "span": [s_eff, e_eff],
                    "chunk_global_id": int(cand.get("chunk_global_id", -1)),
                    "weight": float(best_delta),
                    "best_layer": best_layer
                })

        validated[str(t)].sort(key=lambda r: r["weight"], reverse=True)

    return validated

# ---------------------------- CLI ---------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle_json", required=True, help="Output of make_prompt.py")
    ap.add_argument("--edges_json", required=True, help="Edges from score_edges (teacher-forced Δlog p)")
    ap.add_argument("--model_name", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--layers", default="last12", help='Layer spec: "lastN", "all", or "i,j,k" (0-based)')
    ap.add_argument("--epsilon", type=float, default=0.05, help="Δlog p threshold to accept edge")
    ap.add_argument("--answer_string", type=str, default="", help="If set, restrict validation to tokens matching these entity words")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", required=True, help="Path to write validated_edges.json")
    args = ap.parse_args()

    tok, model = load_lm(args.model_name, args.load_in_4bit)
    bundle = json.loads(open(args.bundle_json, "r", encoding="utf-8").read())
    edges_blob = json.loads(open(args.edges_json, "r", encoding="utf-8").read())

    # Resolve layer indices
    layers_mod = _get_layers_module(model)
    num_layers = len(layers_mod)
    layer_ids = parse_layers(args.layers, num_layers)

    validated = validate_edges(
        tok=tok,
        model=model,
        bundle=bundle,
        edges_blob=edges_blob,
        layer_ids=layer_ids,
        epsilon=args.epsilon,
        answer_string=args.answer_string,
        verbose=args.verbose
    )

    out = {
        "question": edges_blob.get("question"),
        "validated_edges": validated,
        "retrieved": edges_blob.get("retrieved", []),
        "config": {
            "model_name": args.model_name,
            "layers": layer_ids,
            "epsilon": args.epsilon,
            "prompt_len": edges_blob.get("prompt_len", len(bundle["input_ids"]))
        }
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[OK] wrote {args.out}")

if __name__ == "__main__":
    main()
