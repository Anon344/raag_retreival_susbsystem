#!/usr/bin/env python3
"""
RAAG scoring with activation patching over generated answer tokens,
with optional control subtraction and entity-token focusing.

This version ALSO emits:
  - generated_ids: the answer token IDs Y (greedy)
  - generated_text: decoded string of Y

Pipeline:
  1) Read prompt + spans from bundle_json (make_prompt.py).
  2) Greedy-decode answer tokens Y once (no sampling).
  3) Build teacher-forced sequence X = [prompt || Y].
  4) Base pass: compute log p(Y) from X via teacher forcing (use logits at t-1).
  5) For each retrieval span S in PROMPT:
       - For each chosen layer L, zero hidden_states[:, s:e, :] at that layer,
         recompute log p(Y), get Δ = base - patched per answer token.
       - Keep elementwise MAX Δ across selected layers (upper-bound causal effect).
       - (Optional) Subtract a CONTROL Δ computed on a random same-length prompt slice.
  6) (Optional) Restrict targets to answer tokens matching provided entity words.
  7) Emit edges keyed by absolute answer positions, plus generated_ids/text.

Usage:
  python -m scripts.score_edges \
    --bundle_json tmp_bundle.json \
    --model_name meta-llama/Meta-Llama-3-8B-Instruct \
    --load_in_4bit \
    --gen_max_new_tokens 64 \
    --layers last8 \
    --k_edges 8 \
    --length_norm \
    --control_subtract \
    --control_trials 12 \
    --answer_string "Alexander Fleming" \
    --verbose \
    --out outputs/edges_hf.json
"""
import argparse, json, random
from typing import List, Dict, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


# ---------------------------- Loading ---------------------------- #

def load_lm(model_name: str, load_in_4bit: bool):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16) if load_in_4bit else None
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", quantization_config=qcfg,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )
    model.eval()
    return tok, model

@torch.no_grad()
def logprobs(model, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    return torch.log_softmax(out.logits, dim=-1)

@torch.no_grad()
def greedy_decode_answer_ids(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, max_new_tokens: int) -> List[int]:
    gen = model.generate(
        input_ids=input_ids, attention_mask=attention_mask,
        max_new_tokens=max_new_tokens, do_sample=False, temperature=0.0,
        num_beams=1, return_dict_in_generate=True
    )
    seq = gen.sequences[0]
    prompt_len = input_ids.shape[1]
    y_ids = seq[prompt_len:].tolist()
    # Cut at EOS if present
    eos = model.generation_config.eos_token_id
    if isinstance(eos, int):
        if eos in y_ids: y_ids = y_ids[:y_ids.index(eos)]
    elif isinstance(eos, (list, tuple)):
        cut = None
        for e in eos:
            if e in y_ids:
                idx = y_ids.index(e)
                cut = idx if cut is None else min(cut, idx)
        if cut is not None:
            y_ids = y_ids[:cut]
    return y_ids


# ---------------------------- Layers & hooks ---------------------------- #

def _get_layers_module(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"): return model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        return model.model.decoder.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"): return model.transformer.h
    raise AttributeError("Could not locate decoder layers on model.")

def parse_layers(spec: str, num_layers: int) -> List[int]:
    spec = (spec or "").strip().lower()
    if spec == "last4": return list(range(max(0, num_layers - 4), num_layers))
    if spec == "last8": return list(range(max(0, num_layers - 8), num_layers))
    if spec == "all":   return list(range(num_layers))
    if spec.startswith("last") and spec[4:].isdigit():
        n = int(spec[4:])
        return list(range(max(0, num_layers - n), num_layers))
    idxs: List[int] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if tok:
            try:
                i = int(tok)
                if 0 <= i < num_layers: idxs.append(i)
            except ValueError:
                pass
    idxs = sorted(set(idxs))
    return idxs if idxs else list(range(max(0, num_layers - 4), num_layers))

def _make_attn_prehook(s_eff: int, e_eff: int):
    def pre_hook(module, args, kwargs):
        if kwargs and "hidden_states" in kwargs and kwargs["hidden_states"] is not None:
            hs = kwargs["hidden_states"]; B, T, H = hs.shape
            s, e = max(0, min(T, s_eff)), max(0, min(T, e_eff))
            if s < e:
                hs = hs.clone()
                hs[:, s:e, :] = 0
                kwargs["hidden_states"] = hs
            return args, kwargs
        if args and args[0] is not None:
            hs = args[0]; B, T, H = hs.shape
            s, e = max(0, min(T, s_eff)), max(0, min(T, e_eff))
            if s < e:
                hs = hs.clone()
                hs[:, s:e, :] = 0
                args = list(args); args[0] = hs; args = tuple(args)
            return args, kwargs
        return args, kwargs
    return pre_hook

def logp_for_positions(lp: torch.Tensor, ids: torch.Tensor, positions: List[int]) -> List[float]:
    seq = ids[0]
    vals: List[float] = []
    for t in positions:
        if t <= 0: vals.append(float("nan"))
        else:
            tok_id = seq[t].item()
            vals.append(lp[0, t - 1, tok_id].item())
    return vals


# ---------------------------- Helpers: entity focus & control ---------------------------- #

def normalize_token(s: str) -> str:
    return "".join(ch for ch in s.strip() if ch.isalnum()).lower()

def select_entity_positions(tok: AutoTokenizer, tf_ids_list: List[int], prompt_len: int, answer_string: str) -> List[int]:
    target_words = [normalize_token(w) for w in answer_string.split()]
    sel: List[int] = []
    for pos in range(prompt_len, len(tf_ids_list)):
        txt = tok.decode([tf_ids_list[pos]], skip_special_tokens=True)
        if normalize_token(txt) in target_words:
            sel.append(pos)
    return sel  # may be empty; caller will fall back to all answer positions

def sample_control_slice(prompt_len: int, length: int, avoid: List[Tuple[int,int]], trials: int = 12) -> Optional[Tuple[int,int]]:
    for _ in range(trials):
        if prompt_len <= length: break
        s = random.randint(0, prompt_len - length)
        e = s + length
        bad = False
        for (as_, ae_) in avoid:
            if not (e <= as_ or ae_ <= s): bad = True; break
        if not bad: return (s, e)
    return None


# ---------------------------- Main scoring ---------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle_json", required=True)
    ap.add_argument("--model_name", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--gen_max_new_tokens", type=int, default=64)
    ap.add_argument("--layers", default="last8")
    ap.add_argument("--k_edges", type=int, default=8)
    ap.add_argument("--length_norm", action="store_true")
    ap.add_argument("--control_subtract", action="store_true")
    ap.add_argument("--control_trials", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--answer_string", type=str, default="")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    random.seed(args.seed)

    tok, model = load_lm(args.model_name, args.load_in_4bit)

    bundle = json.loads(open(args.bundle_json, "r", encoding="utf-8").read())
    prompt_ids = torch.tensor([bundle["input_ids"]], dtype=torch.long, device=model.device)
    attn_mask = torch.ones_like(prompt_ids)

    # 1) Decode Y
    y_ids = greedy_decode_answer_ids(model, prompt_ids, attn_mask, args.gen_max_new_tokens)
    if args.verbose:
        print(f"[score_edges] decoded {len(y_ids)} new tokens")
    generated_text = tok.decode(y_ids, skip_special_tokens=True) if y_ids else ""

    if len(y_ids) == 0:
        out = {
            "question": "",
            "edges": {},
            "retrieved": bundle.get("retrieved", []),
            "tf_input_ids": bundle["input_ids"],
            "answer_token_positions": [],
            "prompt_len": len(bundle["input_ids"]),
            "generated_ids": [],
            "generated_text": generated_text,
        }
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"[WARN] empty decode; wrote {args.out} with no edges.")
        return

    # 2) Teacher-forced sequence: [prompt || Y]
    tf_ids_list = bundle["input_ids"] + y_ids
    full_ids = torch.tensor([tf_ids_list], dtype=torch.long, device=model.device)
    full_mask = torch.ones_like(full_ids)
    prompt_len = prompt_ids.shape[1]
    all_ans_positions = list(range(prompt_len, full_ids.shape[1]))

    # Optional: focus on entity tokens
    if args.answer_string.strip():
        ans_positions = select_entity_positions(tok, tf_ids_list, prompt_len, args.answer_string)
        if args.verbose:
            print(f"[score_edges] focusing on {len(ans_positions)} entity positions")
        if not ans_positions:
            ans_positions = all_ans_positions
    else:
        ans_positions = all_ans_positions

    # 3) Base log p at targets
    base_lp = logprobs(model, full_ids, full_mask)
    base_logp = logp_for_positions(base_lp, full_ids, ans_positions)

    # 4) Spans from retrieval chunks
    spans: List[Tuple[int, int, int]] = []
    for ch in bundle["chunk_spans"]:
        spans.append((
            int(ch["tok_start_in_prompt"]),
            int(ch["tok_end_in_prompt"]),
            int(ch["chunk_global_id"])
        ))

    # Layer selection
    layers_mod = _get_layers_module(model)
    layer_ids = parse_layers(args.layers, len(layers_mod))
    if args.verbose:
        print(f"[score_edges] using layers: {layer_ids}")

    edges: Dict[str, List[Dict]] = {str(p): [] for p in ans_positions}

    for (s_raw, e_raw, cid) in spans:
        # clamp to prompt so we never ablate generated tokens
        s_eff = max(0, min(prompt_len, s_raw))
        e_eff = max(0, min(prompt_len, e_raw))
        if s_eff >= e_eff:
            if args.verbose:
                print(f"[score_edges] skip empty span [{s_raw},{e_raw}) -> [{s_eff},{e_eff})")
            continue

        # best-per-token deltas across layers for the true span
        best_true = [0.0] * len(ans_positions)
        for L in layer_ids:
            hook = layers_mod[L].self_attn.register_forward_pre_hook(
                _make_attn_prehook(s_eff, e_eff), with_kwargs=True
            )
            lp2 = logprobs(model, full_ids, full_mask)
            hook.remove()

            masked_logp = logp_for_positions(lp2, full_ids, ans_positions)
            for j in range(len(ans_positions)):
                d = base_logp[j] - masked_logp[j]
                if args.length_norm:
                    d = d / max(1, (e_eff - s_eff))
                if d > best_true[j]:
                    best_true[j] = float(d)

        # optional control subtraction
        if args.control_subtract:
            ctrl_best = [0.0] * len(ans_positions)
            ctrl = sample_control_slice(
                prompt_len, e_eff - s_eff, avoid=[(s_eff, e_eff)], trials=args.control_trials
            )
            if ctrl:
                cs, ce = ctrl
                for L in layer_ids:
                    hook = layers_mod[L].self_attn.register_forward_pre_hook(
                        _make_attn_prehook(cs, ce), with_kwargs=True
                    )
                    lp2 = logprobs(model, full_ids, full_mask)
                    hook.remove()

                    masked_logp = logp_for_positions(lp2, full_ids, ans_positions)
                    for j in range(len(ans_positions)):
                        d = base_logp[j] - masked_logp[j]
                        if args.length_norm:
                            d = d / max(1, (ce - cs))
                        if d > ctrl_best[j]:
                            ctrl_best[j] = float(d)
            # subtract control and clamp
            best_true = [max(0.0, bt - cb) for bt, cb in zip(best_true, ctrl_best)]

        # write edges
        for j, p_abs in enumerate(ans_positions):
            d = best_true[j]
            if d > 0.0:
                edges[str(p_abs)].append({
                    "span": [s_raw, e_raw],
                    "chunk_global_id": cid,
                    "weight": d
                })

    # sort & top-k per token
    for p in list(edges.keys()):
        edges[p].sort(key=lambda r: r["weight"], reverse=True)
        edges[p] = edges[p][:args.k_edges]

    # Emit (include generated ids/text so downstream hallucination never re-generates)
    question_text = ""
    if "Question:" in bundle.get("prompt", "") and "\nAnswer:" in bundle.get("prompt", ""):
        question_text = bundle["prompt"].split("Question:", 1)[1].split("\nAnswer:", 1)[0].strip()

    out = {
        "question": question_text,
        "edges": edges,
        "retrieved": bundle.get("retrieved", []),
        "tf_input_ids": tf_ids_list,
        "answer_token_positions": all_ans_positions,
        "prompt_len": prompt_len,
        "generated_ids": y_ids,
        "generated_text": generated_text,
        "config": {
            "model_name": args.model_name,
            "layers": layer_ids,
            "length_norm": bool(args.length_norm),
            "control_subtract": bool(args.control_subtract),
            "control_trials": args.control_trials,
            "gen_max_new_tokens": args.gen_max_new_tokens,
            "k_edges": args.k_edges
        }
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[OK] wrote {args.out}")


if __name__ == "__main__":
    main()

