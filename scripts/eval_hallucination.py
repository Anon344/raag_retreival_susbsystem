#!/usr/bin/env python3#!/usr/bin/env python3
"""
Evaluate hallucination-style metrics for a single (bundle, validated) pair.

What this script does:
- Aggregates validated mass per chunk (from validated_edges.json).
- Labels chunks as support / contradict / neutral via:
    * support: chunk text contains `answer_string` (case-insensitive)
    * contradict: optional --contradict_regex matches chunk text
- Computes:
    * support_mass, contradiction_mass, unlabeled_mass, total_mass
    * HRI = contradiction_mass / (support_mass + contradiction_mass) (safe denom)
- Determines a "generated" answer string via the following precedence:
    1) If --edges_json is provided and has "generated_text" -> use it.
    2) Else if --edges_json has "tf_input_ids" + "prompt_len" -> decode continuation.
    3) Else if --model_name is given -> run a greedy generate on bundle["prompt"].
    4) Else generated = null.

CLI example:
  python -m scripts.eval_hallucination \
    --bundle_json outputs/nq3_bundle.json \
    --validated_json outputs/nq3_validated.json \
    --answer_string "Jane Austen" \
    --edges_json outputs/nq3_edges.json \
    --model_name meta-llama/Meta-Llama-3-8B-Instruct \
    --load_in_4bit \
    --gen_max_new_tokens 32 \
    --out outputs/nq3_hallucination.json
"""

import argparse, json, os, re
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ---------------------------- IO helpers ---------------------------- #

def load_json(p: str):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------- Mass aggregation ---------------------------- #

def aggregate_validated(validated_edges: Dict[str, List[Dict]]) -> Dict[int, float]:
    """
    Sum validated weights per chunk id across all target positions.
    """
    per_chunk = defaultdict(float)
    for _t, edges in validated_edges.items():
        for e in edges:
            cid = int(e.get("chunk_global_id", -1))
            w = float(e.get("weight", 0.0))
            if cid >= 0:
                per_chunk[cid] += w
    return dict(per_chunk)


# ---------------------------- Chunk labeling ---------------------------- #

def label_chunks_from_bundle(
    bundle: Dict,
    answer_string: str,
    contradict_pattern: Optional[re.Pattern]
) -> Tuple[set, set, set, Dict[int, str]]:
    """
    Return (support_set, contradict_set, neutral_set, cid->text)
    Priority: support > contradict > neutral
    """
    cid2text = {int(ch["chunk_global_id"]): ch.get("text", "") for ch in bundle.get("chunk_spans", [])}
    ans_lc = (answer_string or "").lower()

    support, contradict, neutral = set(), set(), set()
    for cid, txt in cid2text.items():
        t_lc = (txt or "").lower()
        is_support = (bool(ans_lc) and (ans_lc in t_lc))
        is_contra = bool(contradict_pattern.search(txt)) if contradict_pattern is not None else False
        if is_support:  # support wins over contradiction match
            support.add(cid)
        elif is_contra:
            contradict.add(cid)
        else:
            neutral.add(cid)
    return support, contradict, neutral, cid2text


# ---------------------------- Generation / decoding ---------------------------- #

def load_tokenizer_for_decode(preferred_model_or_tok: Optional[str]) -> AutoTokenizer:
    name = preferred_model_or_tok or "meta-llama/Meta-Llama-3-8B-Instruct"
    tok = AutoTokenizer.from_pretrained(name, use_fast=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    return tok

def load_lm(model_name: str, load_in_4bit: bool):
    from transformers import BitsAndBytesConfig
    tok = load_tokenizer_for_decode(model_name)
    qcfg = None
    if load_in_4bit:
        qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=qcfg,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model.eval()
    return tok, model

def clean_generated_text(s: str) -> str:
    """
    Normalize some artifacts from earlier prompts (underscores, extra choices).
    Keep first line; strip whitespace.
    """
    if not s: return s
    s = s.replace("_______________________", "").strip()
    # take only the first line; many instruction prompts place the answer on line 1
    s = s.split("\n")[0].strip()
    return s

@torch.no_grad()
def generate_answer(model, tok, prompt: str, max_new_tokens: int = 64) -> str:
    ids = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        num_beams=1,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        return_dict_in_generate=True,
    )
    full = tok.decode(out.sequences[0], skip_special_tokens=True)
    # Return the continuation after the prompt, first line only
    suffix = full[len(prompt):]
    return clean_generated_text(suffix)

def decode_from_tf_ids(tf_input_ids: List[int], prompt_len: int, tokenizer_name: Optional[str]) -> str:
    tok = load_tokenizer_for_decode(tokenizer_name)
    t_end = len(tf_input_ids)
    if prompt_len >= t_end:  # no continuation
        return ""
    cont_ids = tf_input_ids[prompt_len:]
    text = tok.decode(cont_ids, skip_special_tokens=True)
    return clean_generated_text(text)


# ---------------------------- Main ---------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle_json", required=True)
    ap.add_argument("--validated_json", required=True)
    ap.add_argument("--answer_string", required=True)
    ap.add_argument("--contradict_regex", default=None)
    # NEW: edges_json to read generated_text or decode tf_input_ids
    ap.add_argument("--edges_json", default=None)
    # Optional generation if decoding is unavailable
    ap.add_argument("--model_name", default=None)
    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--gen_max_new_tokens", type=int, default=64)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    bundle = load_json(args.bundle_json)
    val = load_json(args.validated_json)
    validated = val.get("validated_edges", {})

    # 1) label chunks
    rx = re.compile(args.contradict_regex, flags=re.I) if args.contradict_regex else None
    support, contradict, neutral, cid2text = label_chunks_from_bundle(bundle, args.answer_string, rx)

    # 2) masses
    masses = aggregate_validated(validated)
    support_mass = sum(masses.get(cid, 0.0) for cid in support)
    contradiction_mass = sum(masses.get(cid, 0.0) for cid in contradict)
    total_mass = sum(masses.values())
    unlabeled_mass = max(0.0, total_mass - support_mass - contradiction_mass)

    denom = max(1e-8, support_mass + contradiction_mass)
    HRI = contradiction_mass / denom

    # 3) get generated text with robust fallbacks
    generated_text = None
    # 3a) Prefer edges_json content
    if args.edges_json:
        try:
            ej = load_json(args.edges_json)
        except Exception:
            ej = None
        if ej is not None:
            gen = ej.get("generated_text", None)
            if isinstance(gen, str) and gen.strip():
                generated_text = clean_generated_text(gen)
            elif isinstance(ej.get("tf_input_ids"), list) and "prompt_len" in ej:
                try:
                    tokenizer_name = args.model_name or bundle.get("generator_tokenizer") or None
                    generated_text = decode_from_tf_ids(
                        tf_input_ids=ej["tf_input_ids"],
                        prompt_len=int(ej["prompt_len"]),
                        tokenizer_name=tokenizer_name,
                    )
                    if not generated_text:
                        generated_text = None
                except Exception:
                    generated_text = None

    # 3b) If still None and we have a model_name, actually generate
    if (generated_text is None) and args.model_name:
        try:
            tok, model = load_lm(args.model_name, args.load_in_4bit)
            prompt = bundle.get("prompt") or ""
            generated_text = generate_answer(model, tok, prompt, max_new_tokens=args.gen_max_new_tokens)
            if not generated_text:
                generated_text = None
        except Exception:
            generated_text = None

    # 4) correctness (only if we have generated text)
    correct = None
    if isinstance(generated_text, str):
        correct = (args.answer_string.lower() in generated_text.lower())

    out = {
        "id": os.path.splitext(os.path.basename(args.validated_json))[0],
        "question": bundle.get("question"),
        "answer_gold": args.answer_string,
        "generated": generated_text,
        "correct": correct,
        "support_mass": support_mass,
        "contradiction_mass": contradiction_mass,
        "unlabeled_mass": unlabeled_mass,
        "total_mass": total_mass,
        "HRI": HRI,
        "support_cids": sorted(list(support)),
        "contradiction_cids": sorted(list(contradict)),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(
        "[OK] wrote", args.out,
        f"| HRI={HRI:.3f} support={support_mass:.3f} contra={contradiction_mass:.3f}",
        f"| generated={'<none>' if generated_text is None else repr(generated_text)}",
        f"| correct={correct}"
    )


if __name__ == "__main__":
    main()
