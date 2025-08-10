# scripts/score_edges.py
import argparse, json, torch, math
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict

def _device_dtype(load_in_4bit: bool):
    if torch.cuda.is_available():
        return {"device_map": "auto", "torch_dtype": torch.bfloat16, "quantization_config": None}
    return {"device_map": {"": "cpu"}, "torch_dtype": torch.float32, "quantization_config": None}

def load_lm(model_name: str, load_in_4bit: bool):
    from transformers import BitsAndBytesConfig
    kw = _device_dtype(load_in_4bit)
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    qcfg = None
    if load_in_4bit and torch.cuda.is_available():
        qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(model_name,
            device_map=kw["device_map"], torch_dtype=kw["torch_dtype"], quantization_config=qcfg)
    model.eval()
    return tok, model

def mask_span(input_ids, span, pad_id):
    x = input_ids.clone()
    s, e = span
    e = min(e, x.shape[1])
    x[0, s:e] = pad_id
    return x

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle_json", required=True, help="JSON from make_prompt.py")
    ap.add_argument("--model_name", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--answer_max_tokens", type=int, default=64)
    ap.add_argument("--k_edges", type=int, default=8)
    args = ap.parse_args()

    tok, model = load_lm(args.model_name, args.load_in_4bit)

    data = json.loads(open(args.bundle_json, "r", encoding="utf-8").read())
    input_ids = torch.tensor([data["input_ids"]], dtype=torch.long, device=model.device)
    attn_mask = torch.ones_like(input_ids)

    # base logprobs
    with torch.no_grad():
        base = model(input_ids=input_ids, attention_mask=attn_mask)
        base_lp = torch.log_softmax(base.logits, dim=-1)

    # answer region: last answer_max_tokens before the end (excluding final position)
    T = input_ids.shape[1]
    target_positions = list(range(max(0, T - args.answer_max_tokens), T - 1))

    pad_id = tok.pad_token_id or tok.eos_token_id
    edges: Dict[int, List] = {t: [] for t in target_positions}

    # For each retrieved chunk span, compute Δ log p for every target token
    for ch in data["chunk_spans"]:
        s = ch["tok_start_in_prompt"]; e = ch["tok_end_in_prompt"]
        corrupted = mask_span(input_ids, (s, e), pad_id)
        with torch.no_grad():
            out2 = model(input_ids=corrupted, attention_mask=attn_mask)
            lp2 = torch.log_softmax(out2.logits, dim=-1)
        for t in target_positions:
            tgt = input_ids[0, t]
            drop = (base_lp[0, t, tgt] - lp2[0, t, tgt]).item()
            if drop > 0.0:
                edges[t].append({
                    "span": [s, e],
                    "chunk_global_id": ch["chunk_global_id"],
                    "weight": float(drop)
                })

    # keep top-k per t
    for t in edges:
        edges[t].sort(key=lambda r: r["weight"], reverse=True)
        edges[t] = edges[t][:args.k_edges]

    print(json.dumps({
        "question": data["prompt"].split("Question:")[-1].split("\nAnswer:")[0].strip(),
        "edges": edges,
        "retrieved": data["retrieved"]
    }, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
