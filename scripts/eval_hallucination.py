#!/usr/bin/env python3
import argparse, json, os, math, hashlib
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, AutoTokenizer as GenTok

# ---------- I/O ----------

def jload(p): return json.load(open(p, "r", encoding="utf-8"))
def jdump(o,p): json.dump(o, open(p,"w",encoding="utf-8"), ensure_ascii=False, indent=2)

def aggregate_validated(validated_edges: Dict[str, List[Dict]]) -> Dict[int, float]:
    mass = defaultdict(float)
    for _t, edges in validated_edges.items():
        for e in edges:
            cid = int(e.get("chunk_global_id", -1))
            if cid >= 0:
                mass[cid] += float(e.get("weight", 0.0))
    return dict(mass)

# ---------- Generated text helpers ----------

def clean_text(s):
    if not s: return s
    s = s.replace("_______________________","").replace("________________","").strip()
    return s

def decode_from_tf(edges_json: Dict, tokenizer_name: str) -> Optional[str]:
    tf = edges_json.get("tf_input_ids"); pl = edges_json.get("prompt_len")
    if not isinstance(tf, list) or pl is None: return None
    tok = GenTok.from_pretrained(tokenizer_name, use_fast=True)
    cont = tf[pl:]
    return clean_text(tok.decode(cont, skip_special_tokens=True))

def ensure_generated(bundle, edges, gen_model_name=None, gen_4bit=False, max_new_tokens=64):
    # prefer edges_json.generated_text
    if isinstance(edges.get("generated_text"), str) and edges["generated_text"].strip():
        return clean_text(edges["generated_text"])
    # try decode from TF ids
    tok_name = bundle.get("generator_tokenizer") or gen_model_name or "meta-llama/Llama-3.2-3B-Instruct"
    text = decode_from_tf(edges, tok_name)
    if text: return text
    # last resort: regenerate (rare in our pipeline)
    if gen_model_name:
        from transformers import BitsAndBytesConfig
        qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16) if gen_4bit else None
        tok = GenTok.from_pretrained(gen_model_name, use_fast=True)
        if tok.pad_token_id is None and tok.eos_token_id is not None: tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(gen_model_name, device_map="auto",
                                                     quantization_config=qcfg,
                                                     torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
        ids = tok(bundle["prompt"], return_tensors="pt").to(model.device)
        out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False, temperature=0.0)
        full = tok.decode(out[0], skip_special_tokens=True)
        return clean_text(full[len(bundle["prompt"]):])
    return None

# ---------- NLI ----------

def load_nli(model_name="sileod/mdeberta-v3-base-tasksource-nli"):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    return tok, model

@torch.no_grad()
def nli_label_batch(premises: List[str], hypotheses: List[str], tok, model, bs=16):
    """
    Returns list of ('entailment'|'contradiction'|'neutral', p_entail, p_contra)
    """
    device = model.device
    outs = []
    for i in range(0, len(premises), bs):
        p = premises[i:i+bs]
        h = hypotheses[i:i+bs]
        enc = tok(p, h, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
        logits = model(**enc).logits
        probs = F.softmax(logits, dim=-1).cpu()  # label order: entailment/neutral/contradiction for many MNLI models? -> check config
        # Map by model.config.id2label (robust)
        id2label = model.config.id2label
        # build indices
        idx_ent = [k for k,v in id2label.items() if v.lower().startswith("entail")][0]
        idx_neu = [k for k,v in id2label.items() if v.lower().startswith("neutral")][0]
        idx_con = [k for k,v in id2label.items() if v.lower().startswith("contrad")][0]
        for row in probs:
            pe, pn, pc = row[idx_ent].item(), row[idx_neu].item(), row[idx_con].item()
            if pc >= max(pe,pn): lab = "contradiction"
            elif pe >= max(pc,pn): lab = "entailment"
            else: lab = "neutral"
            outs.append((lab, pe, pc))
    return outs

def classify_chunks_with_nli(claim: str, cid2text: Dict[int,str], tok, model,
                             tau_entail=0.5, tau_contra=0.5, bs=16) -> Tuple[set,set,set,Dict[int,Tuple]]:
    """
    Return (support_ids, contradiction_ids, neutral_ids, scores).
    scores[cid] = (label, p_entail, p_contra)
    """
    cids = sorted(cid2text.keys())
    premises = [cid2text[cid] or "" for cid in cids]
    hypotheses = [claim]*len(cids)
    res = nli_label_batch(premises, hypotheses, tok, model, bs=bs)
    support, contra, neutral = set(), set(), set()
    scores = {}
    for cid, (lab, pe, pc) in zip(cids, res):
        if lab == "contradiction" and pc >= tau_contra: contra.add(cid)
        elif lab == "entailment" and pe >= tau_entail: support.add(cid)
        else: neutral.add(cid)
        scores[cid] = (lab, pe, pc)
    return support, contra, neutral, scores

# ---------- Main metric ----------

def compute_hri(validated_edges, support_ids, contra_ids):
    mass = aggregate_validated(validated_edges)
    s = sum(mass.get(cid,0.0) for cid in support_ids)
    c = sum(mass.get(cid,0.0) for cid in contra_ids)
    total = sum(mass.values())
    unlabeled = max(0.0, total - s - c)
    denom = s + c
    hri = None if denom <= 0 else (c / denom)
    return dict(
        support_mass=s, contradiction_mass=c, unlabeled_mass=unlabeled,
        total_mass=total, HRI=hri
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle_json", required=True)
    ap.add_argument("--validated_json", required=True)
    ap.add_argument("--edges_json", required=True, help="needed to decode generated answer")
    ap.add_argument("--nli_model", default="sileod/mdeberta-v3-base-tasksource-nli")
    ap.add_argument("--tau_entail", type=float, default=0.50)
    ap.add_argument("--tau_contra", type=float, default=0.50)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--gold_answer", default=None)
    ap.add_argument("--gen_model", default=None)
    ap.add_argument("--gen_4bit", action="store_true")
    ap.add_argument("--gen_max_new_tokens", type=int, default=64)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    bundle = jload(args.bundle_json)
    val    = jload(args.validated_json)
    edges  = jload(args.edges_json)
    validated = val.get("validated_edges", {})

    # decode / get generated answer
    generated = ensure_generated(bundle, edges, gen_model_name=args.gen_model,
                                 gen_4bit=args.gen_4bit, max_new_tokens=args.gen_max_new_tokens)
    # NLI model
    tok, nli = load_nli(args.nli_model)

    # cid->text
    cid2text = {int(ch["chunk_global_id"]): ch.get("text","") for ch in bundle.get("chunk_spans", [])}

    out = {
        "id": os.path.splitext(os.path.basename(args.validated_json))[0],
        "question": bundle.get("question"),
        "generated": generated,
        "gold_answer": args.gold_answer
    }

    # Evaluate HRI with respect to GENERATED claim (if present)
    if generated:
        sup_g, con_g, neu_g, scores_g = classify_chunks_with_nli(generated, cid2text, tok, nli,
                                                                 tau_entail=args.tau_entail, tau_contra=args.tau_contra,
                                                                 bs=args.batch_size)
        hri_g = compute_hri(validated, sup_g, con_g)
        out.update({
            "gen_support_cids": sorted(list(sup_g)),
            "gen_contradiction_cids": sorted(list(con_g)),
            "gen_scores": {int(k): {"label": v[0], "p_entail": v[1], "p_contra": v[2]} for k,v in scores_g.items()},
            "HRI_generated": hri_g
        })

    # Evaluate HRI with respect to GOLD claim (if provided)
    if args.gold_answer:
        sup_y, con_y, neu_y, scores_y = classify_chunks_with_nli(args.gold_answer, cid2text, tok, nli,
                                                                 tau_entail=args.tau_entail, tau_contra=args.tau_contra,
                                                                 bs=args.batch_size)
        hri_y = compute_hri(validated, sup_y, con_y)
        out.update({
            "gold_support_cids": sorted(list(sup_y)),
            "gold_contradiction_cids": sorted(list(con_y)),
            "gold_scores": {int(k): {"label": v[0], "p_entail": v[1], "p_contra": v[2]} for k,v in scores_y.items()},
            "HRI_gold": hri_y
        })

    jdump(out, args.out)
    print(f"[OK] wrote {args.out}")

if __name__ == "__main__":
    main()
