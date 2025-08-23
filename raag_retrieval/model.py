# raag/model.py
import math, os
from typing import List, Iterable, Optional
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from dataclasses import dataclass

# ---------- Global singletons (kept across imports) ----------
_GEN = None
_GEN_TOK = None
_ENC = None
_ENC_TOK = None

@dataclass
class GenConfig:
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct"
    load_in_4bit: bool = True
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

def get_generator(cfg: GenConfig = GenConfig()):
    """Load/return generator LM + tokenizer once."""
    global _GEN, _GEN_TOK
    if _GEN is not None and _GEN_TOK is not None:
        return _GEN_TOK, _GEN
    from transformers import BitsAndBytesConfig
    qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16) if cfg.load_in_4bit else None
    tok = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        device_map="auto",
        quantization_config=qcfg,
        torch_dtype=cfg.dtype,
    )
    model.eval()
    _GEN_TOK, _GEN = tok, model
    return tok, model

@dataclass
class EncConfig:
    encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2"  # d=384
    max_length: int = 256
    device: Optional[str] = None  # "cuda" or "cpu"

def get_encoder(cfg: EncConfig = EncConfig()):
    """Load/return sentence encoder + tok once (mean pooling)."""
    global _ENC, _ENC_TOK
    if _ENC is not None and _ENC_TOK is not None:
        return _ENC_TOK, _ENC, 384  # known dim for MiniLM-L6
    tok = AutoTokenizer.from_pretrained(cfg.encoder_name, use_fast=True)
    model = AutoModel.from_pretrained(cfg.encoder_name)
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    _ENC_TOK, _ENC = tok, model
    # derive dim
    with torch.no_grad():
        ids = tok("x", return_tensors="pt", truncation=True, max_length=8).to(model.device)
        out = model(**ids)
        d = out.last_hidden_state.shape[-1]
    return tok, model, d

@torch.no_grad()
def encode_texts(texts: List[str], batch_size: int = 256, cfg: EncConfig = EncConfig()) -> np.ndarray:
    tok, enc, dim = get_encoder(cfg)
    device = enc.device
    outs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        x = tok(batch, return_tensors="pt", truncation=True, padding=True, max_length=cfg.max_length)
        x = {k: v.to(device) for k, v in x.items()}
        h = enc(**x).last_hidden_state  # [B,T,D]
        mask = (x["attention_mask"].unsqueeze(-1) > 0).float()
        emb = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)  # mean pool
        emb = torch.nn.functional.normalize(emb, p=2, dim=-1)  # cosine -> IP
        outs.append(emb.cpu().numpy())
    return np.concatenate(outs, axis=0)
