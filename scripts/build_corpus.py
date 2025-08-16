#!/usr/bin/env python3
"""
Build a plaintext Wikipedia corpus compatible with your current `corpus.txt`
-> `scripts/build_index.py` workflow, *without* relying on KILT.

Output format: UTF-8 text file where each passage is separated by a blank line.

It will try sources in this order unless you explicitly pass --dataset/--config:
  1) wikimedia/wikipedia  (auto-discover newest English snapshot)
  2) wikipedia            (defaults to 20220301.en)
If both fail, it prints helpful instructions.

Examples
--------
# 1) Just build corpus.txt (25k pages, ~5-sentence chunks)
python -m scripts.build_wikipedia_corpus_txt \
  --out_text corpus/wiki_corpus.txt \
  --max_pages 25000 \
  --chunk_sentences 5 \
  --chunk_chars 900

# 2) Build corpus.txt and immediately build a FAISS index + mapping
python -m scripts.build_wikipedia_corpus_txt \
  --out_text corpus/wiki_corpus.txt \
  --max_pages 25000 \
  --chunk_sentences 5 \
  --chunk_chars 900 \
  --run_index \
  --index_path index/faiss.index \
  --mapping_path index/chunk_meta.jsonl \
  --generator_tokenizer meta-llama/Meta-Llama-3-8B-Instruct \
  --encoder_name sentence-transformers/all-MiniLM-L6-v2 \
  --chunk_size 220 \
  --chunk_overlap 40

# 3) Force a specific dataset/config (if you know it works in your env)
python -m scripts.build_wikipedia_corpus_txt \
  --dataset wikimedia/wikipedia \
  --config 20231101.en \
  --out_text corpus/wiki_corpus.txt
"""
import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ----------------------------- dataset loading -----------------------------

def _try_load_wikimedia_wikipedia(auto_config: bool = True, config: Optional[str] = None):
    from datasets import load_dataset, get_dataset_config_names
    if config:
        ds = load_dataset("wikimedia/wikipedia", config, split="train")
        print(f"[load] loaded wikimedia/wikipedia:{config} | {len(ds):,} pages")
        return ds
    if not auto_config:
        return None
    # auto-discover the latest English snapshot
    cfgs = get_dataset_config_names("wikimedia/wikipedia")
    # configs look like 'YYYYMMDD.lang' e.g., '20231101.en'
    en_cfgs = [c for c in cfgs if c.endswith(".en")]
    def _key(c: str):
        m = re.match(r"^(\d{8})\.en$", c)
        return int(m.group(1)) if m else -1
    en_cfgs.sort(key=_key, reverse=True)
    for c in en_cfgs:
        try:
            ds = load_dataset("wikimedia/wikipedia", c, split="train")
            print(f"[load] loaded wikimedia/wikipedia:{c} | {len(ds):,} pages")
            return ds
        except Exception as _:
            continue
    return None

def _try_load_wikipedia(config: Optional[str] = None):
    from datasets import load_dataset
    cfg = config or "20220301.en"
    ds = load_dataset("wikipedia", cfg, split="train")
    print(f"[load] loaded wikipedia:{cfg} | {len(ds):,} pages")
    return ds

def _load_wikipedia(dataset_name: Optional[str], config: Optional[str]):
    """
    Returns a datasets.Dataset of Wikipedia pages (fields: 'text', 'title', ...).
    Tries user-specified dataset/config first; else robust fallbacks.
    """
    try:
        from datasets import load_dataset  # noqa: F401
    except Exception as e:
        raise SystemExit("Please install Hugging Face Datasets: pip install datasets") from e

    # 1) honor explicit user choice
    if dataset_name:
        if dataset_name == "wikimedia/wikipedia":
            return _try_load_wikimedia_wikipedia(auto_config=not bool(config), config=config)
        elif dataset_name == "wikipedia":
            return _try_load_wikipedia(config=config)
        else:
            from datasets import load_dataset
            ds = load_dataset(dataset_name, config, split="train")
            print(f"[load] loaded {dataset_name}:{config or '(default)'} | {len(ds):,} pages")
            return ds

    # 2) try wikimedia/wikipedia (auto-config)
    try:
        ds = _try_load_wikimedia_wikipedia(auto_config=True, config=None)
        if ds is not None:
            return ds
    except Exception as _:
        pass

    # 3) fallback to wikipedia (20220301.en)
    try:
        return _try_load_wikipedia(config=None)
    except Exception as e:
        raise SystemExit(
            "Could not load any Wikipedia dataset.\n"
            "Tried: wikimedia/wikipedia (auto) and wikipedia:20220301.en.\n"
            "Workarounds:\n"
            "  - Pass --dataset wikimedia/wikipedia --config <YYYYMMDD>.en\n"
            "  - Or --dataset wikipedia --config 20220301.en\n"
            "  - Ensure `datasets` can access the Hub from your environment.\n"
            f"Last error: {repr(e)}"
        )

# ----------------------------- text chunking -----------------------------

_SENT_SPLIT = re.compile(r'(?<=[\.\!\?])\s+')

def _paragraphs_from_example(ex: Dict[str, Any]) -> List[str]:
    """
    For 'wikimedia/wikipedia' or 'wikipedia' datasets, 'text' is typically a str.
    We treat the whole article text and sentence-split it.
    """
    txt = (ex.get("text") or "").strip()
    if not txt:
        return []
    # Split long article into pseudo-paragraph sentences
    # We'll re-accumulate into chunks below.
    return [txt]

def _chunk_text(
    article_text: str,
    chunk_sentences: int,
    chunk_chars: int
) -> List[str]:
    sents = [s for s in _SENT_SPLIT.split(article_text) if s.strip()]
    chunks: List[str] = []
    cur: List[str] = []
    cur_chars = 0
    for s in sents:
        s = s.strip()
        if not s:
            continue
        exceed = (len(cur) >= chunk_sentences) or (cur_chars + len(s) + (1 if cur else 0) > chunk_chars)
        if exceed and cur:
            chunks.append(" ".join(cur).strip())
            cur, cur_chars = [], 0
        cur.append(s)
        cur_chars += len(s) + (1 if cur else 0)
    if cur:
        chunks.append(" ".join(cur).strip())
    # fallback: if splitting failed, keep the raw text chunked by char cap
    if not chunks and article_text:
        txt = article_text.strip()
        step = max(256, min(chunk_chars, 4096))
        for i in range(0, len(txt), step):
            chunks.append(txt[i:i+step].strip())
    return [c for c in chunks if c]

# ----------------------------- writer -----------------------------

def write_corpus_txt(
    out_text: Path,
    ds,
    max_pages: Optional[int],
    chunk_sentences: int,
    chunk_chars: int,
    dedupe: bool
) -> Tuple[int, int]:
    out_text.parent.mkdir(parents=True, exist_ok=True)
    seen: set = set() if dedupe else set()
    n_pages = len(ds)
    limit = min(n_pages, max_pages) if max_pages else n_pages

    pages_written = 0
    passages_written = 0

    with out_text.open("w", encoding="utf-8") as f:
        for i in range(limit):
            ex = ds[i]
            paras = _paragraphs_from_example(ex)
            if not paras:
                continue
            wrote_any = False
            for p in paras:
                chunks = _chunk_text(p, chunk_sentences, chunk_chars)
                for ch in chunks:
                    key = ch.strip()
                    if not key:
                        continue
                    if dedupe:
                        if key in seen:
                            continue
                        seen.add(key)
                    f.write(key + "\n\n")
                    passages_written += 1
                    wrote_any = True
            if wrote_any:
                pages_written += 1
            if (i + 1) % 1000 == 0:
                print(f"[write] processed {i+1:,}/{limit:,} pages | passages so far: {passages_written:,}")

    print(f"[OK] wrote plaintext corpus: {out_text}")
    print(f"     pages written: {pages_written:,} | passages written: {passages_written:,}")
    return pages_written, passages_written

# ----------------------------- optional: index build -----------------------------

def run_build_index(
    corpus_path: Path,
    index_path: Path,
    mapping_path: Path,
    generator_tokenizer: str,
    encoder_name: str,
    chunk_size: int,
    chunk_overlap: int
):
    cmd = [
        sys.executable, "-m", "scripts.build_index",
        "--input", str(corpus_path),
        "--format", "plaintext",
        "--generator_tokenizer", generator_tokenizer,
        "--encoder_name", encoder_name,
        "--chunk_size", str(chunk_size),
        "--chunk_overlap", str(chunk_overlap),
        "--index_path", str(index_path),
        "--mapping_path", str(mapping_path)
    ]
    print("[RUN]", " ".join(cmd), flush=True)
    import subprocess
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(res.stdout)
    if res.returncode != 0:
        raise SystemExit(f"Index build failed (exit {res.returncode}).")

# ----------------------------- CLI -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_text", required=True, help="Path to write plaintext corpus (corpus.txt)")
    ap.add_argument("--max_pages", type=int, default=None, help="Limit number of Wikipedia pages")
    ap.add_argument("--chunk_sentences", type=int, default=5, help="Max sentences per passage")
    ap.add_argument("--chunk_chars", type=int, default=900, help="Max characters per passage")
    ap.add_argument("--dedupe", action="store_true", help="Drop duplicate passages across pages")

    # Optional dataset override
    ap.add_argument("--dataset", default=None, help="Force a dataset name (e.g., wikimedia/wikipedia or wikipedia)")
    ap.add_argument("--config", default=None, help="Dataset config (e.g., 20231101.en or 20220301.en)")

    # Optional: directly run your existing build_index afterwards
    ap.add_argument("--run_index", action="store_true")
    ap.add_argument("--index_path", default="index/faiss.index")
    ap.add_argument("--mapping_path", default="index/chunk_meta.jsonl")
    ap.add_argument("--generator_tokenizer", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--encoder_name", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--chunk_size", type=int, default=220)
    ap.add_argument("--chunk_overlap", type=int, default=40)

    args = ap.parse_args()

    ds = _load_wikipedia(dataset_name=args.dataset, config=args.config)
    out_text = Path(args.out_text)
    out_text.parent.mkdir(parents=True, exist_ok=True)

    write_corpus_txt(
        out_text=out_text,
        ds=ds,
        max_pages=args.max_pages,
        chunk_sentences=args.chunk_sentences,
        chunk_chars=args.chunk_chars,
        dedupe=args.dedupe
    )

    if args.run_index:
        index_path = Path(args.index_path)
        mapping_path = Path(args.mapping_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        run_build_index(
            corpus_path=out_text,
            index_path=index_path,
            mapping_path=mapping_path,
            generator_tokenizer=args.generator_tokenizer,
            encoder_name=args.encoder_name,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )

if __name__ == "__main__":
    main()
