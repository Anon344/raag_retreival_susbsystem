#!/usr/bin/env python3
"""
Build small KILT subsets for Natural Questions and HotpotQA.

Outputs:
  - data/subsets/<task>_kilt_200.jsonl      (id, question, answer)
  - data/subsets/<task>_titles.txt          (unique gold Wikipedia titles)

Usage (examples):
  # Natural Questions (200)
  python -m scripts.build_kilt_subset --task nq --n 200

  # HotpotQA (200)
  python -m scripts.build_kilt_subset --task hotpotqa --n 200

Notes:
- Prefers KILT (kilt_tasks) datasets; robust fallbacks included.
- Picks split in order: validation -> dev -> test -> train (with a warning).
- Answers: first string found in record["output"][*]["answer"] (or top-level variants).
- Titles: union of output[*].provenance[*].title / wikipedia_title (if present).
"""
import argparse, json, os, random, sys
from typing import Dict, Any, Iterable, List, Optional, Tuple, Set
from datasets import load_dataset

OUT_DIR = "data/subsets"

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def pick_split(ds) -> str:
    # prefer validation/dev, else test, else train
    cand = ["validation", "dev", "test", "train"]
    for c in cand:
        if c in ds:
            return c
    # fall back to first available
    keys = list(ds.keys())
    if not keys:
        raise RuntimeError("No splits available on loaded dataset.")
    print(f"[WARN] Using split '{keys[0]}' (no standard split found).", file=sys.stderr)
    return keys[0]

def try_load_kilt(task: str):
    """
    Try to load KILT variants in a robust way.
    Returns a DatasetDict or raises.
    """
    # Primary: KILT tasks
    # common configs: 'nq' and 'hotpotqa'
    try:
        name = "nq" if task == "nq" else "hotpotqa"
        ds = load_dataset("kilt_tasks", name)
        print(f"[load] loaded kilt_tasks/{name} with splits: {list(ds.keys())}")
        return ds
    except Exception as e:
        print(f"[load] failed kilt_tasks/{task}: {e}", file=sys.stderr)

    # Fallbacks that still look like the task:
    if task == "nq":
        for cand in ["natural_questions", "natural_questions_open", "nq_open"]:
            try:
                ds = load_dataset(cand)
                print(f"[load] loaded '{cand}' with splits: {list(ds.keys())}")
                return ds
            except Exception as e:
                print(f"[load] failed '{cand}': {e}", file=sys.stderr)
    else:  # hotpotqa
        # Try fullwiki config first if available
        try:
            ds = load_dataset("hotpot_qa", "fullwiki")
            print(f"[load] loaded hotpot_qa/fullwiki with splits: {list(ds.keys())}")
            return ds
        except Exception as e:
            print(f"[load] failed hotpot_qa/fullwiki: {e}", file=sys.stderr)
        try:
            ds = load_dataset("hotpot_qa")
            print(f"[load] loaded hotpot_qa with splits: {list(ds.keys())}")
            return ds
        except Exception as e:
            print(f"[load] failed hotpot_qa: {e}", file=sys.stderr)

    raise RuntimeError(f"Could not load any dataset for task={task} (tried KILT and common fallbacks).")

def first_string(x) -> Optional[str]:
    if x is None:
        return None
    if isinstance(x, str):
        return x.strip() or None
    if isinstance(x, list):
        for v in x:
            s = first_string(v)
            if s:
                return s
    if isinstance(x, dict):
        # sometimes {'text': 'foo'} or {'answer': 'bar'}
        for k in ["text", "answer", "value"]:
            if k in x:
                return first_string(x[k])
    return None

def record_to_qa_and_titles(rec: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[str], Set[str]]:
    """
    Returns (qid, question, answer, titles)
    Tries KILT-like schema first, then fallbacks.
    """
    qid = rec.get("id") or rec.get("_id") or None
    # question field
    question = rec.get("input") or rec.get("question") or rec.get("query") or None
    if isinstance(question, dict):
        question = question.get("text") or question.get("question")
    if isinstance(question, list):
        question = first_string(question)

    # answer field
    answer = None
    if "output" in rec and isinstance(rec["output"], list):
        # KILT: take first available answer string
        for out in rec["output"]:
            a = first_string(out.get("answer"))
            if a:
                answer = a
                break
    if not answer:
        # try top-level variants
        for k in ["answer", "answers", "gold", "label"]:
            if k in rec:
                answer = first_string(rec[k])
                if answer:
                    break

    # provenance titles
    titles: Set[str] = set()
    if "output" in rec and isinstance(rec["output"], list):
        for out in rec["output"]:
            provs = out.get("provenance", []) or []
            for p in provs:
                t = p.get("title") or p.get("wikipedia_title")
                if t:
                    titles.add(t.strip())
    # Some datasets store evidence separately (rare in KILT, common in task-specific sets)
    if "evidence" in rec:
        for evset in rec["evidence"]:
            for item in evset:
                t = item.get("title") or item.get("wikipedia_title")
                if t:
                    titles.add(t.strip())

    return qid, first_string(question), first_string(answer), titles

def write_jsonl(rows: List[Dict[str, str]], out_path: str):
    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def write_titles(titles: Set[str], out_path: str):
    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w", encoding="utf-8") as f:
        for t in sorted(titles):
            f.write(t + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["nq", "hotpotqa"], help="Which KILT task to sample")
    ap.add_argument("--n", type=int, default=200, help="How many examples to sample")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out_dir", default=OUT_DIR)
    args = ap.parse_args()

    random.seed(args.seed)
    ds = try_load_kilt(args.task)
    split = pick_split(ds)
    data = ds[split]

    # If dataset is huge, sample indices deterministically
    n_total = len(data)
    if args.n > n_total:
        print(f"[WARN] requested n={args.n} but split has only {n_total}; using {n_total}.", file=sys.stderr)
    n_pick = min(args.n, n_total)
    idxs = list(range(n_total))
    random.shuffle(idxs)
    idxs = idxs[:n_pick]

    rows: List[Dict[str, str]] = []
    titles: Set[str] = set()

    for i, idx in enumerate(idxs):
        rec = data[int(idx)]
        qid, question, answer, rec_titles = record_to_qa_and_titles(rec)
        if not question or not answer:
            # skip malformed examples
            continue
        if not qid:
            qid = f"{args.task}_{idx}"
        rows.append({"id": qid, "question": question, "answer": answer})
        titles.update(rec_titles)

    out_jsonl = os.path.join(args.out_dir, f"{'nq' if args.task=='nq' else 'hotpot'}_kilt_{len(rows)}.jsonl")
    out_titles = os.path.join(args.out_dir, f"{'nq' if args.task=='nq' else 'hotpot'}_titles.txt")
    write_jsonl(rows, out_jsonl)
    write_titles(titles, out_titles)
    print(f"[OK] wrote {out_jsonl} and {out_titles}  |  {len(rows)} rows, {len(titles)} titles")

if __name__ == "__main__":
    main()
