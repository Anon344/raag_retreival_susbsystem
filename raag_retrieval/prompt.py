
from typing import List, Dict
from transformers import AutoTokenizer

def build_prompt_and_map_spans(question: str, retrieved: List[Dict], generator_tokenizer_name: str) -> Dict:
    tok = AutoTokenizer.from_pretrained(generator_tokenizer_name, use_fast=True)
    parts = ["You are a helpful assistant. Use the retrieved evidence to answer.\n"]
    for r in retrieved:
        parts.append(f"[DOC {r['rank']}]\n{r['text']}\n")
    parts.append(f"Question: {question}\nAnswer:")
    prompt = "\n".join(parts)

    enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
    # Fast, deterministic span mapping based on concatenation order
    chunk_spans = []
    # Compute token bounds by mapping chunk text occurrences in the built prompt
    # We rely on the deterministic assembly above; a simple scanning approach:
    offsets = enc["offset_mapping"]
    full = prompt
    cursor = full.find("\n") + 1  # after first line
    for r in retrieved:
        header = f"[DOC {r['rank']}]\n"
        header_pos = full.find(header, cursor)
        assert header_pos != -1
        text_start = header_pos + len(header)
        text_end = text_start + len(r['text'])
        # find token indices covering [text_start, text_end)
        tok_start = next(i for i,(a,b) in enumerate(offsets) if a >= text_start)
        tok_end = next((i for i,(a,b) in enumerate(offsets) if a >= text_end), len(offsets))
        chunk_spans.append({
            "chunk_global_id": r["chunk_global_id"],
            "tok_start_in_prompt": tok_start,
            "tok_end_in_prompt": tok_end,
            "text": r["text"],
        })
        cursor = text_end + 1

    q_prefix = f"Question: {question}\nAnswer:"
    q_start_char = full.rfind(q_prefix)
    tok_q_start = next(i for i,(a,b) in enumerate(offsets) if a >= q_start_char)
    return {
        "prompt": prompt,
        "input_ids": enc["input_ids"],
        "chunk_spans": chunk_spans,
        "question_span": (tok_q_start, len(offsets)),
    }
