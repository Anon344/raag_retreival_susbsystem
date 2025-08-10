
from typing import List, Dict
from transformers import AutoTokenizer

def chunk_by_generator_tokens(text: str, generator_tokenizer, chunk_size: int = 256, chunk_overlap: int = 32) -> List[Dict]:
    enc = generator_tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    n = len(offsets)
    chunks = []
    i = 0
    while i < n:
        j = min(n, i + chunk_size)
        if i < j:
            char_start = offsets[i][0]
            char_end = offsets[j-1][1]
            ch_text = text[char_start:char_end]
            chunks.append({
                "text": ch_text,
                "tok_start": i,
                "tok_end": j,
                "char_start": char_start,
                "char_end": char_end,
            })
        if j == n:
            break
        i = max(i + chunk_size - chunk_overlap, i + 1)
    return chunks
