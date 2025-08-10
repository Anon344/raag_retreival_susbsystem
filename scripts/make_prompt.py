# scripts/make_prompt.py
import argparse, json
from raag_retrieval.search import search as do_search
from raag_retrieval.prompt import build_prompt_and_map_spans

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", required=True)
    ap.add_argument("--index_path", default="index/faiss.index")
    ap.add_argument("--mapping_path", default="index/chunk_meta.jsonl")
    ap.add_argument("--encoder_name", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--generator_tokenizer", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    hits = do_search(args.question, args.index_path, args.mapping_path, args.encoder_name, args.topk)
    bundle = build_prompt_and_map_spans(args.question, hits, args.generator_tokenizer)
    # keep only what we need for the next stage
    out = {
        "prompt": bundle["prompt"],
        "input_ids": bundle["input_ids"],
        "chunk_spans": bundle["chunk_spans"],
        "question_span": bundle["question_span"],
        "retrieved": hits,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
