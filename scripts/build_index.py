
#!/usr/bin/env python3
import argparse
from raag_retrieval.corpus import read_plaintext, read_jsonl
from raag_retrieval.indexer import build_index

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--format", choices=["plaintext","jsonl"], default="plaintext")
    ap.add_argument("--text_field", default="text")
    ap.add_argument("--id_field", default="id")
    ap.add_argument("--title_field", default="title")
    ap.add_argument("--generator_tokenizer", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--encoder_name", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--chunk_size", type=int, default=256)
    ap.add_argument("--chunk_overlap", type=int, default=32)
    ap.add_argument("--index_path", default="index/faiss.index")
    ap.add_argument("--mapping_path", default="index/chunk_meta.jsonl")
    args = ap.parse_args()

    docs = read_plaintext(args.input) if args.format=="plaintext" else read_jsonl(args.input, text_field=args.text_field, id_field=args.id_field, title_field=args.title_field)
    n = build_index(docs, args.generator_tokenizer, args.encoder_name, args.index_path, args.mapping_path, args.chunk_size, args.chunk_overlap)
    print(f"[OK] Indexed {n} chunks -> {args.index_path}")

if __name__ == "__main__":
    main()
