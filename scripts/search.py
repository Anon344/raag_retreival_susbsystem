
#!/usr/bin/env python3
import argparse, json
from raag_retrieval.search import search as do_search

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--encoder_name", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--index_path", default="index/faiss.index")
    ap.add_argument("--mapping_path", default="index/chunk_meta.jsonl")
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()
    hits = do_search(args.query, args.index_path, args.mapping_path, args.encoder_name, args.topk)
    print(json.dumps(hits, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
