
from dataclasses import dataclass

@dataclass
class RetrievalBuildConfig:
    generator_tokenizer_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 256
    chunk_overlap: int = 32
    index_dir: str = "index"
    mapping_path: str = "index/chunk_meta.jsonl"
    index_path: str = "index/faiss.index"

@dataclass
class RetrievalSearchConfig:
    encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    index_path: str = "index/faiss.index"
    mapping_path: str = "index/chunk_meta.jsonl"
    topk: int = 5
