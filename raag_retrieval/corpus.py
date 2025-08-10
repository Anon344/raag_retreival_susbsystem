
from typing import Iterable, Dict
import json

def read_plaintext(path: str) -> Iterable[Dict]:
    with open(path, 'r', encoding='utf-8') as f:
        buf = []
        doc_id = 0
        for line in f:
            if line.strip() == "":
                if buf:
                    text = "".join(buf).strip()
                    if text:
                        yield {"id": str(doc_id), "title": None, "text": text}
                        doc_id += 1
                    buf = []
            else:
                buf.append(line)
        if buf:
            text = "".join(buf).strip()
            if text:
                yield {"id": str(doc_id), "title": None, "text": text}

def read_jsonl(path: str, text_field: str = "text", id_field: str = "id", title_field: str = "title") -> Iterable[Dict]:
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            text = obj.get(text_field, "")
            if not text:
                continue
            yield {
                "id": str(obj.get(id_field, "")),
                "title": obj.get(title_field, None),
                "text": text,
            }
