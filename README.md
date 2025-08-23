
# RAAG: Retrieval-Augmented Attribution Generation

A transparent RAG evaluation pipeline based on activation patching and validated retrieval-to-generation attributions. This repository produces per-token attribution edges, provenance metrics, hallucination/NLI labels, robustness scores (RARS), baselines, and publication-ready graphs/tables.

## 🚀 Quick Start

### Environment Setup

**Requirements:**
- Python 3.10+
- CUDA GPU (recommended)
- Dense index (FAISS or equivalent) and chunk mapping

**Installation:**
```bash
pip install -r requirements.txt
```

**Inputs:**
- `--index_path`: Built offline dense index
- `--mapping_path`: JSONL file with chunk metadata (chunk_global_id, text, etc.)
- QA dataset (*.jsonl) with fields: id, question, answer

**Model Defaults:**
- **Retriever**: `sentence-transformers/all-MiniLM-L6-v2`
- **Generator LM**: `meta-llama/Llama-3.2-3B-Instruct` (supports `--load_in_4bit`)

## 🔄 End-to-End Pipeline

### Batch Processing

Use the validated batch runner for processing entire datasets:

```bash
python -m scripts.run_batch \
  --qa_jsonl data/hotpot_qa.jsonl \
  --index_path index/my_faiss.index \
  --mapping_path index/chunk_meta.jsonl \
  --encoder_name sentence-transformers/all-MiniLM-L6-v2 \
  --generator_tokenizer meta-llama/Llama-3.2-3B-Instruct \
  --model_name meta-llama/Llama-3.2-3B-Instruct \
  --load_in_4bit \
  --topk 3 \
  --gen_max_new_tokens 32 \
  --layers last8 \
  --k_edges 3 \
  --epsilon 0.1 \
  --length_norm \
  --k_prov 3 \
  --work_dir outputs/hotpot_qa \
  --out_csv outputs/hotpot_qa/metrics.csv
```

The batch runner executes all pipeline steps per question and generates one file per step using the pattern `outputs/{id}_*.json`.

## 📋 Pipeline Steps & Artifacts

### A. Prompt Construction (Retrieval)
**Script:** `scripts.make_prompt`  
**Input:** question, index_path, mapping_path  
**Output:** `*_bundle.json`

**Contents:**
- `prompt`: "[DOC 0] … [DOC k] … Question: … Answer:" template
- `retrieved`: selected chunk ids/metadata
- `input_ids`: tokenized prompt
- `chunk_spans`: token spans per chunk (tok_start_in_prompt, tok_end_in_prompt, chunk_global_id)

### B. Scoring Candidate Edges (Activation Patching)
**Script:** `scripts.score_edges` (external, validated)

**Key Idea:** Greedy-decode answer Y, build teacher-forced sequence [prompt || Y]. For each retrieved span and selected layers, zero hidden states over the span, recompute log-probs of Y, and take the Δ log p (= base – masked). Keep per-token max Δ across layers (upper bound of causal impact).

**Output:** `*_edges.json` with:
- `edges`: {abs_answer_pos -> [{span, chunk_global_id, weight}]} (top-k per token)
- `generated_ids`, `generated_text`
- `tf_input_ids`, `prompt_len`, `answer_token_positions`
- Config echo (layers, k, length_norm, control_subtract)

**Notes:** 
- `--length_norm` divides Δ by span length
- `--control_subtract` subtracts a random same-length prompt slice effect

### C. Validation (Per-Layer Best Δ, Epsilon Cutoff)
**Script:** `scripts.validate_edges` (external, validated)

**Key Idea:** Re-evaluate each candidate span→token at each layer, pick the layer with the largest Δ; accept if Δ ≥ epsilon.

**Output:** `*_validated.json` with:
- `validated_edges`: {abs_answer_pos -> [{span, chunk_global_id, weight, best_layer}]}
- Config echo (layers, epsilon, prompt_len)

### D. Provenance Metrics (Evidence Ranking Quality)
**Script:** `scripts.eval_provenance`  
**Input:** `*_bundle.json`, `*_validated.json`, `--answer`  
**Output:** `*_prov.json` with:

- P@1, P@3, NDCG@3, AP, AUPRC
- ("relevant" is derived from chunk–answer match heuristics; see script)

### E. Hallucination HRI (MNLI) - Optional
**Script:** `scripts.compute_hallucination`  
**Input:** bundle, validated, edges  
**Output:** `*_hallucination.json` (MNLI labels per chunk, HRI mass ratios)

*Note: In our study HRI often degenerated to 0; we do not report it in the main tables.*

### F. RARS (Retrieval Attribution Robustness)
**Script:** `scripts.compute_rars`

**Key Idea:** Quantifies robustness of attributions via union ablations and normalized validated mass.

**Output:** `*_rars.json` with `RARS_union` and `RARS_validated`.

### G. Baselines (For Agreement/Ablation)
- **Retrieval ranking baseline:** `*_baseline_retrieval.json`
  - Ranks chunks by retriever scores
- **Document Shapley baseline:** `scripts.baseline_shapley --mode doc` → `*_shapley_doc.json`
  - Shapley values over chunks using question-only vs masked context coalitions

### H. Graphs (Optional)
- **Interactive HTML:** `scripts.build_graph_interactive` → `*_raag.html`
  - Adjacency tensor, per-layer edge hover, optional SVD grouping
- **Static PNG:** `scripts.graph_to_png` → `*_graph.png`
  - Layer-colored edges for figures

## 📁 Directory Layout

```
outputs/
  {id}_bundle.json
  {id}_edges.json
  {id}_validated.json
  {id}_prov.json
  {id}_rars.json
  {id}_shapley_doc.json        # if baseline computed
  {id}_baseline_retrieval.json # if baseline computed
  {id}_raag.html               # optional
  {id}_graph.png               # optional
  metrics.csv                  # from run_batch
```

## 🔧 Manual Single-Example Run

### 1. Prompt Construction
```bash
python -m scripts.make_prompt \
  --question "Who discovered penicillin, and when?" \
  --index_path index/my_faiss.index \
  --mapping_path index/chunk_meta.jsonl \
  --encoder_name sentence-transformers/all-MiniLM-L6-v2 \
  --generator_tokenizer meta-llama/Llama-3.2-3B-Instruct \
  --topk 3 \
  --out outputs/toy_bundle.json
```

### 2. Edge Scoring
```bash
python -m scripts.score_edges \
  --bundle_json outputs/toy_bundle.json \
  --model_name meta-llama/Llama-3.2-3B-Instruct --load_in_4bit \
  --gen_max_new_tokens 32 \
  --layers last8 --k_edges 3 --length_norm \
  --answer_string "Alexander Fleming 1928" \
  --out outputs/toy_edges.json
```

### 3. Edge Validation
```bash
python -m scripts.validate_edges \
  --bundle_json outputs/toy_bundle.json \
  --edges_json outputs/toy_edges.json \
  --model_name meta-llama/Llama-3.2-3B-Instruct --load_in_4bit \
  --layers last8 --epsilon 0.1 \
  --out outputs/toy_validated.json
```

### 4. Provenance Evaluation
```bash
python -m scripts.eval_provenance \
  --bundle_json outputs/toy_bundle.json \
  --validated_json outputs/toy_validated.json \
  --answer "Alexander Fleming 1928" \
  --k 3 \
  --out outputs/toy_prov.json
```

### 5. RARS Computation
```bash
python -m scripts.compute_rars \
  --bundle_json outputs/toy_bundle.json \
  --edges_json outputs/toy_edges.json \
  --validated_json outputs/toy_validated.json \
  --model_name meta-llama/Llama-3.2-3B-Instruct --load_in_4bit \
  --layers last8 \
  --out outputs/toy_rars.json
```

## 📊 Summaries & Paper Tables

### Table A1 (Evidence Ranking)
Aggregate `precision_at_1`, `precision_at_3`, `ndcg_at_3`, `average_precision`, `auprc` from `*_prov.json`.

### Table A2 (Agreement vs Doc-Shapley)
Compute Spearman ρ, top-1 match, Jaccard@3 between RAAG chunk ranks (`validated_edges` mass) and `*_shapley_doc.json`.

### Table A3 (RARS)
Aggregate `RARS_union`, `RARS_validated` from `*_rars.json`.

**Helper Scripts:**
- `scripts.summarize_results.py`: Aggregates provenance + RARS and emits LaTeX tables
- `scripts.compute_baseline_agreement.py`: Matches RAAG vs `*_shapley_doc.json` and emits LaTeX

## ⚙️ Key Flags (Cheat Sheet)

| Category | Flag | Options |
|----------|------|---------|
| **Retrieval** | `--topk` | Integer |
| **Decode** | `--gen_max_new_tokens` | Integer |
| **Layers** | `--layers` | `last8` \| `last12` \| `all` \| `0,5,10` |
| **Edges** | `--k_edges` | Integer |
| **Edges** | `--length_norm` | Boolean |
| **Edges** | `--control_subtract` | Boolean |
| **Edges** | `--control_trials` | Integer |
| **Validation** | `--epsilon` | Float |
| **Quantization** | `--load_in_4bit` | Boolean |
| **Work Dir** | `--work_dir` | Path |

## 🚨 Troubleshooting

### `validated.json` Not Written

**Common Causes:**
1. **No answer tokens generated** (`empty generated_ids`) → Increase `--gen_max_new_tokens`
2. **No candidate edges** (Δ≈0 everywhere) → Relax `--epsilon`, increase `--k_edges`, or drop `--length_norm`
3. **GPU OOM / driver error** → Reduce layers (e.g., `last4`) or disable control subtraction
4. **Paths/permissions** → Ensure `--work_dir` exists and is writable

### Slow Scoring
- Limit `--layers` (e.g., `last8`)
- Reduce `--k_edges`
- Increase `--topk` only if necessary
- Use `--load_in_4bit`

### Graphs Look Sparse
- Enable `--include_all_answer_tokens` in the graph builder
- Lower `--min_edge`
- Set `--token_topm` to a larger value

## 🔬 Reproducibility & Logs

- The batch runner prints every sub-command and captures stdout/stderr
- **Seeds:** `score_edges`/`validate_edges` are deterministic given prompt and layers
- **Doc-Shapley:** Uses a fixed `--seed` and `--permutations`

## 📚 Citation

If you use this code in your research, please cite our work on transparent RAG evaluation using activation patching and validated retrieval-to-generation attributions.

## 🤝 Contributing

This pipeline is designed for research reproducibility and evaluation. Contributions that improve the robustness, efficiency, or interpretability of the attribution methods are welcome.

## 📄 License

See [LICENSE](LICENSE) file for details.
