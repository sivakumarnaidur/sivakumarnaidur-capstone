# Week 11 — RAG Evaluation (offline proxy, no LLM calls)

Evaluation runs entirely offline by default: deterministic token-overlap
proxies, no API key, no cost, same result on every run.

## Layout

| File | Purpose |
|---|---|
| `golden_set/capstone_v1.jsonl` | 20 golden Q&A entries (generated, see below) |
| `src/eval/golden_dataset.py` | Writes the golden set to `golden_set/` |
| `src/eval/golden.py` | `GoldenEntry` model and `load_golden_set` |
| `src/eval/harness.py` | Builds eval rows (synthetic offline) and prints a report |
| `src/eval/raga_metrics.py` | Faithfulness, answer relevance, context precision, context recall |
| `src/eval/judge.py` | Task-success grader: `judge_offline` (default) or LLM `judge_one` |

## Run

From `week11/`, using the repo venv:

```bash
../.venv/bin/python src/eval/golden_dataset.py   # once: create the golden set
../.venv/bin/python src/eval/harness.py          # offline eval report
```

Expected: four metrics labelled `offline-proxy` and a task-success rate of 85%.
Rows `g005` (weak retrieval), `g015` (multi-hop miss) and `g019` (hallucination)
are seeded failures and should score lowest.

## How the proxies work

All use key-term overlap (stop words and tokens of 2 characters or fewer are dropped).

| Metric | Proxy |
|---|---|
| faithfulness | share of answer terms found in the retrieved contexts |
| answer_relevance | blend of question-term coverage and topical touch |
| context_precision | share of retrieved chunks sharing terms with the ground truth |
| context_recall | share of ground-truth terms present in the contexts |

The offline judge scores 1–4 on three dimensions:

- **accuracy:** coverage of the ideal answer's terms.
- **groundedness:** share of answer terms supported by the contexts.
- **format:** length ratio against the ideal answer.

`task_success` requires accuracy ≥ 3 and groundedness ≥ 3.

## Live mode (opt-in)

Proxies are not RAGAS scores. For real numbers, set all of:

```bash
export EVAL_LIVE=1 OPENAI_API_KEY=...   # plus QDRANT_URL; needs `ragas` and src/rag/*
```

`EVAL_OFFLINE=1` always forces offline, even if `EVAL_LIVE` is set.
