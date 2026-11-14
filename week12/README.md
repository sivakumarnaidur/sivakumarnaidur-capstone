# Week 12 — RAG Pipeline Tracing & Debugging

A fully synthetic, offline, deterministic RAG pipeline (no OpenAI, no Qdrant,
no cost) built to practise one production skill: **reading a trace to localise
which stage of a RAG query broke.** Every run produces the same trace on every
machine, so the debugging drill is reproducible for every learner.

## Layout

| File | Purpose |
|---|---|
| `golden_set/capstone_v1.jsonl` | 20 golden Q&A entries (generated, see below) |
| `src/debug/golden.py` | Writes the golden set to `golden_set/` |
| `src/debug/trace.py` | `Trace` / `StageRecord` — a readable record of one query's six stages |
| `src/debug/pipeline.py` | The six-stage pipeline (`run_query`) that produces a `Trace` |
| `src/debug/scenarios.py` | Six broken-pipeline scenarios + the grading/answer-key engine |

## Run

From `week12/`, using the repo venv:

```bash
../.venv/bin/python src/debug/golden.py      # once: create the golden set
../.venv/bin/python src/debug/pipeline.py    # one healthy query, printed as a trace
../.venv/bin/python src/debug/scenarios.py   # full walkthrough: all 6 broken scenarios + answer key
```

All three scripts resolve paths relative to their own file location, so they
work the same regardless of your current working directory.

## The pipeline

`run_query(entry, scenario_id=...)` in `pipeline.py` runs six stages over a
golden question and records a `Trace`:

```
embed → retrieve → rerank → cache → prompt → generate
```

`scenario_id="healthy"` (the default) runs the pipeline unmodified.
`scenario_id="S1".."S6"` breaks exactly one stage, so a learner can practise
spotting the outlier stage from the trace alone.

## The failure taxonomy

Every scenario is classified into exactly one bucket:

| Bucket | Meaning |
|---|---|
| `retrieval` | the right chunk wasn't retrieved (precision or recall) |
| `generation` | right chunks retrieved, but the LLM answered wrong / hallucinated |
| `prompt` | the prompt itself is wrong (e.g. missing the grounding instruction) |
| `infra` | the pipeline broke (timeout, API error, stale cache) |

## The six scenarios

| ID | Title | Bucket | Question |
|---|---|---|---|
| S1 | The multi-hop answer is missing half | retrieval | g015 |
| S2 | The answer is about the office kitchen | retrieval | g004 |
| S3 | A confident fact that isn't in the docs | generation | g001 |
| S4 | An extra step nobody asked for | prompt | g007 |
| S5 | A fluent answer to a different question | infra | g010 |
| S6 | No answer at all | infra | g008 |

The full answer key for each scenario (`root_cause`, `fix`, `trace_tell`) lives
in `scenarios.SCENARIOS` and is only revealed by `grade()` after a correct
bucket guess — or in full via `scenarios.print_answer_key()`.

## Try it yourself

```python
from pathlib import Path
import sys
sys.path.insert(0, "src/debug")

from pipeline import load_golden, run_query
from scenarios import get, grade

golden = load_golden()
s = get("S2")
tr = run_query(golden[s.question_id], scenario_id=s.id, golden=golden)
print(tr.render())          # read the trace — which stage looks wrong?

print(grade(s.id, "retrieval"))   # guess a bucket; wrong guesses hint at the stage to re-check
```
