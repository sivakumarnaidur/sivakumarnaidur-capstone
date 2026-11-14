"""src/rag/pipeline.py — a deterministic, offline, staged RAG pipeline.

Runs the six stages of a RAG query — embed → retrieve → rerank → cache → prompt
→ generate — over the capstone golden set, recording a Trace at each step. It is
fully synthetic (no OpenAI, no Qdrant, no cost) and deterministic, so the W12
debugging drill produces the *same* trace every time on every machine.

`run_query(entry)` runs the healthy pipeline. `run_query(entry, scenario_id=...)`
breaks exactly one stage so learners can practise localising the failure from
the trace. The scenario definitions live in src/debug/scenarios.py.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

try:
    from .trace import Trace
except ImportError:  # run directly as a script
    from trace import Trace

REPO_ROOT = Path(__file__).resolve().parents[2]   # week12/
GOLDEN_PATH = REPO_ROOT / "golden_set" / "capstone_v1.jsonl"

# A small fixed pool of clearly off-topic chunks — the "noise" a retriever can
# wrongly pull into the top-K.
DISTRACTORS = [
    {"chunk_id": "kb-441", "text": "The office kitchen is restocked every Monday and reception opens at eight."},
    {"chunk_id": "kb-887", "text": "Visitor parking is on level 2 and requires a validation stamp from the desk."},
    {"chunk_id": "kb-203", "text": "The annual company picnic is held in July at the lakeside pavilion."},
]

# The single line that keeps generation grounded. Dropping it is a prompt failure.
GROUNDING = ("Answer ONLY using the provided context. If the answer is not in the "
             "context, say you don't know and suggest contacting the relevant team.")

_STOP = {"the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "is",
         "are", "was", "were", "be", "as", "at", "by", "it", "this", "that", "from",
         "your", "you", "may", "must", "can", "do", "does", "how", "what", "who",
         "i", "my", "we", "our", "per", "up", "any", "all", "via", "not", "s"}


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", (text or "").lower())
            if len(t) > 2 and t not in _STOP}


def _score(chunk_text: str, question: str) -> float:
    q = _tokens(question)
    if not q:
        return 0.0
    return round(len(_tokens(chunk_text) & q) / len(q), 3)


def load_golden(path: str | Path = GOLDEN_PATH) -> dict[str, dict]:
    """Return {id: entry}. Entry = {id, question, ideal_answer, notes}."""
    out: dict[str, dict] = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("//"):
            e = json.loads(line)
            out[e["id"]] = e
    return out


def _relevant_chunks(entry: dict) -> list[dict]:
    """Split the ideal answer into two 'retrieved' chunks (a two-hop answer)."""
    words = str(entry["ideal_answer"]).split()
    mid = max(1, len(words) // 2)
    return [
        {"chunk_id": f"{entry['id']}-a", "text": " ".join(words[:mid])},
        {"chunk_id": f"{entry['id']}-b", "text": " ".join(words[mid:])},
    ]


def _candidates(entry: dict) -> list[dict]:
    """All candidate chunks (relevant + distractors) scored against the question."""
    cands = _relevant_chunks(entry) + [dict(d) for d in DISTRACTORS]
    for c in cands:
        c["score"] = _score(c["text"], entry["question"])
    return cands


# ─── the pipeline ────────────────────────────────────────────────────────────
def run_query(entry: dict, scenario_id: str = "healthy",
              golden: dict | None = None) -> Trace:
    """Run the six-stage pipeline. `scenario_id` breaks exactly one stage."""
    tr = Trace(question=entry["question"])
    rel_ids = {f"{entry['id']}-a", f"{entry['id']}-b"}

    # 1 ── embed ──────────────────────────────────────────────────────────────
    emb = sorted(_tokens(entry["question"]))
    tr.add("embed", f"{len(emb)} content tokens embedded", {"tokens": emb[:6]}, latency_ms=42)

    # 2 ── cache (checked early; a hit short-circuits the whole pipeline) ──────
    golden = golden or {}
    if scenario_id == "S5":                       # stale cache: hit from another question
        stale = golden.get("g003", {"ideal_answer": "(stale)"})
        tr.add("cache", "HIT — returned cached answer",
               {"cache_hit": True, "cached_for_question": "g003",
                "note": "answer served from cache; retrieve + generate skipped"}, latency_ms=6)
        tr.answer = stale["ideal_answer"]
        return tr
    tr.add("cache", "miss — running full pipeline", {"cache_hit": False}, latency_ms=5)

    # 3 ── retrieve ────────────────────────────────────────────────────────────
    cands = _candidates(entry)
    if scenario_id == "S1":                       # recall miss: drop the 2nd hop
        cands = [c for c in cands if c["chunk_id"] != f"{entry['id']}-b"]
    elif scenario_id == "S2":                     # precision noise: distractors on top
        for c in cands:
            if c["chunk_id"].startswith("kb-"):
                c["score"] = round(0.9 - 0.05 * next(i for i, d in enumerate(DISTRACTORS)
                                                     if d["chunk_id"] == c["chunk_id"]), 3)
            else:
                c["score"] = 0.15
    retrieved = sorted(cands, key=lambda c: c["score"], reverse=True)[:3]
    tr.add("retrieve", f"top-{len(retrieved)} by similarity",
           {"chunk_ids": [c["chunk_id"] for c in retrieved],
            "scores": [c["score"] for c in retrieved]}, latency_ms=88)

    # 4 ── rerank ──────────────────────────────────────────────────────────────
    reranked = sorted(retrieved, key=lambda c: c["score"], reverse=True)
    tr.add("rerank", "cross-encoder reorder",
           {"order": [c["chunk_id"] for c in reranked],
            "top1_score": reranked[0]["score"] if reranked else 0.0}, latency_ms=61)

    # 5 ── prompt ─────────────────────────────────────────────────────────────
    grounding = "" if scenario_id == "S4" else GROUNDING   # S4 drops the instruction
    ctx = "\n".join(f"- {c['text']}" for c in reranked)
    prompt = f"{grounding}\n\nContext:\n{ctx}\n\nQuestion: {entry['question']}".strip()
    tr.add("prompt", f"prompt built ({len(prompt.split())} tokens)",
           {"has_grounding_instruction": bool(grounding),
            "prompt_preview": prompt.replace("\n", " ")[:120]}, latency_ms=3)

    # 6 ── generate ───────────────────────────────────────────────────────────
    top = reranked[0] if reranked else {"chunk_id": "", "text": ""}
    top_relevant = top["chunk_id"] in rel_ids

    if scenario_id == "S6":                        # infra: generation error/timeout
        tr.add("generate", "ERROR — model call failed",
               {"error": "OpenAI timeout after 30000ms", "retries": 0},
               ok=False, latency_ms=30000)
        tr.answer = "(no answer returned — generation error)"
        return tr

    if scenario_id == "S3":                        # generation: hallucination
        answer = (str(entry["ideal_answer"]) +
                  " Unused leave can also be cashed out at 150% of salary on request.")
        note = "answer adds a claim not present in any retrieved chunk"
    elif scenario_id == "S4":                      # prompt: no grounding → invents a step
        answer = (str(entry["ideal_answer"]) +
                  " You should also post the incident in the public #general Slack channel.")
        note = "no grounding instruction; answer includes an unsupported step"
    elif scenario_id in ("S1",):                   # recall miss → partial answer
        answer = reranked[0]["text"] if reranked else ""
        note = "only the first hop was retrieved; answer is incomplete"
    elif scenario_id in ("S2",) or not top_relevant:   # wrong chunks → wrong answer
        answer = f"According to the retrieved context: {top['text']}"
        note = "answer generated from off-topic chunks"
    else:                                          # healthy: grounded, correct
        answer = str(entry["ideal_answer"])
        note = "grounded in retrieved context"
    tr.add("generate", "answer generated",
           {"tokens": len(answer.split()), "cost_usd": 0.00021, "note": note}, latency_ms=430)
    tr.answer = answer
    return tr


def main() -> None:
    """Smoke-test demo: run one healthy query and print its trace."""
    golden = load_golden()
    tr = run_query(golden["g001"], scenario_id="healthy", golden=golden)
    print(tr.render())


if __name__ == "__main__":
    main()