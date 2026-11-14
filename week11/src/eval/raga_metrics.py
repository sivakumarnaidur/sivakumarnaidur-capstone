"""src/eval/ragas_metrics.py — the four RAG-specific metrics, two ways to run.

W11 standardises on RAGAS for the four RAG metrics:
  • faithfulness        — is the answer supported by the retrieved context?
  • answer_relevance    — does the answer address the question?
  • context_precision   — are the top-K retrieved chunks the right ones?
  • context_recall      — did we retrieve everything we needed?

Each public function takes a list of `rows` and returns a `MetricResult`
(per-row scores + an aggregate). A row is a dict with:
    {"id", "question", "answer", "contexts": [str, ...], "ground_truth"}

TWO MODES
---------
LIVE  (opt-in: EVAL_LIVE=1, an API key, and `ragas` importable):
      real RAGAS metrics, each backed by an LLM judge call per row.

OFFLINE (the default — no LLM call, no API key):
      deterministic token-overlap *proxies*. These are NOT RAGAS numbers —
      they exist so the eval suite runs in CI / on a laptop with no API key,
      proving the wiring is sound. Every offline MetricResult is labelled
      `mode="offline-proxy"` so a proxy number is never mistaken for a real
      RAGAS score. The smoke test runs this path.

This split is the whole point of "eval as code": the suite is runnable and
assertable on every commit (offline), and produces real numbers on demand
(live). See docs/eval/README.md.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field


# ─── Result container ────────────────────────────────────────────────────────
@dataclass
class MetricResult:
    """One metric's outcome across all rows."""

    name: str
    mode: str                       # "live" or "offline-proxy"
    aggregate: float                # mean across rows, 0..1
    per_row: dict[str, float] = field(default_factory=dict)  # id -> score

    def __str__(self) -> str:
        return f"{self.name}={self.aggregate:.3f} ({self.mode}, n={len(self.per_row)})"


# ─── Mode detection ──────────────────────────────────────────────────────────
def offline_mode() -> bool:
    """Offline (no LLM call) by default. Live RAGAS is opt-in via EVAL_LIVE=1."""
    if os.getenv("EVAL_OFFLINE", "").strip().lower() in {"1", "true", "yes"}:
        return True
    if os.getenv("EVAL_LIVE", "").strip().lower() not in {"1", "true", "yes"}:
        return True
    if not os.getenv("OPENAI_API_KEY"):
        return True
    try:
        import ragas  # noqa: F401
    except Exception:
        return True
    return False


# ─── Token helpers (offline proxy) ───────────────────────────────────────────
_STOP = {
    "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on", "for",
    "with", "is", "are", "was", "were", "be", "been", "as", "at", "by", "it",
    "this", "that", "these", "those", "from", "your", "you", "may", "must",
    "can", "do", "does", "how", "what", "who", "i", "my", "me", "we", "our",
    "per", "up", "any", "all", "via", "not", "no", "they", "their",
}


def _tokens(text: str) -> set[str]:
    if not text:
        return set()
    raw = re.findall(r"[a-z0-9]+", text.lower())
    return {t for t in raw if len(t) > 2 and t not in _STOP}


def _overlap_fraction(a: set[str], b: set[str]) -> float:
    """Fraction of `a`'s tokens that appear in `b`. 0..1. Empty `a` -> 0.0."""
    if not a:
        return 0.0
    return len(a & b) / len(a)


# ─── Offline proxies (deterministic, dependency-free) ────────────────────────
def _proxy(name: str, rows: list[dict], fn) -> MetricResult:
    per_row = {r["id"]: round(fn(r), 4) for r in rows}
    agg = round(sum(per_row.values()) / len(per_row), 4) if per_row else 0.0
    return MetricResult(name=name, mode="offline-proxy", aggregate=agg, per_row=per_row)


def _faithfulness_proxy(r: dict) -> float:
    # How much of the answer is covered by the retrieved context?
    ans = _tokens(r.get("answer", ""))
    ctx = _tokens(" ".join(r.get("contexts", [])))
    return _overlap_fraction(ans, ctx)


def _answer_relevance_proxy(r: dict) -> float:
    # Does the answer address the question? Blend (a) how many of the question's
    # key terms the answer covers with (b) whether the answer engages the topic
    # at all. Short questions make pure token-overlap jumpy, so the topical-touch
    # term stabilises a correct answer near ~0.7 while a truly off-topic answer
    # (no shared terms) still collapses to 0.0.
    q = _tokens(r.get("question", ""))
    ans = _tokens(r.get("answer", ""))
    if not q:
        return 0.0
    coverage = _overlap_fraction(q, ans)            # 0..1
    touches = 1.0 if (q & ans) else 0.0             # any shared key term?
    return round(0.5 * coverage + 0.5 * touches, 4)


def _context_precision_proxy(r: dict) -> float:
    # Of the retrieved chunks, what fraction are relevant to the ground truth?
    gt = _tokens(r.get("ground_truth", ""))
    contexts = r.get("contexts", []) or []
    if not contexts:
        return 0.0
    relevant = sum(1 for c in contexts if _tokens(c) & gt)
    return relevant / len(contexts)


def _context_recall_proxy(r: dict) -> float:
    # How much of the ground truth is present across the retrieved chunks?
    gt = _tokens(r.get("ground_truth", ""))
    ctx = _tokens(" ".join(r.get("contexts", [])))
    return _overlap_fraction(gt, ctx)


# ─── Live RAGAS ──────────────────────────────────────────────────────────────
def _ragas_score(name: str, rows: list[dict], metric, column: str) -> MetricResult:
    """Run one RAGAS metric over the rows. Imported lazily (heavy deps).

    Targets ragas ~0.1.x (see requirements.txt for the pin). RAGAS has changed
    column names and metric imports across versions — if this raises, check the
    pinned version and the Lab Guide's 'RAGAS version pinning' note.
    """
    from datasets import Dataset
    from ragas import evaluate

    ds = Dataset.from_dict({
        "question":     [r["question"] for r in rows],
        "answer":       [r["answer"] for r in rows],
        "contexts":     [list(r.get("contexts", [])) for r in rows],
        "ground_truth": [r.get("ground_truth", "") for r in rows],
    })
    result = evaluate(ds, metrics=[metric])
    df = result.to_pandas()

    per_row = {
        rows[i]["id"]: round(float(df[column].iloc[i]), 4)
        for i in range(len(rows))
    }
    agg = round(float(df[column].mean()), 4)
    return MetricResult(name=name, mode="live", aggregate=agg, per_row=per_row)


# ─── Public API — one function per metric ────────────────────────────────────
def score_faithfulness(rows: list[dict]) -> MetricResult:
    if offline_mode():
        return _proxy("faithfulness", rows, _faithfulness_proxy)
    from ragas.metrics import faithfulness
    return _ragas_score("faithfulness", rows, faithfulness, "faithfulness")


def score_answer_relevance(rows: list[dict]) -> MetricResult:
    if offline_mode():
        return _proxy("answer_relevance", rows, _answer_relevance_proxy)
    from ragas.metrics import answer_relevancy
    return _ragas_score("answer_relevance", rows, answer_relevancy, "answer_relevancy")


def score_context_precision(rows: list[dict]) -> MetricResult:
    if offline_mode():
        return _proxy("context_precision", rows, _context_precision_proxy)
    from ragas.metrics import context_precision
    return _ragas_score("context_precision", rows, context_precision, "context_precision")


def score_context_recall(rows: list[dict]) -> MetricResult:
    if offline_mode():
        return _proxy("context_recall", rows, _context_recall_proxy)
    from ragas.metrics import context_recall
    return _ragas_score("context_recall", rows, context_recall, "context_recall")


# Convenience: all four at once (used by run_eval.py and the baseline doc).
ALL_METRICS = (
    score_faithfulness,
    score_answer_relevance,
    score_context_precision,
    score_context_recall,
)


def score_all(rows: list[dict]) -> dict[str, MetricResult]:
    results = [fn(rows) for fn in ALL_METRICS]   # one call each (matters live!)
    return {r.name: r for r in results}
