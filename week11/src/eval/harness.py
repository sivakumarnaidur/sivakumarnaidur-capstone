"""src/eval/harness.py — build the eval rows once, two ways.

Shared by eval/conftest.py (the pytest suite) and scripts/run_eval.py (the CLI),
so the offline synthetic pipeline and the live capstone pipeline are defined in
exactly one place.

Row shape (identical offline and live):
    {"id", "question", "answer", "contexts": [str,...], "ground_truth",
     "cache_hit", "latency_ms", "cost_usd", "notes"}
"""
from __future__ import annotations

from pathlib import Path

try:
    from .golden import GoldenEntry, load_golden_set
    from .judge import judge_offline
    from .raga_metrics import offline_mode, score_all, score_faithfulness
except ImportError:  # run directly as a script
    from golden import GoldenEntry, load_golden_set
    from judge import judge_offline
    from raga_metrics import offline_mode, score_all, score_faithfulness

REPO_ROOT = Path(__file__).resolve().parents[2]   # week11/
GOLDEN_PATH = REPO_ROOT / "golden_set" / "capstone_v1.jsonl"

# KB version active at eval time. Live mode reads it from the pipeline; offline
# pins the W10 value so every eval run and snapshot is tagged by KB state (D4).
OFFLINE_KB_VERSION = "v2.w10"

# A clean distractor sharing almost no content tokens with policy answers, so
# context precision reflects relevant-vs-retrieved honestly.
_DISTRACTOR = (
    "The office kitchen is restocked every Monday morning and the building "
    "reception desk is open from eight until six."
)


def _split_two(text: str) -> tuple[str, str]:
    words = text.split()
    mid = max(1, len(words) // 2)
    return " ".join(words[:mid]), " ".join(words[mid:])


def _synth_row(entry: GoldenEntry, index: int) -> dict:
    """Deterministically build one (answer, contexts) pair for a golden entry.

    Most rows are faithful and complete. A few are deliberately degraded so the
    report has real signal: g019 hallucinates, g015/g005 suffer retrieval misses.
    """
    gt = entry.ground_truth
    c1, c2 = _split_two(gt)
    contexts = [c1, c2, _DISTRACTOR]   # 2 relevant + 1 distractor
    answer = gt

    if entry.id == "g019":
        answer = gt + (" The company also matches up to five percent of salary "
                       "into a managed crypto retirement fund.")   # hallucination
    elif entry.id == "g015":
        contexts = [c1, _DISTRACTOR]                                # multi-hop miss
    elif entry.id == "g005":
        contexts = [c1[: max(1, len(c1) // 2)], _DISTRACTOR]        # weak retrieval

    cache_hit = index < 13          # ~65% warm hit rate, matching wk10
    return {
        "id":           entry.id,
        "question":     entry.question,
        "answer":       answer,
        "contexts":     contexts,
        "ground_truth": gt,
        "cache_hit":    cache_hit,
        "latency_ms":   184 if cache_hit else 785,
        "cost_usd":     0.000041 if cache_hit else 0.000412,
        "notes":        entry.notes,
    }


def _live_rows(golden: list[GoldenEntry]) -> list[dict]:  # pragma: no cover - live only
    """Run the real capstone over the golden set. Needs the learner's src/rag."""
    try:
        from src.rag.qdrant_rag import ask_rag
        from src.rag.qdrant_store import QdrantStore
        from src.rag.retrieval import (
            build_bm25_index,
            hybrid_retrieve,
            load_chunks_from_qdrant,
        )
    except Exception as exc:
        raise RuntimeError(
            "Live eval needs your capstone code (src/rag/*) on the path and "
            "OPENAI_API_KEY + QDRANT_URL set, with EVAL_LIVE=1. "
            "Default is offline (no LLM call). "
            f"Import error: {exc}"
        ) from exc

    store = QdrantStore()
    chunks = load_chunks_from_qdrant(store)          # tombstone-filtered (W10)
    bm25 = build_bm25_index(chunks)
    rows: list[dict] = []
    for e in golden:
        res = ask_rag(e.question, store, k=3)
        # Full chunk text for RAGAS contexts comes from retrieval, not from
        # ask_rag's stripped citations (D3).
        retrieved = hybrid_retrieve(store, bm25, chunks, e.question, top_k=3)
        rows.append({
            "id":           e.id,
            "question":     e.question,
            "answer":       res["answer"],
            "contexts":     [c.get("text", "") for c in retrieved],
            "ground_truth": e.ground_truth,
            "cache_hit":    res.get("cache_hit", False),
            "latency_ms":   res.get("latency_ms", 0),
            "cost_usd":     res.get("cost_usd", 0.0),
            "notes":        e.notes,
        })
    return rows


def active_kb_version() -> str:
    if offline_mode():
        return OFFLINE_KB_VERSION
    try:  # pragma: no cover - live only
        from src.rag.pipeline import KB_VERSION
        return KB_VERSION
    except Exception:  # pragma: no cover
        return "unknown"


def load_golden() -> list[GoldenEntry]:
    return load_golden_set(GOLDEN_PATH)


def build_rows(golden: list[GoldenEntry] | None = None) -> list[dict]:
    """The (question, answer, contexts, ground_truth, ...) rows under test."""
    golden = golden or load_golden()
    if offline_mode():
        return [_synth_row(e, i) for i, e in enumerate(golden)]
    return _live_rows(golden)

def main() -> None:
    rows = build_rows()
    print('rows:', len(rows), '| kb_version:', active_kb_version())

    sample = rows[0]
    print('\n-- a healthy row --')
    print('id:', sample['id'])
    print('answer:', sample['answer'][:90], '...')
    print('contexts:', len(sample['contexts']), 'chunks')

    print('\n-- the seeded failures --')
    for r in rows:
        if r['id'] in ('g005', 'g015', 'g019'):
            print(r['id'], '| contexts:', len(r['contexts']), '| note:', r['notes'][:40])

    faith = score_faithfulness(rows)
    print(faith)

    print('\nlowest-scoring rows:')
    for rid, sc in sorted(faith.per_row.items(), key=lambda kv: kv[1])[:5]:
        print(f'  {rid}: {sc:.3f}')

    print()
    for name, res in score_all(rows).items():
        print(f'{name:<20} {res.aggregate:.3f}   ({res.mode})')

    print('\n-- offline task-success judge --')
    scores = [judge_offline(r['question'], r['answer'], r['ground_truth'], r['contexts']) for r in rows]
    for r, s in zip(rows, scores):
        if not s.task_success:
            print(f"  {r['id']}: FAIL  acc={s.accuracy} grd={s.groundedness} fmt={s.format}")
    print(f'task success rate: {sum(s.task_success for s in scores) / len(scores):.0%}')


if __name__ == '__main__':
    main()
