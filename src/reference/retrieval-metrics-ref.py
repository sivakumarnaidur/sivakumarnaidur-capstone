#!/usr/bin/env python3
"""
Retrieval evaluation metrics reference: Precision@K, Recall@K, F1, MRR,
NDCG@K, Context Relevance, and Context Coverage, worked through one shared
toy example so the numbers can be compared side by side.

Scenario: query "What is RAG?" -- the dataset has 5 documents that are truly
relevant to this query. The retriever returns a ranked top-5 list, where 3 of
those 5 results are relevant.

Run: python3 src/reference/retrieval-metrics-ref.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# 1. Core rank-based metrics: operate on a ranked list of 0/1 relevance
#    judgments (1 = relevant, 0 = not relevant), plus how many relevant docs
#    exist in the whole dataset (needed for recall).
# ---------------------------------------------------------------------------
def precision_at_k(relevance: list[int], k: int) -> float:
    """Of the top K shown, what fraction are relevant?"""
    top_k = relevance[:k]
    return sum(top_k) / k


def recall_at_k(relevance: list[int], k: int, total_relevant: int) -> float:
    """Of all relevant docs in the dataset, what fraction did the top K find?"""
    top_k = relevance[:k]
    return sum(top_k) / total_relevant


def f1_score(precision: float, recall: float) -> float:
    """Harmonic mean of precision and recall; 0 if both are 0."""
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def reciprocal_rank(relevance: list[int]) -> float:
    """1 / rank of the first relevant result; 0 if none found."""
    for i, rel in enumerate(relevance, start=1):
        if rel:
            return 1 / i
    return 0.0


def dcg_at_k(relevance: list[int], k: int) -> float:
    """Discounted cumulative gain: relevant hits count less the lower they rank."""
    return sum(rel / math.log2(i + 1) for i, rel in enumerate(relevance[:k], start=1))


def ndcg_at_k(relevance: list[int], k: int) -> float:
    """DCG normalized against the ideal ordering (all relevant docs ranked first)."""
    ideal = sorted(relevance, reverse=True)
    idcg = dcg_at_k(ideal, k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(relevance, k) / idcg


# ---------------------------------------------------------------------------
# 2. RAG-specific metrics: judge the *content* handed to the LLM, not just
#    which document IDs were fetched.
# ---------------------------------------------------------------------------
def context_relevance(chunk_relevance_scores: list[float]) -> float:
    """Average relevance (0-1, e.g. from an LLM judge) across retrieved chunks."""
    return sum(chunk_relevance_scores) / len(chunk_relevance_scores)


def context_coverage(required_facts: set[str], facts_found_in_context: set[str]) -> float:
    """Fraction of the facts needed to answer the question that are present."""
    return len(required_facts & facts_found_in_context) / len(required_facts)


@dataclass
class RetrievedDoc:
    doc_id: str
    is_relevant: bool


def main() -> None:
    # Ranked top-5 results for "What is RAG?"; 5 relevant docs exist in the
    # whole dataset (D1-D5), and this ranking surfaces 3 of them.
    results = [
        RetrievedDoc("D1", True),
        RetrievedDoc("D7", False),
        RetrievedDoc("D2", True),
        RetrievedDoc("D9", False),
        RetrievedDoc("D3", True),
    ]
    relevance = [1 if doc.is_relevant else 0 for doc in results]
    total_relevant_in_dataset = 5
    k = 5

    precision = precision_at_k(relevance, k)
    recall = recall_at_k(relevance, k, total_relevant_in_dataset)
    f1 = f1_score(precision, recall)
    mrr = reciprocal_rank(relevance)
    ndcg = ndcg_at_k(relevance, k)

    print(f"Ranked results: {[(d.doc_id, d.is_relevant) for d in results]}")
    print(f"Total relevant docs in dataset: {total_relevant_in_dataset}\n")
    print(f"Precision@{k}: {precision:.2%}")
    print(f"Recall@{k}:    {recall:.2%}")
    print(f"F1 Score:     {f1:.2%}")
    print(f"MRR:          {mrr:.3f}")
    print(f"NDCG@{k}:      {ndcg:.2%}")

    # Context relevance: an LLM judge scores each retrieved chunk 0-1 for
    # how relevant it is to the question (here it agrees with the binary
    # relevance labels above).
    chunk_scores = [1.0, 0.0, 1.0, 0.0, 1.0]
    print(f"\nContext Relevance: {context_relevance(chunk_scores):.2%}")

    # Context coverage: does the retrieved context contain every fact needed
    # to fully answer the question, regardless of which documents they came from?
    required_facts = {"retrieval", "generation", "grounding", "reduces_hallucination"}
    facts_found = {"retrieval", "generation", "grounding"}
    print(f"Context Coverage:   {context_coverage(required_facts, facts_found):.2%}")


if __name__ == "__main__":
    main()
