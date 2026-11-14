"""src/eval/golden.py — loader and validator for the capstone golden set.

Lineage: carried forward unchanged from W5. The golden set is the ground truth
that every later week's KPI is computed against — treat it as infrastructure,
not a one-off fixture. As of W10 it lives at `golden_set/capstone_v1.jsonl`
(the W5 set plus W7/W9 additions), 20 entries, ids g001..g020.

A golden entry has four fields:
  • id            — stable identifier (g001..g020). Referenced in eval runs
                    and across docs.
  • question      — exactly as a real user would phrase it.
  • ideal_answer  — full reference text. RAGAS uses this as `ground_truth`;
                    the judge uses it as the ideal answer. For open-ended
                    questions this may be a properties dict.
  • notes         — category (happy/harder/edge), source doc, why it matters.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

_ID_PATTERN = re.compile(r"^g\d{3}$")


class GoldenEntry(BaseModel):
    """One row of the golden set."""

    id: str = Field(..., description="Stable identifier g001..g999")
    question: str = Field(..., min_length=1)
    ideal_answer: str | dict[str, Any] = Field(...)
    notes: str = Field(default="", description="Category, source, why this matters")

    @field_validator("id")
    @classmethod
    def _id_shape(cls, v: str) -> str:
        if not _ID_PATTERN.match(v):
            raise ValueError(f"id must match g001..g999; got {v!r}")
        return v

    @property
    def ground_truth(self) -> str:
        """RAGAS expects a string ground_truth. Flatten dict ideals to text."""
        if isinstance(self.ideal_answer, dict):
            return json.dumps(self.ideal_answer)
        return self.ideal_answer


def load_golden_set(path: str | Path) -> list[GoldenEntry]:
    """Read a .jsonl file. Each non-blank, non-comment line is one entry."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Golden set not found at {p}")
    entries: list[GoldenEntry] = []
    with p.open() as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("//"):
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{p}:{lineno} — malformed JSON: {e.msg}") from e
            entries.append(GoldenEntry.model_validate(data))
    return entries


def validate_golden_set(path: str | Path) -> dict[str, Any]:
    """Validate the file. Returns a summary dict; raises on errors."""
    entries = load_golden_set(path)

    ids = [e.id for e in entries]
    if len(ids) != len(set(ids)):
        seen: set[str] = set()
        dupes = [i for i in ids if i in seen or seen.add(i)]  # noqa: B015
        raise ValueError(f"Duplicate ids in {path}: {sorted(set(dupes))}")

    def _bucket(note: str) -> str:
        low = note.lower()
        if low.startswith("happy"):
            return "happy"
        if low.startswith("harder") or low.startswith("multi"):
            return "harder"
        if low.startswith("edge"):
            return "edge"
        return "uncategorised"

    counts: dict[str, int] = {}
    for e in entries:
        counts[_bucket(e.notes)] = counts.get(_bucket(e.notes), 0) + 1

    return {"path": str(path), "n": len(entries), "ids": ids, "coverage": counts}

golden = load_golden_set('golden_set/capstone_v1.jsonl')
summary = validate_golden_set('golden_set/capstone_v1.jsonl')
print('entries:', summary['n'])
print('coverage:', summary['coverage'])
print('first:', golden[0].id, '—', golden[0].question)
