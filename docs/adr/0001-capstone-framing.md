# ADR-0001: Capstone Framing — Knowledge Assistant

- **Status:** Draft v1
- **Date:** 2026-08-30
- **Author:** Sivakumar Naidu

## Context

Right now anyone with a question about our internal docs has to dig through scattered markdown files and old wiki pages to find an answer, and half the time they just ask a teammate instead. This capstone is a small Q&A assistant that sits on top of that corpus so people can ask a question in plain English and get a sourced answer back instead of grepping through folders.

## Decision — Solution Framing Canvas

| Box | Your answer |
|-----|-------------|
| **Inputs** | A free-text question from the user, plus optional filters (e.g. limit to a doc folder) |
| **Outputs** | A short text answer with the source file(s) it was pulled from, so the user can go verify it |
| **Tools** | OpenAI chat completions for generation, a local retriever over the doc corpus (embeddings + similarity search), and `python-dotenv` for config |
| **Memory** | None for v1 — each question is answered independently. Might add last-N-turns memory later if follow-up questions become common |
| **Autonomy level** | Closer to the chatbot end — it answers questions but doesn't take actions or chain tool calls on its own yet |
| **Decision boundaries** | It can decide which passages to retrieve and how to phrase the answer. It can't edit source docs, and if it can't find a confident answer it should say so rather than guess |

## Consequences

- **Positive:** faster answers for common questions, answers are traceable back to a source doc, and the corpus stays the single source of truth instead of tribal knowledge
- **Negative / risks:** retrieval quality depends heavily on how the docs are chunked, and a bad chunking choice early on will be annoying to redo later. Also need to watch for the model answering confidently on things that aren't actually in the corpus
- **Things we'll re-visit:** whether to add conversation memory once we see real usage, and whether the corpus needs to grow beyond the initial small set before this is genuinely useful