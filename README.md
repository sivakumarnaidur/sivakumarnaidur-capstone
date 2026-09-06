# Capstone — Knowledge Assistant

A 30-week build of a Q&A assistant over a small document corpus, completed as part of the *Agentic AI & RAG Engineering* programme.

## Corpus

<one-sentence description from Step 1a — what corpus, source>

## Project structure

```
.
├── requirements.txt        # Pinned Python dependencies for the whole project
├── run_labs.sh              # Runs hello_llm.py against 3 sample prompts, saves output to docs/runs/
├── start_api.sh              # Activates .venv and launches the FastAPI backend (uvicorn)
├── start_streamlit.sh         # Activates .venv and launches the Streamlit UI
├── docs/
│   ├── adr/                  # Architecture Decision Records (one per major design choice)
│   │   └── 0001-capstone-framing.md
│   └── runs/                 # Saved LLM outputs for evidence and reference
│       ├── 01-what-is-rag.txt
│       ├── 02-why-hallucinate.txt
│       └── 03-vector-db-uses.txt
└── src/
    ├── config.py              # Loads settings (OPENAI_API_KEY, CHAT_MODEL) from .env
    ├── api.py                  # FastAPI app: /chat (non-streaming) and /ask (streaming) endpoints
    ├── streamlit_app.py          # Streamlit UI that calls /chat and /ask side by side
    ├── hello_llm.py               # Minimal standalone script: one OpenAI chat completion call
    └── reference/                  # Standalone learning references (not used by the app)
        ├── async-await-ref.py        # async/await/asyncio concepts, runnable examples
        ├── async-with-without-await.py # await vs. no-await behavior demo
        ├── asyncio-gather-tricky.py    # asyncio.gather() edge cases, answered with runnable proof
        └── pydantic-model-ref.py       # Pydantic v2 reference: validation, nesting, ConfigDict, etc.
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the repo root with your OpenAI key:

```
OPENAI_API_KEY=sk-...
CHAT_MODEL=gpt-4o-mini   # optional, defaults to gpt-4o-mini
```

## Running the apps

| Script | What it does |
|---|---|
| `./start_api.sh` | Starts the FastAPI backend at `http://127.0.0.1:8000` (`--reload` on). Docs at `/docs`. |
| `./start_streamlit.sh` | Starts the Streamlit UI at `http://localhost:8501`. Requires the API to be running. |
| `./run_labs.sh` | Runs `src/hello_llm.py` against 3 sample prompts and writes results to `docs/runs/`. |

Run the API and the Streamlit UI in two separate terminals (both scripts activate `.venv` themselves):

```bash
./start_api.sh          # terminal 1
./start_streamlit.sh    # terminal 2
```

## Programs in `src/`

- **`config.py`** — `load_settings()` reads `OPENAI_API_KEY` and `CHAT_MODEL` from the environment/`.env` and returns a `Settings` object. Raises if the API key is missing.
- **`api.py`** — FastAPI app with:
  - `POST /chat` — blocks until the full LLM answer is ready, returns one JSON object.
  - `POST /ask` — streams the answer back chunk by chunk (`text/plain`).
  - `GET /health` — liveness check.
  - Run directly with `uvicorn api:app --reload --app-dir src`.
- **`streamlit_app.py`** — UI that fires `/chat` and `/ask` concurrently for the same question and renders both side by side, so the non-streaming vs. streaming behavior is visible in real time.
- **`hello_llm.py`** — smallest possible example: sends one question to the OpenAI chat API and prints the answer. Usage: `python src/hello_llm.py "your question"`.

### `src/reference/`

Standalone, runnable scripts kept for learning/reference — none of them are imported by the app:

- **`async-await-ref.py`** — tour of `asyncio.run`, coroutines, `gather`, tasks, timeouts, locks, async generators, and concurrent OpenAI calls.
- **`async-with-without-await.py`** — minimal demo contrasting calling a coroutine with vs. without `await`.
- **`asyncio-gather-tricky.py`** — answers tricky `asyncio.gather()` questions (sibling cancellation, `return_exceptions`, result ordering, outer cancellation, unpacking, timeouts) with runnable proof.
- **`pydantic-model-ref.py`** — Pydantic v2 reference: basic models, validation errors, field constraints, enums, nested models, custom validators, and `ConfigDict`.

Run any of them directly, e.g.:

```bash
python3 src/reference/pydantic-model-ref.py
```

## `docs/`

- **`docs/adr/`** — Architecture Decision Records, one markdown file per major design decision (e.g. [`0001-capstone-framing.md`](docs/adr/0001-capstone-framing.md) covers the initial solution framing: inputs/outputs, tools, memory, autonomy, and decision boundaries).
- **`docs/runs/`** — saved LLM outputs produced by `run_labs.sh`, kept as evidence/reference for later steps.

## Week 1

- [x] Set up repo + secrets discipline
- [x] Build `hello_llm.py` (Lab Step 2)
- [ ] Write ADR v1 (Lab Step 3)

