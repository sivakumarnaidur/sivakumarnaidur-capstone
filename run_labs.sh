#!/usr/bin/env bash
set -euo pipefail

# Activate the virtual environment
source .venv/bin/activate

mkdir -p docs/runs

python src/hello_llm.py "What is RAG in one sentence?" > docs/runs/01-what-is-rag.txt
python src/hello_llm.py "Why might an LLM hallucinate?" > docs/runs/02-why-hallucinate.txt
python src/hello_llm.py "Name three uses of vector databases." > docs/runs/03-vector-db-uses.txt

echo "Done. Outputs written to docs/runs/"
