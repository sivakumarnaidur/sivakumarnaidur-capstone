#!/usr/bin/env bash
# Starts the FastAPI app (api:app) using the project's venv.
set -euo pipefail

cd "$(dirname "$0")"

source .venv/bin/activate

which uvicorn   # should print .../sivakumarnaidur-capstone/.venv/bin/uvicorn

exec uvicorn api:app --reload --app-dir src
