#!/usr/bin/env bash
# Starts the Streamlit UI (streamlit_app.py) using the project's venv.
set -euo pipefail

cd "$(dirname "$0")"

source .venv/bin/activate

which streamlit   # should print .../sivakumarnaidur-capstone/.venv/bin/streamlit

exec streamlit run src/streamlit_app.py
