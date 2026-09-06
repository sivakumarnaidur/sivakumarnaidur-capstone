#!/usr/bin/env python3
"""Streamlit UI comparing the non-streaming (/chat) and streaming (/ask)
endpoints in api.py side by side, so the difference is visible directly.

Run (in one terminal, with the venv active):
    ./start_api.sh

Run (in another terminal, with the venv active):
    streamlit run src/streamlit_app.py
"""
import json
import queue
import threading
import time

import requests
import streamlit as st

CHAT_URL = "http://127.0.0.1:8000/chat"
ASK_URL = "http://127.0.0.1:8000/ask"
STREAM_URL = "http://127.0.0.1:8000/stream"
STRUCTURED_URL = "http://127.0.0.1:8000/chat/structured"
COST_META_MARKER = "\x00COST_META\x00"

st.set_page_config(page_title="Agentic AI & RAG Chat", page_icon="🤖", layout="wide")
st.title("🤖 Agentic AI & RAG Chatbot - Capstone")
st.caption("Ask a question and compare the non-streaming, streaming, true-streaming, and structured responses side by side.")

with st.expander("ℹ️ How does /chat/structured (tool calling) work?"):
    st.markdown(
        "Normally the model just returns free-form text. Tool calling instead lets "
        "you describe a function it can *request*, with a strict JSON-schema for "
        "the arguments -- so its \"answer\" comes back as validated, typed data "
        "instead of a prose blob you'd have to parse."
    )
    st.markdown(
        "1. The API sends the question plus a tool named `submit_answer`, whose "
        "argument schema comes straight from a Pydantic model (`AnswerPayload`).\n"
        "2. `tool_choice` forces the model to call that tool -- it can't just reply "
        "with plain text.\n"
        "3. The model's `content`/`confidence`/`sources` are combined with API-added "
        "bookkeeping (`cost_usd`, `retries`, `schema_version`) into one `Answer` "
        "model, re-validated by Pydantic before returning -- so the shape can "
        "never drift from what's documented in `/openapi.json`.\n"
        "4. If the tool call is missing or malformed, the API retries (up to a "
        "limit) and reports how many attempts it took in `retries`."
    )
    st.code(
        '{\n'
        '  "content": "string",\n'
        '  "confidence": "number (0-1)",\n'
        '  "sources": "string[]",\n'
        '  "cost_usd": "number",\n'
        '  "retries": "integer",\n'
        '  "schema_version": "string"\n'
        '}',
        language="json",
    )

question = st.text_input("Your question about Agentic AI & RAG:")


def fetch_chat(result_queue, question, start_time):
    """Call the blocking /chat endpoint and report the outcome via the queue."""
    try:
        response = requests.post(CHAT_URL, json={"message": question}, timeout=60)
        response.raise_for_status()
        elapsed = time.perf_counter() - start_time
        result_queue.put(("done", response.json(), elapsed))
    except requests.RequestException as e:
        result_queue.put(("error", str(e), None))


def fetch_ask(result_queue, question, start_time):
    """Call the streaming /ask endpoint, pushing each chunk as it arrives."""
    try:
        with requests.post(ASK_URL, json={"question": question}, stream=True, timeout=60) as response:
            response.raise_for_status()
            buffer = ""
            for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                if not chunk:
                    continue
                buffer += chunk
                if COST_META_MARKER in buffer:
                    text, meta_json = buffer.split(COST_META_MARKER, 1)
                    if text:
                        result_queue.put(("chunk", text, time.perf_counter() - start_time))
                    result_queue.put(("done", json.loads(meta_json), time.perf_counter() - start_time))
                    return
                result_queue.put(("chunk", buffer, time.perf_counter() - start_time))
                buffer = ""
        result_queue.put(("done", None, time.perf_counter() - start_time))
    except requests.RequestException as e:
        result_queue.put(("error", str(e), None))


def fetch_structured(result_queue, question, start_time):
    """Call the tool-calling-backed /chat/structured endpoint and report the outcome."""
    try:
        response = requests.post(STRUCTURED_URL, json={"message": question}, timeout=60)
        response.raise_for_status()
        elapsed = time.perf_counter() - start_time
        result_queue.put(("done", response.json(), elapsed))
    except requests.RequestException as e:
        result_queue.put(("error", str(e), None))


def fetch_true_stream(result_queue, question, start_time):
    """Call /stream, which forwards real OpenAI tokens (stream=True), pushing each chunk as it arrives."""
    try:
        with requests.post(STREAM_URL, json={"question": question}, stream=True, timeout=60) as response:
            response.raise_for_status()
            buffer = ""
            for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                if not chunk:
                    continue
                buffer += chunk
                if COST_META_MARKER in buffer:
                    text, meta_json = buffer.split(COST_META_MARKER, 1)
                    if text:
                        result_queue.put(("chunk", text, time.perf_counter() - start_time))
                    result_queue.put(("done", json.loads(meta_json), time.perf_counter() - start_time))
                    return
                result_queue.put(("chunk", buffer, time.perf_counter() - start_time))
                buffer = ""
        result_queue.put(("done", None, time.perf_counter() - start_time))
    except requests.RequestException as e:
        result_queue.put(("error", str(e), None))


def render_cost(container, meta):
    """Show model/token/cost metadata returned alongside an answer."""
    if not meta:
        return
    cost = meta.get("cost_usd")
    cost_str = f"${cost:.6f}" if cost is not None else "n/a"
    container.caption(
        f"🧾 Model: **{meta['model']}** | "
        f"Prompt: {meta['prompt_tokens']} tok | Completion: {meta['completion_tokens']} tok | "
        f"Cost: **{cost_str}**"
    )


def render_structured_cost(container, answer):
    """Show model/token/cost metadata plus retries/schema for the flat Answer shape."""
    if not answer:
        return
    render_cost(container, answer)
    container.caption(
        f"🔁 Retries: **{answer.get('retries', 0)}** | "
        f"Schema: **{answer.get('schema_version', 'n/a')}** | "
        f"Model: **{answer.get('model', 'n/a')}** | "
        f"Prompt: {answer.get('prompt_tokens', 0)} tok | Completion: {answer.get('completion_tokens', 0)} tok | "
        f"Cost: **{answer.get('cost_usd', 0.0):.6f}**"
        
    )


if st.button("Ask", disabled=not question.strip()):
    col_plain, col_stream, col_true_stream, col_structured = st.columns(4)

    with col_plain:
        st.subheader("⏳ Non-streaming (/chat)")
        st.caption("Client blocks until the full answer is ready, then it appears all at once.")
        chat_status = st.empty()
        chat_placeholder = st.empty()
        chat_caption = st.empty()
        chat_status.info("Waiting for the full answer...")

    with col_stream:
        st.subheader("⚡ Streaming (/ask)")
        st.caption("Fakes streaming: waits for the full answer, then replays it word by word.")
        ask_placeholder = st.empty()
        ask_caption = st.empty()

    with col_true_stream:
        st.subheader("🚀 True streaming (/stream)")
        st.caption("Real model-level streaming: OpenAI's stream=True forwards tokens as they're generated.")
        true_stream_placeholder = st.empty()
        true_stream_caption = st.empty()

    with col_structured:
        st.subheader("🧱 Structured (/chat/structured)")
        st.caption("Tool calling forces the reply into a fixed content/confidence/sources shape.")
        structured_status = st.empty()
        structured_placeholder = st.empty()
        structured_details = st.empty()
        structured_caption = st.empty()
        structured_status.info("Waiting for the structured answer...")

    # Fire all requests at the same time in background threads so their
    # progress can be rendered side by side in real time, instead of
    # waiting for one to finish before starting the next.
    start = time.perf_counter()
    chat_queue = queue.Queue()
    ask_queue = queue.Queue()
    true_stream_queue = queue.Queue()
    structured_queue = queue.Queue()
    threading.Thread(target=fetch_chat, args=(chat_queue, question, start), daemon=True).start()
    threading.Thread(target=fetch_ask, args=(ask_queue, question, start), daemon=True).start()
    threading.Thread(target=fetch_true_stream, args=(true_stream_queue, question, start), daemon=True).start()
    threading.Thread(target=fetch_structured, args=(structured_queue, question, start), daemon=True).start()

    chat_done = False
    ask_done = False
    true_stream_done = False
    structured_done = False
    ask_answer = ""
    ask_first_chunk_at = None
    true_stream_answer = ""
    true_stream_first_chunk_at = None
    chat_meta = None
    ask_meta = None
    true_stream_meta = None
    structured_meta = None

    while not (chat_done and ask_done and true_stream_done and structured_done):
        if not chat_done:
            try:
                kind, data, elapsed = chat_queue.get_nowait()
                if kind == "done":
                    chat_status.empty()
                    chat_placeholder.markdown(data["reply"])
                    chat_caption.caption(f"⏱️ Answer appeared after **{elapsed:.2f}s** (all at once)")
                    render_cost(chat_caption, data)
                    chat_meta = data
                    chat_done = True
                elif kind == "error":
                    chat_status.empty()
                    chat_caption.error(f"Could not reach the API at {CHAT_URL}: {data}")
                    chat_done = True
            except queue.Empty:
                pass

        if not ask_done:
            try:
                kind, data, elapsed = ask_queue.get_nowait()
                if kind == "chunk":
                    if ask_first_chunk_at is None:
                        ask_first_chunk_at = elapsed
                    ask_answer += data
                    ask_placeholder.markdown(ask_answer)
                elif kind == "done":
                    ask_caption.caption(
                        f"⏱️ First chunk after **{ask_first_chunk_at:.2f}s**, "
                        f"fully done after **{elapsed:.2f}s**"
                    )
                    render_cost(ask_caption, data)
                    ask_meta = data
                    ask_done = True
                elif kind == "error":
                    ask_caption.error(f"Could not reach the API at {ASK_URL}: {data}")
                    ask_done = True
            except queue.Empty:
                pass

        if not true_stream_done:
            try:
                kind, data, elapsed = true_stream_queue.get_nowait()
                if kind == "chunk":
                    if true_stream_first_chunk_at is None:
                        true_stream_first_chunk_at = elapsed
                    true_stream_answer += data
                    true_stream_placeholder.markdown(true_stream_answer)
                elif kind == "done":
                    true_stream_caption.caption(
                        f"⏱️ First token after **{true_stream_first_chunk_at:.2f}s**, "
                        f"fully done after **{elapsed:.2f}s**"
                    )
                    render_cost(true_stream_caption, data)
                    true_stream_meta = data
                    true_stream_done = True
                elif kind == "error":
                    true_stream_caption.error(f"Could not reach the API at {STREAM_URL}: {data}")
                    true_stream_done = True
            except queue.Empty:
                pass

        if not structured_done:
            try:
                kind, data, elapsed = structured_queue.get_nowait()
                if kind == "done":
                    structured_status.empty()
                    content = data["content"]
                    if data.get("sources"):
                        content += "\n\n**Sources:** " + ", ".join(data["sources"])
                    structured_placeholder.markdown(content)
                    structured_caption.caption(f"⏱️ Answer appeared after **{elapsed:.2f}s**")
                    with structured_details.container():
                        st.progress(data["confidence"], text=f"Model confidence: {data['confidence']:.0%}")
                        with st.expander("🔧 Raw submit_answer tool-call arguments"):
                            st.caption("This is the exact JSON the model returned as the tool call -- not parsed prose.")
                            st.json(data)
                    render_structured_cost(structured_caption, data)
                    structured_meta = data
                    structured_done = True
                elif kind == "error":
                    structured_status.empty()
                    structured_caption.error(f"Could not reach the API at {STRUCTURED_URL}: {data}")
                    structured_done = True
            except queue.Empty:
                pass

        if not (chat_done and ask_done and true_stream_done and structured_done):
            time.sleep(0.05)

    total_cost = sum(
        m["cost_usd"] for m in (chat_meta, ask_meta, true_stream_meta, structured_meta) if m and m.get("cost_usd") is not None
    )
    st.metric("💰 Total cost for this question (all four calls)", f"${total_cost:.6f}")

