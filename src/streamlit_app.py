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
COST_META_MARKER = "\x00COST_META\x00"

st.set_page_config(page_title="Agentic AI & RAG Chat", page_icon="🤖", layout="wide")
st.title("🤖 Agentic AI & RAG Chatbot")
st.caption("Ask a question and compare the non-streaming vs streaming response side by side.")

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


if st.button("Ask", disabled=not question.strip()):
    col_plain, col_stream = st.columns(2)

    with col_plain:
        st.subheader("⏳ Non-streaming (/chat)")
        st.caption("Client blocks until the full answer is ready, then it appears all at once.")
        chat_status = st.empty()
        chat_placeholder = st.empty()
        chat_caption = st.empty()
        chat_status.info("Waiting for the full answer...")

    with col_stream:
        st.subheader("⚡ Streaming (/ask)")
        st.caption("Chunks render as they arrive, so text appears progressively.")
        ask_placeholder = st.empty()
        ask_caption = st.empty()

    # Fire both requests at the same time in background threads so their
    # progress can be rendered side by side in real time, instead of
    # waiting for /chat to finish before starting /ask.
    start = time.perf_counter()
    chat_queue = queue.Queue()
    ask_queue = queue.Queue()
    threading.Thread(target=fetch_chat, args=(chat_queue, question, start), daemon=True).start()
    threading.Thread(target=fetch_ask, args=(ask_queue, question, start), daemon=True).start()

    chat_done = False
    ask_done = False
    ask_answer = ""
    ask_first_chunk_at = None
    chat_meta = None
    ask_meta = None

    while not (chat_done and ask_done):
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

        if not (chat_done and ask_done):
            time.sleep(0.05)

    total_cost = sum(
        m["cost_usd"] for m in (chat_meta, ask_meta) if m and m.get("cost_usd") is not None
    )
    st.metric("💰 Total cost for this question (both calls)", f"${total_cost:.6f}")

