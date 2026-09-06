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
STRUCTURED_URL = "http://127.0.0.1:8000/chat/structured"
COST_META_MARKER = "\x00COST_META\x00"

st.set_page_config(page_title="Agentic AI & RAG Chat", page_icon="🤖", layout="wide")
st.title("🤖 Agentic AI & RAG Chatbot")
st.caption("Ask a question and compare the non-streaming, streaming, and structured responses side by side.")

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
        "3. The model's reply is `{\"content\": ..., \"confidence\": ..., \"sources\": [...]}` "
        "as the tool's arguments, which the API re-validates with the same Pydantic "
        "model before returning it -- so the shape can never drift from what's "
        "documented in `/openapi.json`."
    )
    st.code(
        '{\n'
        '  "type": "function",\n'
        '  "function": {\n'
        '    "name": "submit_answer",\n'
        '    "parameters": {\n'
        '      "content": "string",\n'
        '      "confidence": "number (0-1)",\n'
        '      "sources": "string[]"\n'
        '    }\n'
        '  }\n'
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
    col_plain, col_stream, col_structured = st.columns(3)

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

    with col_structured:
        st.subheader("🧱 Structured (/chat/structured)")
        st.caption("Tool calling forces the reply into a fixed content/confidence/sources shape.")
        structured_status = st.empty()
        structured_placeholder = st.empty()
        structured_details = st.empty()
        structured_caption = st.empty()
        structured_status.info("Waiting for the structured answer...")

    # Fire both requests at the same time in background threads so their
    # progress can be rendered side by side in real time, instead of
    # waiting for /chat to finish before starting /ask.
    start = time.perf_counter()
    chat_queue = queue.Queue()
    ask_queue = queue.Queue()
    structured_queue = queue.Queue()
    threading.Thread(target=fetch_chat, args=(chat_queue, question, start), daemon=True).start()
    threading.Thread(target=fetch_ask, args=(ask_queue, question, start), daemon=True).start()
    threading.Thread(target=fetch_structured, args=(structured_queue, question, start), daemon=True).start()

    chat_done = False
    ask_done = False
    structured_done = False
    ask_answer = ""
    ask_first_chunk_at = None
    chat_meta = None
    ask_meta = None
    structured_meta = None

    while not (chat_done and ask_done and structured_done):
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

        if not structured_done:
            try:
                kind, data, elapsed = structured_queue.get_nowait()
                if kind == "done":
                    structured_status.empty()
                    answer = data["answer"]
                    content = answer["content"]
                    if answer.get("sources"):
                        content += "\n\n**Sources:** " + ", ".join(answer["sources"])
                    structured_placeholder.markdown(content)
                    structured_caption.caption(f"⏱️ Answer appeared after **{elapsed:.2f}s**")
                    with structured_details.container():
                        st.progress(answer["confidence"], text=f"Model confidence: {answer['confidence']:.0%}")
                        with st.expander("🔧 Raw submit_answer tool-call arguments"):
                            st.caption("This is the exact JSON the model returned as the tool call -- not parsed prose.")
                            st.json(answer)
                    render_cost(structured_caption, data)
                    structured_meta = data
                    structured_done = True
                elif kind == "error":
                    structured_status.empty()
                    structured_caption.error(f"Could not reach the API at {STRUCTURED_URL}: {data}")
                    structured_done = True
            except queue.Empty:
                pass

        if not (chat_done and ask_done and structured_done):
            time.sleep(0.05)

    total_cost = sum(
        m["cost_usd"] for m in (chat_meta, ask_meta, structured_meta) if m and m.get("cost_usd") is not None
    )
    st.metric("💰 Total cost for this question (all three calls)", f"${total_cost:.6f}")

