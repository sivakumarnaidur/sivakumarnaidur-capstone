#!/usr/bin/env python3
"""
OpenAI JSON-mode and tool-calling reference: force clean, parseable JSON
output, let the model request function calls with structured arguments, and
connect through corporate/classroom SSL proxies via the OS trust store.
Run: python3 src/reference/openai-json-mode-ref.py
"""
from __future__ import annotations

import json
import os
import re
import ssl

from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# 1. Without JSON mode, models often wrap JSON in prose ("Here is your
#    answer: ..."), which breaks json.loads() because it requires the ENTIRE
#    string to be valid JSON, not just a substring of it.
# ---------------------------------------------------------------------------
def naive_parsing_fails_example():
    raw = (
        "Here is your JSON answer:\n\n"
        '{"content": "RAG combines retrieval with generation.", '
        '"confidence": 0.9, "sources": []}\n\n'
        "Hope this helps!"
    )
    try:
        json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"json.loads failed as expected: {e}")


# ---------------------------------------------------------------------------
# 2. Fallback for models/APIs that don't support a JSON-mode response format:
#    extract the outermost {...} substring, then parse just that slice.
# ---------------------------------------------------------------------------
def extract_json_object(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("no JSON object found in text")
    return json.loads(match.group(0))


def extract_json_fallback_example():
    raw = (
        "Here is your JSON answer:\n\n"
        '{"content": "RAG combines retrieval with generation.", '
        '"confidence": 0.9, "sources": []}\n\n'
        "Hope this helps!"
    )
    print(extract_json_object(raw))


# ---------------------------------------------------------------------------
# 3. Corporate/classroom networks often intercept HTTPS with a certificate
#    issued by an internal CA. truststore.SSLContext delegates verification
#    to the OS trust store (which already trusts that CA), instead of
#    disabling verification like the tiktoken proxy workaround does.
# ---------------------------------------------------------------------------
def build_client_with_os_trust_store():
    import truststore
    from openai import DefaultHttpxClient, OpenAI

    return OpenAI(
        http_client=DefaultHttpxClient(verify=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)),
    )


# ---------------------------------------------------------------------------
# 4. response_format={"type": "json_object"} constrains the model to emit
#    only valid JSON -- no wrapping prose, no extraction step needed.
# ---------------------------------------------------------------------------
def json_mode_example():
    if not os.getenv("OPENAI_API_KEY"):
        print("skipped: set OPENAI_API_KEY to run this example")
        return

    client = build_client_with_os_trust_store()
    model = os.getenv("CHAT_MODEL", "gpt-4o-mini")

    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": (
                "Return JSON only with keys content, confidence, and sources. "
                "Question: What is RAG?"
            ),
        }],
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content or "{}"
    print(raw)
    print("parsed_keys:", list(json.loads(raw)))


# ---------------------------------------------------------------------------
# 5. Tool calling: unlike JSON mode (which shapes the model's text answer),
#    tool calling lets the model REQUEST an action -- it returns the tool
#    name + arguments (as a JSON string) instead of a final answer. The
#    model never executes anything; your code does, then reports the result
#    back so the model can use it in its next reply.
# ---------------------------------------------------------------------------
def get_weather(city: str) -> dict:
    """Stand-in for a real API call -- the model only ever sees this return value."""
    fake_data = {"paris": 18, "chennai": 31, "london": 15}
    return {"city": city, "temperature_c": fake_data.get(city.lower(), 20)}


TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current temperature for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]


def tool_calling_example():
    if not os.getenv("OPENAI_API_KEY"):
        print("skipped: set OPENAI_API_KEY to run this example")
        return

    client = build_client_with_os_trust_store()
    model = os.getenv("CHAT_MODEL", "gpt-4o-mini")
    messages = [{"role": "user", "content": "What's the weather in Chennai right now?"}]

    # Round 1: the model decides it needs the tool instead of answering directly.
    response = client.chat.completions.create(model=model, messages=messages, tools=TOOLS)
    tool_call = response.choices[0].message.tool_calls[0]
    args = json.loads(tool_call.function.arguments)
    print(f"model requested: {tool_call.function.name}({args})")

    # We -- not the model -- actually run the function.
    result = get_weather(**args)

    # Round 2: send the tool's result back so the model can phrase the final answer.
    messages.append(response.choices[0].message)
    messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result)})
    final = client.chat.completions.create(model=model, messages=messages, tools=TOOLS)
    print("final answer:", final.choices[0].message.content)


if __name__ == "__main__":
    print("\n--- 1. naive_parsing_fails_example ---")
    naive_parsing_fails_example()

    print("\n--- 2. extract_json_fallback_example ---")
    extract_json_fallback_example()

    print("\n--- 3/4. json_mode_example ---")
    json_mode_example()

    print("\n--- 5. tool_calling_example ---")
    tool_calling_example()
