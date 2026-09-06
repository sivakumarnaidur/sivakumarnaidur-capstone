#!/usr/bin/env python3
"""
tiktoken reference: count prompt/completion tokens and estimate USD cost,
for the same prompt, across every current OpenAI text model.
Run: python3 src/reference/tiktoken-cost-ref.py
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from dataclasses import dataclass

import requests
import tiktoken
from urllib3.exceptions import InsecureRequestWarning


# ---------------------------------------------------------------------------
# 1. Corporate/classroom networks often intercept HTTPS with a self-signed
#    cert, which breaks tiktoken's one-time download of its BPE encoding
#    file. This context manager temporarily disables cert verification for
#    that download only, then restores the original requests.get.
# ---------------------------------------------------------------------------
@contextmanager
def allow_self_signed_tiktoken_download():
    """Retry the first tiktoken download behind classroom/corporate SSL proxies."""
    original_get = requests.get

    def get_without_cert_check(url, *args, **kwargs):
        kwargs.setdefault("verify", False)
        return original_get(url, *args, **kwargs)

    requests.get = get_without_cert_check
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InsecureRequestWarning)
            yield
    finally:
        requests.get = original_get


# ---------------------------------------------------------------------------
# 2. Pricing table: USD per 1,000,000 tokens, standard short-context tier.
#    Snapshot from https://developers.openai.com/api/docs/pricing as of
#    2026-09-05 -- OpenAI changes prices often, so re-check before relying
#    on this for real billing decisions.
# ---------------------------------------------------------------------------
@dataclass
class ModelPricing:
    input_per_1m: float | None
    output_per_1m: float | None


PRICING: dict[str, ModelPricing] = {
    # Flagship / current generation
    "gpt-5.6-sol": ModelPricing(4.00, 20.00),
    "gpt-5.6-terra": ModelPricing(2.00, 12.00),
    "gpt-5.6-luna": ModelPricing(0.20, 1.20),
    "gpt-5.5": ModelPricing(5.00, 30.00),
    "gpt-5.5-pro": ModelPricing(30.00, 180.00),
    "gpt-5.4": ModelPricing(2.50, 15.00),
    "gpt-5.4-mini": ModelPricing(0.75, 4.50),
    "gpt-5.4-nano": ModelPricing(0.20, 1.25),
    "gpt-5.4-pro": ModelPricing(30.00, 180.00),
    "gpt-5.2": ModelPricing(1.75, 14.00),
    "gpt-5.2-pro": ModelPricing(21.00, 168.00),
    "gpt-5.1": ModelPricing(1.25, 10.00),
    "gpt-5": ModelPricing(1.25, 10.00),
    "gpt-5-mini": ModelPricing(0.25, 2.00),
    "gpt-5-nano": ModelPricing(0.05, 0.40),
    "gpt-5-pro": ModelPricing(15.00, 120.00),
    # GPT-4.1 / GPT-4o generation
    "gpt-4.1": ModelPricing(2.00, 8.00),
    "gpt-4.1-mini": ModelPricing(0.40, 1.60),
    "gpt-4.1-nano": ModelPricing(0.10, 0.40),
    "gpt-4o": ModelPricing(2.50, 10.00),
    "gpt-4o-2024-05-13": ModelPricing(5.00, 15.00),
    "gpt-4o-mini": ModelPricing(0.15, 0.60),
    # Reasoning (o-series) models
    "o1": ModelPricing(15.00, 60.00),
    "o1-pro": ModelPricing(150.00, 600.00),
    "o3-pro": ModelPricing(20.00, 80.00),
    "o3": ModelPricing(2.00, 8.00),
    "o4-mini": ModelPricing(1.10, 4.40),
    "o3-mini": ModelPricing(1.10, 4.40),
    # Legacy GPT-4 / GPT-3.5 / base completion models
    "gpt-4-turbo-2024-04-09": ModelPricing(10.00, 30.00),
    "gpt-4-0613": ModelPricing(30.00, 60.00),
    "gpt-3.5-turbo": ModelPricing(0.50, 1.50),
    "gpt-3.5-turbo-1106": ModelPricing(1.00, 2.00),
    "gpt-3.5-turbo-instruct": ModelPricing(1.50, 2.00),
    "davinci-002": ModelPricing(2.00, 2.00),
    "babbage-002": ModelPricing(0.40, 0.40),
}


# ---------------------------------------------------------------------------
# 3. Token counting: each model family tokenizes with a specific encoding.
#    tiktoken maps most model names automatically; models released after
#    this tiktoken version (e.g. brand-new flagships) fall back to
#    o200k_base, the encoding OpenAI's newest models use.
# ---------------------------------------------------------------------------
def resolve_encoding(model: str) -> tiktoken.Encoding:
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("o200k_base")


def count_tokens(text: str, encoding: tiktoken.Encoding) -> int:
    return len(encoding.encode(text))


# ---------------------------------------------------------------------------
# 4. Cost estimate: prompt (input) and completion (output) tokens are
#    billed separately, each per-million-tokens.
# ---------------------------------------------------------------------------
def estimate_cost(prompt_tokens: int, completion_tokens: int, pricing: ModelPricing) -> float | None:
    if pricing.input_per_1m is None or pricing.output_per_1m is None:
        return None
    input_cost = prompt_tokens / 1_000_000 * pricing.input_per_1m
    output_cost = completion_tokens / 1_000_000 * pricing.output_per_1m
    return input_cost + output_cost


# ---------------------------------------------------------------------------
# 5. Build one row per model for a given prompt + an assumed completion size
#    (completion length isn't known until the model actually responds).
# ---------------------------------------------------------------------------
def build_cost_table(prompt: str, completion_tokens: int) -> list[tuple]:
    rows = []
    for model, pricing in PRICING.items():
        encoding = resolve_encoding(model)
        prompt_tokens = count_tokens(prompt, encoding)
        cost = estimate_cost(prompt_tokens, completion_tokens, pricing)
        rows.append((model, encoding.name, prompt_tokens, completion_tokens,
                     pricing.input_per_1m, pricing.output_per_1m, cost))
    return rows


# ---------------------------------------------------------------------------
# 6. Print as a fixed-width table (rates shown per 1M tokens).
# ---------------------------------------------------------------------------
def print_cost_table(rows: list[tuple]) -> None:
    header = ("Model", "Encoding", "Prompt", "Completion", "Input $/1M", "Output $/1M", "Cost USD")
    widths = [24, 13, 8, 12, 12, 13, 12]

    def fmt_rate(rate: float | None) -> str:
        return f"${rate:.2f}" if rate is not None else "-"

    def fmt_cost(cost: float | None) -> str:
        return f"${cost:.8f}" if cost is not None else "add rates"

    print("".join(h.ljust(w) for h, w in zip(header, widths)))
    for model, encoding, prompt_tok, completion_tok, in_rate, out_rate, cost in rows:
        cells = (model, encoding, str(prompt_tok), str(completion_tok),
                 fmt_rate(in_rate), fmt_rate(out_rate), fmt_cost(cost))
        print("".join(c.ljust(w) for c, w in zip(cells, widths)))


if __name__ == "__main__":
    prompt = "Explain retrieval augmented generation (RAG) in 2 bullet points. Include: cost, latency, accuracy."
    estimated_completion_tokens = 100

    # Pre-warm every encoding once, tolerating self-signed SSL proxies.
    with allow_self_signed_tiktoken_download():
        for model in PRICING:
            resolve_encoding(model)

    print(f"Prompt: {prompt}")
    print(f"Estimated completion tokens: {estimated_completion_tokens}\n")

    rows = build_cost_table(prompt, estimated_completion_tokens)
    print_cost_table(rows)
