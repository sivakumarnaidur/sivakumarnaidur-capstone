"""mp1/mp1_prompt_lab.py — MP1 prompt-engineering lab: structured extraction.

Compares four prompting strategies (zero-shot, few-shot, structured/role-based,
chain-of-thought) on the same task: extract {company_name, job_role,
minimum_years_experience} from a short job posting, as valid JSON. Each
strategy is run against every job posting in data/job_snippets.json, scored
against data/golden_set.json (exact-field accuracy + an LLM-judge rubric),
and compared on accuracy, parse rate, judge score, cost, and latency.

This makes real OpenAI API calls — it needs OPENAI_API_KEY (loaded from the
repo-root .env) and incurs a small per-run cost. See mp1_comparison.md /
mp1_writeup.md for recorded results and analysis.
"""
import asyncio
import json
import os
import re
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI

# ─── setup ───────────────────────────────────────────────────────────────
# Load OPENAI_API_KEY (and friends) from the repo-root .env file, same
# convention as src/config.py and src/hello_llm.py.
load_dotenv()

# Make sure your OPENAI_API_KEY is set in the environment
assert os.environ.get('OPENAI_API_KEY'), 'Set OPENAI_API_KEY first'

client = AsyncOpenAI()

MODEL = 'gpt-4o-mini'
JUDGE_MODEL = 'gpt-4o'
TEMPERATURE = 0.0

# Cost rates ($ per token) — from W4 cost.py
RATES = {
    'gpt-4o-mini': {'in': 0.15 / 1_000_000, 'out': 0.60 / 1_000_000},
    'gpt-4o':      {'in': 2.50 / 1_000_000, 'out': 10.00 / 1_000_000},
}

print('Setup complete.')

# ─── golden data ─────────────────────────────────────────────────────────
# Anchored to this file's own directory so the script works regardless of cwd.
DATA_DIR = Path(__file__).resolve().parent / 'data'

job_postings = json.loads((DATA_DIR / 'job_snippets.json').read_text())
job_postings_by_id = {row['id']: row for row in job_postings}
golden = {row['id']: row for row in json.loads((DATA_DIR / 'golden_set.json').read_text())}

print(f'Loaded {len(job_postings)} job postings, {len(golden)} golden entries.')
print('Sample job posting:', job_postings[0])


# ─── prompt strategies ───────────────────────────────────────────────────
def prompt_zero_shot(job_text: str) -> list[dict]:
    """Strategy 1 — zero-shot. Just ask, no examples, no persona."""
    return [{
        'role': 'user',
        'content': f"""
Extract the following fields from this job posting:
- company name
- job role
- minimum years of experience


Job Posting:
{job_text}
"""
    }]


def prompt_few_shot(job_text: str) -> list[dict]:
    """Strategy 2 — few-shot. Include 2-3 worked examples in the prompt."""
    examples = """
Example 1:
Job Posting: "We are hiring a Data Analyst at Northwind Ltd. Preferred 2 years of experience."
Output:
{"company_name": "Northwind Ltd", "job_role": "Data Analyst", "minimum_years_experience": 2}

Example 2:
Job Posting: "Acme Corp is looking for a Senior Software Engineer with at least 5 years of experience."
Output:
{"company_name": "Acme Corp", "job_role": "Senior Software Engineer", "minimum_years_experience": 5}

"""
    return [{
        'role': 'user',
        'content': f"""
Use the examples below to extract the requested fields from the target job posting.

{examples}

Task:
Extract the following fields from the job posting below:
- company name
- job role
- minimum years of experience


Job Posting:
{job_text}
"""
    }]


def prompt_structured(job_text: str) -> list[dict]:
    """Strategy 3 — structured / role-based. Use a system prompt with a persona and explicit JSON schema."""
    schema = """
{
  "company_name": string | null,
  "job_role": string | null,
  "minimum_years_experience": "integer | null"
}
"""
    return [
        {
            'role': 'system',
            'content': """
You are an expert recruiting analyst who can can indepth analysis of the postings. Your job is to read a job posting and extract three fields:
1) company name
2) job role
3) minimum years of experience

Return a valid JSON object that matches this schema exactly:
""" + schema + """

"""
        },
        {
            'role': 'user',
            'content': f"""
Job Posting:
{job_text}
"""
        },
    ]


def prompt_cot(job_text: str) -> list[dict]:
    """Strategy 4 — chain-of-thought. Ask the model to reason before answering."""
    return [
        {
            'role': 'user',
            'content': f"""
Extract these three fields from the job posting:
- company name
- job role
- minimum years of experience

1. Analyze the job posting carefully.
2. Extract these three fields from the job posting:
- company name
- job role
- minimum years of experience
3. Output valid JSON only with exactly these keys:
{{
  "company_name": "...",
  "job_role": "...",
  "minimum_years_experience": <integer or null>
}}


Job Posting:
{job_text}
"""
        }
    ]


STRATEGIES = {
    'Zero_shot': prompt_zero_shot,
    'Few_shot': prompt_few_shot,
    'Structured': prompt_structured,
    'CoT': prompt_cot,
}


# ─── response parsing ────────────────────────────────────────────────────
def parse_response(text: str) -> Optional[dict]:
    """Try to parse a JSON object out of the model's response. Return None if it doesn't parse.
    
    Hint: models sometimes wrap JSON in ```json ... ``` fences. Strip them first.
    """
    if text is None:
        return None

    cleaned = text.strip()
    if cleaned.startswith('```'):
        cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\s*```\s*$', '', cleaned, flags=re.IGNORECASE)

    # Try to isolate the JSON object if there is extra prose.
    match = re.search(r'\{.*\}', cleaned, flags=re.DOTALL)
    if match:
        cleaned = match.group(0)

    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


# ─── pipeline: one call, then all calls in parallel ──────────────────────
async def run_one(strategy_name: str, job_posting: dict) -> dict:
    """Run one strategy on one job posting. Return a dict with all the captured fields."""
    strategy_fn = STRATEGIES[strategy_name]
    messages = strategy_fn(job_posting['snippet'])

    t0 = time.perf_counter()
    response = await client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=TEMPERATURE,
    )
    elapsed = time.perf_counter() - t0

    raw_text = response.choices[0].message.content or ''
    parsed = parse_response(raw_text)

    prompt_tokens = response.usage.prompt_tokens if response.usage else 0
    completion_tokens = response.usage.completion_tokens if response.usage else 0
    cost_in = prompt_tokens * RATES[MODEL]['in']
    cost_out = completion_tokens * RATES[MODEL]['out']
    cost_usd = cost_in + cost_out

    return {
        'strategy': strategy_name,
        'job_posting_id': job_posting['id'],
        'raw_response': raw_text,
        'parsed_extraction': parsed,
        'latency_s': round(elapsed, 6),
        'cost_usd': round(cost_usd, 12),
        'prompt_tokens': prompt_tokens,
        'completion_tokens': completion_tokens,
    }


async def run_all() -> list[dict]:
    """Run all 10 × 4 = 40 calls in parallel. Use asyncio.gather."""
    tasks = [
        run_one(strategy_name, job_posting)
        for strategy_name in STRATEGIES
        for job_posting in job_postings
    ]
    return await asyncio.gather(*tasks)


# ─── scoring ──────────────────────────────────────────────────────────────
def score_accuracy(extracted: Optional[dict], gold: dict) -> int:
    """Compare 3 fields. Case-insensitive, whitespace-trimmed for strings. Return 0, 1, 2, or 3."""
    if not isinstance(extracted, dict):
        return 0

    def normalise(value, field_name=None):
        if value is None:
            return None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            value = int(value)
        value = str(value).strip().lower()
        if field_name == 'minimum_years_experience':
            value = value.replace('+', '').replace('years', '').strip()
        return value

    # The model is prompted to extract company_name/job_role/minimum_years_experience,
    # but golden_set.json stores the same facts under company/role/years_experience_required.
    FIELD_MAP = {
        'company_name': 'company',
        'job_role': 'role',
        'minimum_years_experience': 'years_experience_required',
    }

    score = 0
    for field, gold_field in FIELD_MAP.items():
        gold_value = gold.get(gold_field)
        extracted_value = extracted.get(field)
        if normalise(gold_value, field) == normalise(extracted_value, field):
            score += 1
    return score


async def score_llm_judge(job_text: str, extracted: Optional[dict], gold: dict) -> int:
    """Use gpt-4o as a judge. Return integer 1-4.

    Rubric (suggested):
      4 — all three fields correct
      3 — two of three correct, no fabricated data
      2 — one of three correct, or fabricated a field
      1 — none correct or unparsable
    """
    extracted_text = json.dumps(extracted, ensure_ascii=False) if isinstance(extracted, dict) else 'None'
    gold_text = json.dumps(gold, ensure_ascii=False)

    judge_messages = [
        {
            'role': 'system',
            'content': 'You are a strict evaluator for a structured extraction task. Return only an integer from 1 to 4.'
        },
        {
            'role': 'user',
            'content': f"""
Judge the extracted output against the gold answer.

Rubric:
- 4 = all three fields correct
- 3 = two of three correct, no fabricated data
- 2 = one of three correct, or fabricated a field
- 1 = none correct or unparsable

Job Posting:
{job_text}

Gold JSON:
{gold_text}

Extracted JSON:
{extracted_text}

Return only a single integer 1, 2, 3, or 4 and nothing else.
"""
        },
    ]

    response = await client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=judge_messages,
        temperature=TEMPERATURE,
    )
    text = response.choices[0].message.content or ''
    match = re.search(r'\b([1-4])\b', text)
    return int(match.group(1)) if match else 1


# ─── orchestration ────────────────────────────────────────────────────────
async def main() -> None:
    """Run every strategy × job posting through the model (`run_all`), score
    each result, and print a per-strategy summary table."""
    results = await run_all()
    print(f'Got {len(results)} results.')
    print(results[0])

    # Apply scoring to all 40 results
    scored = []
    for result in results:
        job_posting = job_postings_by_id[result['job_posting_id']]
        gold = golden[result['job_posting_id']]

        result['accuracy'] = score_accuracy(result.get('parsed_extraction'), gold)
        result['parse_success'] = 1 if result.get('parsed_extraction') is not None else 0
        result['llm_judge_score'] = await score_llm_judge(
            job_posting['snippet'], result.get('parsed_extraction'), gold
        )
        scored.append(result)

    print(f'Scored {len(scored)} results.')

    df = pd.DataFrame(scored)

    strategy_order = list(STRATEGIES.keys())

    summary = (
        df.groupby('strategy', sort=False)
        .agg({
            'accuracy': 'mean',
            'parse_success': 'mean',
            'llm_judge_score': 'mean',
            'cost_usd': 'sum',
            'latency_s': 'median',
        })
        .reindex(strategy_order)
    )

    summary['parse_success'] = summary['parse_success'] * 100
    summary['llm_judge_score'] = summary['llm_judge_score'] * (25 / 4)
    summary = summary.round({
        'accuracy': 2,
        'parse_success': 2,
        'llm_judge_score': 2,
        'cost_usd': 8,
        'latency_s': 3,
    })

    summary.columns = [
        'Accuracy (mean of 3)',
        'Parse rate (%)',
        'Judge score (out of 25)',
        'Total cost ($)',
        'Latency p50 (s)',
    ]

    # .style.format(...) only renders in a notebook; print a plain table here
    # so the summary is visible when this script is run from the terminal.
    print(summary.to_string(formatters={
        'Accuracy (mean of 3)': '{:.2f}'.format,
        'Parse rate (%)': '{:.2f}%'.format,
        'Judge score (out of 25)': '{:.2f}'.format,
        'Total cost ($)': '{:.8f}'.format,
        'Latency p50 (s)': '{:.3f}'.format,
    }))


if __name__ == '__main__':
    asyncio.run(main())