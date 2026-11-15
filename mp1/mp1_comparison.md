# MP1 Prompt Strategy Comparison

## Step 5 summary

The results below follow the same order as the `STRATEGIES` variable definition: `Zero_shot`, `Few_shot`, `Structured`, and `CoT`.

| Strategy | Accuracy (mean of 3) | Parse rate (%) | Judge score (out of 25) | Total cost ($) | Latency p50 (s) |
|---|---:|---:|---:|---:|---:|
| Zero_shot | 0.00 | 0.00% | 6.25 | 0.00029460 | 1.582 |
| Few_shot | 3.00 | 100.00% | 24.38 | 0.00047700 | 1.311 |
| Structured | 2.70 | 100.00% | 23.75 | 0.00041940 | 1.640 |
| CoT | 2.80 | 100.00% | 24.38 | 0.00042630 | 1.589 |

*Note: these are real, non-deterministic LLM calls — re-running produces slightly
different numbers each time, and this run's ranking is noticeably different from
the previous two runs (see below).*

## Quick takeaways

- Few_shot hit a perfect accuracy of 3.00 out of 3 this run — its best result yet, up from 2.90 last run. CoT held steady at 2.80, while Structured dropped to its weakest accuracy so far at 2.70 (down from 2.80).
- Few_shot and CoT are now tied for the highest judge score at 24.38 out of 25; Structured fell to the lowest judge score among the working strategies at 23.75 — the first run where Structured hasn't tied or led on judge score.
- Few_shot, Structured, and CoT all again achieved a 100.00% parse rate, while Zero_shot again achieved 0.00%.
- Latency ranking shifted again: Few_shot stayed fastest at 1.311s, Zero_shot and CoT were close behind (1.582s / 1.589s), and Structured was slowest this run at 1.640s — a reversal from two runs ago, when Structured was the fastest strategy.
- Zero_shot had the lowest cost at $0.00029460, but its 0.00% parse rate still makes that saving operationally meaningless. Few_shot remained the most expensive at $0.00047700, with Structured now the cheapest of the three working strategies at $0.00041940.

## Final conclusion from Step 5

Few_shot is now the clearest overall winner across the three runs recorded here: a perfect 3.00/3 accuracy, tied for the highest judge score (24.38/25), a 100.00% parse rate, and the fastest latency (1.311s) of any working strategy — it only loses on cost, where it remains the most expensive option. CoT is a solid, cheaper alternative, matching Few_shot's judge score (24.38) with slightly lower accuracy (2.80) and middling latency. Structured, which looked competitive or even fastest in earlier runs, had its weakest showing yet on every quality dimension (lowest accuracy and judge score among the working strategies, and now the slowest) — it's now mainly notable for being the cheapest of the three working options. Zero_shot remains unusable in production across all three runs: a 0.00% parse rate makes every downstream extraction fail regardless of its lowest cost. Across three runs, Few_shot has been the most consistently strong performer while Structured and CoT have traded places; this reinforces that a single run (or even three) shouldn't be treated as a final verdict — the rankings have shifted meaningfully each time with identical prompts and model settings, so averaging over many more runs would give a far more reliable comparison before committing to a strategy in production.