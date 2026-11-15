# MP1 Prompt Lab Reflection

The strategies are discussed in the same order as the `STRATEGIES` variable: `Zero_shot`, `Few_shot`, `Structured`, and `CoT`.

## 1. Which strategy won, and on what dimension?

Across this (third) run, Few_shot is now the clearest overall winner on every quality dimension except cost: a perfect accuracy of 3.00 out of 3 fields, tied for the highest judge score (24.38 out of 25, with CoT), a 100.00% parse rate, and the fastest latency of any working strategy at 1.311 seconds. Its only weakness is cost — at $0.00047700 it remains the most expensive option. CoT matched Few_shot's judge score (24.38) with slightly lower accuracy (2.80) and mid-pack latency (1.589s), at a lower cost ($0.00042630). Structured posted its weakest results of the three runs so far: the lowest accuracy (2.70) and judge score (23.75) among the working strategies, and now the slowest latency (1.640s) — though it remains the cheapest of the three working options ($0.00041940). Zero_shot failed completely again: 0.00% parse rate and accuracy, and the lowest judge score (6.25 out of 25), despite having the lowest cost ($0.00029460).

Accuracy measures the mean number of correct values among `company_name`, `job_role`, and `minimum_years_experience`. Parse rate measures whether the response could be loaded as valid JSON. Judge score is the evaluator's 1-to-4 result rescaled to 25 points. These metrics still show why cost alone is not a useful winner criterion: Zero_shot is cheapest in every run, but it has never produced a single usable record, while Few_shot, Structured, and CoT all reliably parse and mostly extract correctly for a few cents' difference per thousand calls.

## 2. What surprised you?

The surprise this time wasn't Zero_shot — its complete failure (0.00% parse rate) has now been consistent across all three runs, so that's a pattern, not a fluke. The real surprise was Structured's reversal: in earlier runs it tied for the best judge score and was even the fastest strategy once, but in this run it posted the weakest accuracy, weakest judge score, and slowest latency of the three working strategies. Meanwhile Few_shot, which only tied for best accuracy in the first run, has now won outright on accuracy for two runs running and reached a perfect score this time, while also becoming the fastest strategy — a genuine trend, not noise.

This matters for how to read a single comparison run: with live, non-deterministic model calls, a strategy's standing can flip entirely between runs with zero prompt or setting changes, exactly as Structured and Few_shot swapped positions here. The only things that have held steady across all three runs are Few_shot's consistent strength and Zero_shot's consistent, complete failure — everything in between (Structured vs. CoT) has been noisier. For an HR knowledge assistant, this is the practical lesson: don't trust a single evaluation run to pick a production strategy; look for patterns that repeat across several runs before committing.

## 3. For your capstone domain, which strategy would you reach for first?

My capstone will be an HR knowledge enterprise assistant that helps employees and HR teams find reliable answers about policies, benefits, leave, onboarding, payroll processes, and compliance procedures. 

## 4. If you had another day, what would you try next?

First, I'd run this comparison a few more times to see whether Few_shot's lead holds up as a fourth and fifth data point, or whether it too eventually trades places the way Structured and CoT have — three runs is a trend, not proof. Then I'd build a domain-specific evaluation set for the HR assistant, covering policy version conflicts, location-specific benefits, employee-type differences, leave eligibility, payroll cut-off dates, and questions that should be escalated to HR. I would compare retrieval quality as well as answer quality by measuring citation correctness, groundedness, refusal accuracy, and whether the assistant identifies when the knowledge base does not contain enough information.