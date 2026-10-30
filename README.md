# Agent Cost Optimization Lab

A reproducible, **offline** engineering workshop accompanying the Medium tutorial **Cloudflare Cut Its AI Code-Review Bill by 68%. Here's How to Audit Your Own Agent.**

This repository models three questions:

1. What does a request actually cost after ordinary input, cache writes, cache reads, and output are separated?
2. How does a variable field before a cache breakpoint destroy reuse across workspaces?
3. How do we detect an apparently cheaper review system that misses more labeled defects?

The code uses the Python standard library. It makes **no external API requests**, downloads no models, and never claims to reproduce Cloudflare's private data. All prices, token counts, and synthetic findings are scenario assumptions.

## Quick start

Python **3.10+** recommended.

```bash
python -m unittest discover -s tests -v
python -m agent_cost_lab.cache_experiment
python -m agent_cost_lab.workflow_experiment
python -m agent_cost_lab.trace_analysis examples/synthetic_normalized_trace.jsonl
python -m agent_cost_lab.compare_traces examples/synthetic_before.jsonl examples/synthetic_after.jsonl
python -m agent_cost_lab.quality_eval examples/synthetic_quality_cases.json
python -m agent_cost_lab.prefix_audit examples/repo-one_request.json examples/repo-two_request.json --stable-messages 1
```

Each command produces an inspectable artifact in `results/` or a console summary.

## Reproducible local results

The default **96-job synthetic first-call prefix experiment** uses 12,000 shared prefix tokens, 2,500 dynamic tokens, 450 output tokens, 16 rotating repositories, a 45-second interarrival gap, and a 300-second sliding TTL.

With changing repository information included before the cache breakpoint, the simulator counted 0 of 96 cache hits and modeled **$5.6880** in first-call costs. With a stable breakpoint before changing data, it counted 95 of 96 hits and modeled **$1.7550**, a **69.1% reduction for the simulated first calls only**. This number is coincidentally close to Cloudflare's 68% production result, but they measure entirely different things and should not be equated.

The illustrative **16-call versus 10-call** workflow simulation modeled **$0.6510 versus $0.3234**, a **50.3% scenario reduction**. It does not model real wall-clock latency or complete provider prefix-cache behavior.

The paired trace comparison includes 96 matching synthetic job IDs, with an illustrative **69.1% reduction** and a case-level bootstrap interval. This reflects only the simulator settings, not a live deployment or an independent confirmation of Cloudflare.

The synthetic 24-review quality fixture deliberately makes the optimized arm less complete. The baseline's manually defined finding-ID recall is **92.9%**, while the optimized arm reaches **75.0%**; the optimized arm has higher precision but misses more performance issues. No model generated these predictions. This demonstrates the scoring process, not an actual AI-agent regression.

## Pricing assumptions

`agent_cost_lab/pricing.py` uses hypothetical USD per million tokens:

- Ordinary input: $3.00
- Cache write: $3.75
- Cache read: $0.30
- Output: $15.00

These are **not a current price quotation for any named provider**. Cloudflare's 2026 report describes 1.25x input for a five-minute cache write and 0.1x input for a read for its deployment. Substitute real prices and **the usage buckets from the specific API response** before comparing production costs. Some providers include cached tokens in reported input totals, so do not count them twice.

## Project structure

```text
agent_cost_optimization_lab/
  agent_cost_lab/
    pricing.py               Price mutually exclusive token buckets
    prefix_audit.py          Fingerprint stable and full serialized prompt content
    cache_experiment.py      Simulate prefix writes, hits, expiry and cost
    workflow_experiment.py   Compare call-count scenarios
    trace_analysis.py        Summarize normalized request JSONL
    compare_traces.py        Compare paired job costs across JSONL arms
    quality_eval.py          Measure finding recall, precision and paired uncertainty
  examples/
    repo-one_request.json
    repo-two_request.json
    synthetic_normalized_trace.jsonl
    synthetic_before.jsonl
    synthetic_after.jsonl
    synthetic_quality_cases.json
  results/
    cache_comparison.json
    workflow_comparison.json
    trace_summary.json
    paired_cost_comparison.json
    quality_summary.json
  tests/test_lab.py
  ARTICLE.md
  README.md
```

## Normalize real request logs

`trace_analysis` expects JSONL records like this:

```json
{"job_id":"r-001","model":"example-model","reviewer":"quality","tier":"full","duration_ms":1834,"usage":{"regular_input":2800,"cache_write":12000,"cache_read":0,"output":410}}
```

Use **mutually exclusive** usage counts. If the provider returns total prompt tokens that already include cache reads, calculate `regular_input = total_input - cache_reads - cache_writes` only when documentation confirms that decomposition. Respect model-specific price tiers, cache TTLs, and reasoning-token billing. Duration per LLM call is not the same as the wall-clock duration of an entire agent job, especially with parallel reviewers.

The lab intentionally contains no keys or customer transcripts. For real deployments, collect non-sensitive request metadata and stable **scoped hashes** instead of raw prompts, review access controls and privacy policies, and retain an explicit pricing-version field alongside every run.

## Experimental design for real agent comparisons

Use the **same merge requests** under both configurations. Stratify by risk tier and change size. Pin prompt versions and model identifiers. Randomize assignment and order when appropriate. Preserve tool permissions and timeouts. Evaluate actual correctness and missing defects through expert-reviewed gold labels or blinded adjudication. Measure precision, recall and severity calibration separately for code quality, security and performance. Include review failures and retries in cost totals.

Record request-level input, cache writes, cache reads, output and price, but also job-level p50, p95 and p99 latency from real start/end events. The provided trace analyzer intentionally refuses to invent job latency from overlapping model-call durations.

The quality evaluator uses a paired bootstrap over review cases. A toy confidence interval cannot establish anything about a deployed model.

## What this lab does not do

- No model calls, live pricing lookup or real latency benchmarks
- No access to Cloudflare's internal trace data
- No simulation of actual provider GPU-cache internals, tokenization or scheduling
- No automatic correctness verification for generated code-review comments
- No claim that cache reuse is safe or possible across unrelated tenants
- No claim that reducing requests preserves quality

## Sources

- Cloudflare, Ryan Skidmore, *Cutting our inference bill by 68% by being obsessive about optimization* (October 19, 2026): https://blog.cloudflare.com/obsessive-inference-optimization/
- Cloudflare, *Orchestrating AI Code Review at scale*: https://blog.cloudflare.com/ai-code-review/
- Anthropic, *Prompt caching*: https://platform.claude.com/docs/en/build-with-claude/prompt-caching
- OpenAI, *Prompt caching*: https://developers.openai.com/api/docs/guides/prompt-caching

## License

The examples in this companion project are provided for educational use. Check repository license requirements before distributing them under a specific license.
