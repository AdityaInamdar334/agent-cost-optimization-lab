# Cloudflare Cut Its AI Code-Review Bill by 68%. The Real Optimization Was in the Agent, Not Just the Model.

## A hands-on guide to prompt-cache boundaries, token accounting, tool-call elimination, model routing, and testing whether cheaper AI agents still find real bugs

An AI coding agent reviews a pull request. It reads the diff, opens a few source files, produces a comment, and exits. The review looks like a single unit of work.

The invoice tells a different story.

Behind that comment might be sixteen language-model calls. Each call may resend the conversation. The first call may write a long system prompt to a paid cache. The second may read it. The third may append a file that should have been read earlier. A forbidden tool call might fail and cause the agent to try something else. After posting its final verdict, the agent might take one more model turn to summarize a result nobody will ever read.

None of these operations is visible in the final review. All of them can cost money.

Cloudflare ran into exactly this kind of problem with its internal AI code reviewer. In an October 19, 2026 engineering report, Ryan Skidmore described how its cost per review grew despite a broadly stable coordinator-and-specialist architecture. After auditing what the agents actually sent to model providers, restructuring the shared prompt prefix, removing wasted agent steps, and experimenting with model and reasoning-effort settings, the team reported a drop in average review cost from **$2.43 to $0.77**, approximately **68%**. Median completion time fell from **4 minutes 9 seconds to 1 minute 32 seconds**. [1]

There is a much more interesting lesson here than “use a cheaper model.”

**An agent's bill is a product of its execution graph, not merely the model's price per token.**

And there is a dangerous second lesson. A cheaper reviewer is not necessarily a better reviewer. Cloudflare found that its code-quality specialist, configured at high effort, retained roughly the same number of important findings while producing less noise. Its performance specialist, at medium effort, lost meaningful findings. [1]

This tutorial is both a case study and an engineering workshop. We will reconstruct the accounting, explain why an apparently excellent cache hit rate can coexist with rising costs, build a prefix-cache simulator, profile normalized request traces, model wasted tool turns, and evaluate the quality regression that cost dashboards cannot detect.

Everything in the companion Python lab runs locally, without a model API or secret keys. The experiments are **our synthetic scenarios**, clearly separated from Cloudflare's reported production measurements.

---

## Table of Contents

1. The $2.43 problem: What Cloudflare actually measured
2. What an agent really costs
3. Why a 97% cache hit rate can still be expensive
4. The first-request cache miss nobody noticed
5. Anatomy of a stable, cacheable prompt prefix
6. Build a cache-cost simulator in Python
7. Run the prefix experiment and interpret the results
8. The break-even economics of prompt caching
9. How unnecessary tools create hidden model calls
10. Measure the economics of a growing conversation
11. Model routing and reasoning-effort allocation
12. Why fewer findings do not necessarily mean lower quality
13. Build a quality evaluator for code-review agents
14. Reproducing a cost-and-quality tradeoff
15. Profiling real request logs without leaking prompts
16. Designing a rigorous, fair production experiment
17. Latency, retries, and cancellation as cost controls
18. Security, privacy, and isolation in prompt caching
19. A practical optimization sequence
20. What Cloudflare's 68% does and does not establish
21. Conclusion
22. References

---

## 1. The $2.43 Problem: What Cloudflare Actually Measured

Cloudflare's AI code reviewer is not one prompt sent to one model. It has a coordinator, review tiers, and specialist reviewers for concerns such as code quality, security, performance, and internal standards. The organization documented the broader code-review system in April 2026. [2]

A tiered review architecture is sensible. A trivial change should not necessarily consume the same inspection budget as a large, sensitive infrastructure modification. The workflow also has to fit naturally into development tools and merge-request review cycles.

But complexity grows quietly.

Cloudflare reported an average of about $1.19 per review in its earlier April write-up. By late August, costs were around $2.46. Part of the increase came from changes becoming larger and more demanding, but the team found that model and configuration choices were a major driver. [1]

At the time of the optimization project, the average cost during the two most expensive pre-optimization weeks was $2.43. After the work, the average was $0.77. The difference is $1.66 per review, or roughly 68.3% of the starting figure.

The time improvement was also considerable. Median review latency decreased from 249 seconds to 92 seconds. The 99th percentile fell from 15 minutes 20 seconds to 4 minutes 41 seconds. These are **Cloudflare's operational results**, not measurements from the lab in this tutorial. [1]

Those results are meaningful, but the comparison was not a single randomized trial where one variable changed. Cloudflare deployed multiple code and model changes across time. In its own analysis, the team cautioned that comparing weekday intervals around successive releases was not a statistically sound causal experiment. [1]

This matters whenever engineers try to copy a headline reduction. We should aim to reproduce the **mechanism** and evaluation discipline, not assert that a similar percentage is guaranteed.

### The most revealing baseline numbers

The report describes the code-quality reviewer making **16.6 model calls per run on average**. Across those calls, the reviewer wrote approximately **72,000 tokens to the cache** and read about **1.08 million tokens** back. Each subsequent call included the conversation history, so longer interactions caused repeated context processing. [1]

At the spending level examined, cached reads accounted for about **34%** of the bill, cache writes about **43%**, and output tokens about **23%**. Those were shares of monetary cost, not shares of raw token volume. [1]

That difference is vital. A million cheap cached reads can cost less than a much smaller number of fresh tokens. The accounting unit is dollars, not token count alone.

## 2. What an Agent Really Costs

A single model call has at least four meaningful token categories when the provider supports separately priced prompt caching:

- Regular input tokens, processed at the standard input price.
- Cache-write tokens, saved at a provider-specific write rate.
- Cache-read tokens, reused at a provider-specific read rate.
- Generated output tokens, which commonly have their own price.

Some providers also charge for separate reasoning-related operations, tools, storage, or extra-long context tiers. Billing schemas differ, and the four-bucket model must be adapted to the API actually used.

The accounting logic is simple if the buckets are mutually exclusive:

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Rates:
    regular_input: float = 3.00
    cache_write: float = 3.75
    cache_read: float = 0.30
    output: float = 15.00

@dataclass(frozen=True)
class Usage:
    regular_input: int = 0
    cache_write: int = 0
    cache_read: int = 0
    output: int = 0

def cost(usage: Usage, rates: Rates) -> float:
    return (
        usage.regular_input * rates.regular_input
        + usage.cache_write * rates.cache_write
        + usage.cache_read * rates.cache_read
        + usage.output * rates.output
    ) / 1_000_000
```

The prices are in **USD per million tokens**. They are illustrative scenario inputs, not a live quotation for any named model. We chose a cache write price of 1.25 times ordinary input and a read price of 0.1 times ordinary input because those multipliers illustrate the pricing regime relevant to Cloudflare's case. [1][3]

To use this model with real providers, first understand what their usage object actually means. A reported input total may already contain cached tokens. If you add it to cache-read tokens instead of subtracting the overlap, you double-count.

A useful cost profile should attribute each request to a **job**, a **specialist**, a **model version**, and a **review tier**. That lets us ask whether security reviews are expensive, whether large changes have a disproportionate number of retries, and whether a new model makes fewer calls because it resolves work faster or because it simply stops inspecting files.

For example, a code-quality reviewer that takes ten calls and a security reviewer that takes two should not be lumped together just because they happen inside one merge-request job.

### Price is not utilization

A provider may become 20% cheaper per output token while a new agent prompt causes the system to generate 60% more tokens. The actual bill rises.

Likewise, a model may cost the same per token as its predecessor while producing the same accepted result in fewer calls. Then model quality, control flow, and token cost interact.

Cloudflare observed this when changing models and effort settings. Some of its later models cost the same per token as the models they replaced, yet completed agent workflows with far fewer model calls. [1]

The correct unit of comparison is usually **cost per successful, adequately evaluated task**, not price per million input tokens.

## 3. Why a 97% Cache Hit Rate Can Still Be Expensive

Caching is often described as a magic performance multiplier. If most tokens are cache hits, shouldn't the bill be tiny?

Not necessarily.

Cloudflare reported a cache hit rate increasing from **85.7% in April to 97% in September** while the total number of tokens per review increased from approximately **0.91 million to 2.99 million**. [1]

The hit-rate percentage improved. The amount of processed context more than tripled.

Suppose an agent sends 100,000 input tokens per job, and 90,000 are cheap cached reads. Another agent sends 1 million tokens, and 970,000 are cached reads. The second system has a higher cache-hit fraction, but it still processes over ten times as many cached tokens.

And cached reads are not free.

This leads to two distinct metrics:

**Token-weighted cache-read share** tells us the fraction of counted input tokens that were billed as reads.

**First-request shared-prefix hit rate** asks a different question: when a new agent job starts, does it reuse the large block of shared instructions and tools that other jobs have already cached?

A system can have a high overall cache-read share because each individual conversation repeatedly reuses its own history while still failing to reuse the stable prefix between different jobs.

That was the crucial distinction in Cloudflare's investigation.

A third metric is useful as well: **effective dollars per useful finding**. A reviewer with impressive cache statistics and poor defect recall may still be the expensive choice.

## 4. The First-Request Cache Miss Nobody Noticed

Cloudflare found that its reviewers shared a substantial amount of identical setup material. The same tools, rules, and agent instructions appeared at the start of many sessions. Depending on the reviewer, these common portions contained roughly **11,000 to 21,000 tokens**. [1]

That sounds ideal for prefix caching.

But on the first model request of a new session, a cached prefix was being read only **14% to 26%** of the time.

The cause was subtle and entirely structural.

In the older OpenCode harness, an agent prompt was followed by an environment section with the current working directory and date. The cache breakpoint came **after** this changing environment content.

Conceptually, the request looked like this:

```text
System block
  Common tool definitions
  Common review instructions
  Working directory: /workspace/repo-001
  Date: 2026-10-30
  [CACHE BREAKPOINT]

Review request
  Repository-specific diff
  Reviewer-specific instructions
```

The prefix that the provider could cache included values that changed across repositories and sessions.

Even if the first 12,000 tokens were identical, the provider was being asked to cache a longer prefix ending after the variable fields. That long prefix did not match the equivalent prefix from another repository.

Cloudflare changed the request construction to split the static and dynamic material at a marker, creating a separate breakpoint after stable instructions. Its first-request hit rate reportedly increased to **91% to 99.5%** after the fix. [1]

The change is not about generating shorter English instructions. It is about serializing the same instruction content into a **reusable prefix**.

This is one of the most important prompt-engineering lessons for agent systems: the exact shape and ordering of messages can materially affect cost without changing their meaning to a human reader.

## 5. Anatomy of a Stable, Cacheable Prompt Prefix

A practical cache-aware request separates relatively stable information from per-job data.

```text
COMMON PREFIX
  Tool names and schemas
  Security and review rules
  Shared output format
  Stable examples
  [CACHE BREAKPOINT]

JOB-SPECIFIC SUFFIX
  Repository identity
  Working directory
  Date and commit SHA
  Patch content
  Specialist instructions
  User request
```

This ordering serves two purposes.

First, many requests can reuse the exact common prefix.

Second, adding per-job information does not invalidate that shared entry.

Of course, stable does not mean globally reusable. A prompt may depend on customer permissions, internal standards, region, or other confidential material. The cache must respect the provider's documented isolation and retention behavior. Never put tenant-specific secrets into a prefix just to increase sharing.

### Tool schemas are part of the prompt

A tool definition is not free metadata. The model receives a representation of its name, description, and schema. Changing the tool list can invalidate the stable prefix or increase token cost.

That is why Cloudflare had to ensure that the list of available skills and tool descriptions did not vary before the cache breakpoint. [1]

One optimization can therefore reveal another problem. If a sub-agent is given tools it cannot use, those definitions consume prompt tokens and create the opportunity for wasteful attempted calls.

### Message reordering must preserve semantics

The safe objective is **stable shared instructions first, dynamic context later**, while preserving instruction hierarchy and the provider's supported prompt structure.

Do not move an untrusted repository document into a higher-priority system message to get a cache hit. Do not flatten user content and trusted instructions into one unreviewed block. Do not change the permission model silently.

A cost optimization that weakens prompt-injection boundaries is not an optimization worth shipping.

## 6. Build a Cache-Cost Simulator in Python

Now we can isolate the exact economics in a local experiment.

The companion project contains `agent_cost_lab/cache_experiment.py`. It models an in-memory prefix cache keyed by tenant, model, and a digest of the canonical serialized prompt prefix.

The cache expires after a configured interval and refreshes its simulated expiry after each hit.

Its central idea looks like this:

```python
import hashlib
import json


def canonical_bytes(payload: dict) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def prefix_key(tenant: str, model: str, payload: dict) -> str:
    digest = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    return f"{tenant}:{model}:{digest}"
```

The hash ensures that two logically identical serialized objects in our test harness receive the same key. Tenant and model identifiers scope the key so the simulation never treats unrelated tenants as sharing a cache entry.

**This is not a representation of how any provider implements its internal prefix cache.** Real provider matching depends on tokens, supported breakpoints, routing, retention, and other implementation details. Our fingerprint is an audit aid.

The cache itself is intentionally simple:

```python
class Cache:
    def __init__(self, ttl_seconds=300):
        self.ttl_seconds = ttl_seconds
        self.expiry = {}

    def access(self, key, now):
        deadline = self.expiry.get(key)
        hit = deadline is not None and now < deadline
        self.expiry[key] = now + self.ttl_seconds
        return hit
```

A fresh key incurs a cache-write charge. A matching live key incurs a cache-read charge.

We then generate a sequence of independent code-review jobs. All share the same rules and tool definitions, but each belongs to a different repository in a rotating set.

The **bad** condition hashes the shared instructions together with the changing repository metadata. The **good** condition places the cache boundary immediately after the stable part.

Both conditions use identical modeled input and output token counts, so the experiment isolates the breakpoint-placement effect. There is no stochastic language-model response to confound the result.

## 7. Run the Prefix Experiment and Interpret the Results

Clone or extract the companion project and run:

```bash
python -m agent_cost_lab.cache_experiment \
    --jobs 96 \
    --repos 16 \
    --interval 45 \
    --ttl 300 \
    --out results/cache_comparison.json
```

The default experiment assumes a common prefix of **12,000 tokens**, another **2,500 dynamic input tokens** per first request, and **450 output tokens**. Jobs arrive every 45 seconds, move between 16 repositories, and use a 300-second sliding cache lifetime.

We use the illustrative rates from Section 2: $3.00 per million ordinary input tokens, $3.75 per million cache-write tokens, $0.30 per million cache-read tokens, and $15.00 per million output tokens.

The actual local results were:

**Dynamic content before the breakpoint:** 0 cache hits across 96 first requests. Total modeled cost was **$5.6880**.

**Stable prefix before the dynamic content:** 95 hits across 96 first requests. Total modeled cost was **$1.7550**.

For these first requests, the scenario's modeled cost reduction was **69.1%**.

This is an interesting result because its percentage is close to Cloudflare's reported 68%. But that similarity is coincidental.

Cloudflare measured its entire code-review workflow, including multiple specialists, model changes, output tokens, retries, and several operational improvements. We measured **one synthetic first-call component**, using invented token volumes and prices. The figures must not be presented as independent confirmation of Cloudflare's production savings.

You should also test counterexamples.

What happens if the same cacheable prefix is requested every six minutes with a five-minute TTL? The simulator correctly records no hits in that case.

What happens if a tenant uses different model identifiers? Its entries remain separate in our simulation.

What happens if shared instructions change every deployment? The key changes, so the first request after a change must write a new prefix.

These are useful sensitivity tests because they make the mechanism falsifiable.

### Why stable fingerprints do not guarantee a cache hit

Run the prefix audit tool:

```bash
python -m agent_cost_lab.prefix_audit \
    examples/repo-one_request.json \
    examples/repo-two_request.json \
    --stable-messages 1
```

Both example requests start with the same system instructions and then diverge in repository-specific content.

The tool reports identical fingerprints for the selected stable prefix but different fingerprints for the complete serialized request.

A matching local hash means the content we compared is stable. It does **not** prove that the provider used a cache breakpoint at the same location or charged the next request as a read. Only actual response usage data can establish billing behavior.

## 8. The Break-Even Economics of Prompt Caching

Caching is not automatically cheaper for every isolated request.

A cache write may cost more than ordinary input. In Cloudflare's described pricing arrangement, the relevant write rate was 1.25 times standard input and the read rate 0.1 times. [1][3]

Imagine a prefix that costs one dollar to process normally, purely for illustration.

Writing it once to cache costs $1.25.

Reading it once costs $0.10.

Writing it and reading it once therefore costs $1.35, compared with $2.00 for two ordinary uncached requests.

The benefit grows with repeated reads as long as the entry remains live.

But that simple example hides important real-world constraints.

If the shared prefix is too short to meet the provider's eligibility threshold, caching may not occur.

If the instructions change frequently, the system repeatedly pays the write cost.

If requests rarely arrive within the retention window, the hit rate may be poor.

If writing and reading the prefix adds coordination delay, total user-facing latency might increase even while token cost falls.

And if a provider uses a different cache billing model, the arithmetic changes.

For these reasons, production systems should measure **real cache-write tokens**, **real cache-read tokens**, **first-request cache behavior**, and **cost per completed job**. Do not infer savings from a local hash alone.

## 9. How Unnecessary Tools Create Hidden Model Calls

The prefix fix was the single biggest individual saving described by Cloudflare, but it was not the entire program.

The team found several cases where agent orchestration itself produced needless work. [1]

One was a permission mismatch. Sub-reviewers were given access to a tool intended for the final review-publishing step. Many tried to call it, received a server-side rejection, and then made another model call to recover. Removing the unavailable tool eliminated the failed attempt and reduced the tool-schema context by roughly 2,900 tokens per request.

Another was serialized file reading. In many code-quality turns the agent opened one file, processed that result, and only then opened another. The team changed the workflow so reviewers could receive shared context and patches earlier and read relevant files together where appropriate. That reduced round trips.

A third was redundant post-completion work. After posting a review, the old agent would receive its own result and sometimes run an additional summarization turn that no user saw. Cloudflare estimated about seven cents per job could be saved by stopping at the actual terminal action.

A fourth was waste during re-reviews. A substantial share of jobs were second or later passes over a merge request. Some specialist reviewers had no new relevant changes to examine. Avoiding those unnecessary passes reduced work.

Finally, canceled jobs sometimes kept running. The report describes checking job status periodically to stop work that developers had already canceled. [1]

None of these requires inventing a new Transformer architecture.

They require inspecting the execution graph and deciding which nodes provide real value.

### A state-machine view of an agent

A coding agent might execute this sequence:

```text
START
  |
  v
Classify risk tier
  |
  v
Prepare shared review context
  |
  v
Launch required specialists
  |
  +--> Read patch and files
  |       |
  |       v
  |    Analyze findings
  |
  v
Merge, validate and publish verdict
  |
  v
STOP
```

The old failure modes resemble extra edges added to this graph: attempting a forbidden tool, rerunning a timed-out specialist without differentiating timeout from provider failure, reviewing unchanged files, or continuing after publication.

The optimization question is not simply “Can we make each node cheaper?”

It is **“Does this node need to exist for this task?”**

## 10. Measure the Economics of a Growing Conversation

Long agent sessions accumulate context. The next model call may include previous tool results, earlier reasoning, generated review notes, and metadata.

Some of this context may be cacheable. Some may be fresh. The exact charges depend on the provider's cache semantics.

Our second lab isolates how call count and growing context interact under one deliberately simplified billing scenario.

```bash
python -m agent_cost_lab.workflow_experiment \
    --baseline-calls 16 \
    --optimized-calls 10 \
    --out results/workflow_comparison.json
```

The scenario assumes each request shares a 12,000-token cached prefix. Dynamic context starts at 1,500 tokens and grows by another 1,100 regular-input tokens on each subsequent call. Each call generates 350 output tokens.

The simulator reports:

**Sixteen calls:** $0.6510 modeled cost.

**Ten calls:** $0.3234 modeled cost.

That is a **50.3% reduction under the specified synthetic assumptions**.

Again, these figures are neither Cloudflare measurements nor claims about any real provider. In production, earlier conversation tokens may themselves be eligible for caching. Some models use different output pricing, and tool calls have variable lengths.

The point of the experiment is structural: fewer interactions can eliminate not only the cost of a response, but also the future context that response would otherwise add.

This makes the cost of an unnecessary early model call larger than its immediate invoice line.

### Be careful with parallelism

If three reviewers execute concurrently, summing their model-call durations does not give you the job's wall-clock latency.

Similarly, eliminating one request from a noncritical parallel branch may save money without improving the user-visible completion time.

A proper trace should record both request-level timing and job-level start and end timestamps. The companion trace analyzer intentionally reports no invented job latency.

## 11. Model Routing and Reasoning-Effort Allocation

Once the workflow is efficient, model selection becomes more informative.

Cloudflare compared different models and effort settings rather than assuming a more expensive model always offered better practical value. It also ran randomized comparisons for some decisions. [1]

Its October findings include an especially useful example. A newer Sonnet model delivered substantially lower per-run costs in code-quality, performance, and security specialist workflows even though the compared models had the same per-token price in that test. The newer model often made fewer calls and generated fewer tokens. [1]

This is a crucial distinction.

A model can be a more efficient **agent** without having a lower advertised input-token rate.

But fewer calls can also signal less thorough investigation.

Cloudflare noticed that at medium effort, its code-quality reviewer frequently finished in a single call without opening additional repository files. That could be acceptable for trivial changes. It could also be concerning for a large patch whose correctness depends on surrounding code.

At high effort, the reviewer made more calls and found more serious issues. In one short randomized comparison reported by Cloudflare, high effort cost about 61% more per run and produced approximately 58% more critical and warning findings. The team investigated whether those additional findings were genuinely useful before deciding to use high effort for code quality. [1]

### A practical routing policy

For an engineering team, a useful starting point is conditional effort allocation:

A small documentation change could use a lightweight path.

A routine, well-tested source change might receive a normal review.

A high-risk authentication or concurrency change should trigger a stronger reviewer, more context, and possibly human escalation.

The policy should be evaluated against known defect categories, not chosen solely by the cost dashboard.

A routing classifier can be wrong. If it sends a critical security patch to the cheapest path, the apparent saving may be an expensive failure.

For that reason, track routing errors and missed severe defects by tier.

## 12. Why Fewer Findings Do Not Necessarily Mean Lower Quality

Suppose an older AI reviewer makes ten critical findings. A newer version makes six.

Is the new version worse?

You cannot know from counts alone.

Maybe four of the earlier findings were duplicates or minor style issues labeled critical.

Maybe the new reviewer genuinely misses four exploitable defects.

Both situations can produce identical summary metrics.

Cloudflare attempted to separate these explanations using two signals. First, it checked whether an engineer changed the lines flagged by a finding in the next push. Second, it used blinded model judges from different providers to classify a sample of findings as important, minor, duplicate, incorrect, or unverifiable, and to evaluate severity calibration. [1]

The author-change signal is useful but imperfect. Code can change for unrelated reasons. A model judge can also be wrong, and findings that require repository-wide context may be difficult to verify from a diff alone.

The reported quality results were nuanced.

For code quality at high effort, the judges found a broadly similar number of important issues per review before and after optimization. Serious findings were less often overstated. Authors also acted on a larger fraction of the serious findings.

For performance review at medium effort, however, the judges identified a meaningful loss in important findings. Cloudflare described the impact as roughly one additional missed important performance issue every 24 full-tier reviews. The reported security decrease was less conclusive statistically. [1]

That is why **number of findings**, **precision**, **recall**, **severity calibration**, and **developer action rate** should never be collapsed into one vague quality score.

## 13. Build a Quality Evaluator for Code-Review Agents

Our third lab demonstrates how an engineer can detect an apparently attractive but harmful optimization.

We start with a synthetic fixture of 24 review cases. Each case contains a set of labeled gold findings, with categories such as correctness, security, and performance.

We also define two sets of predicted finding identifiers, one for the baseline and one for an imagined optimized reviewer.

A record looks like this:

```json
{
  "case_id": "review-00",
  "gold": [
    {"id": "0-correctness", "category": "correctness"},
    {"id": "0-performance", "category": "performance"},
    {"id": "0-security", "category": "security"}
  ],
  "baseline": ["0-performance", "0-security", "0-noise"],
  "optimized": ["0-correctness", "0-spurious"]
}
```

The gold set is **hand-authored for teaching**, not derived from a real repository. No LLM generated these predictions.

The evaluator counts true positives, false positives, and false negatives. It computes precision and recall and breaks recall down by defect category.

This is a crucial difference from judging only whether a reviewer produces a plausible comment.

If the optimized system discovers only easy correctness issues but loses performance findings, category-level recall exposes that regression.

Run:

```bash
python -m agent_cost_lab.quality_eval \
    examples/synthetic_quality_cases.json \
    --out results/quality_summary.json
```

The results also include a **paired bootstrap**, resampling complete review cases rather than independent findings. That choice better reflects the fact that multiple findings from the same review may be correlated.

The bootstrap interval is descriptive uncertainty for the synthetic fixture, not evidence about a real model.

### What a real gold dataset requires

For a production-quality evaluation, create a fixed set of code changes with independently verified defects. Where possible, include unit tests, static analyses, security reproductions in isolated environments, and expert review. Mark each issue with a stable identity and severity.

Then let both agent configurations review the **same change at the same revision**.

When outputs differ in wording, normalize findings by the underlying defect rather than their exact sentence. Use blinded adjudication to avoid favoring the newer configuration.

Without a credible gold set, recall is not identifiable. You can calculate the precision of a sampled set of claims, but you cannot safely infer how many defects the reviewer missed.

## 14. Reproducing a Cost-and-Quality Tradeoff

Our synthetic fixture intentionally makes the optimized reviewer appear cleaner in one respect and worse in another.

The local evaluator reports:

The baseline has **89.7% precision** and **92.9% recall** across the labeled synthetic findings.

The optimized arm has **93.3% precision** and **75.0% recall**.

In other words, the optimized arm makes fewer false-positive claims but misses a larger fraction of known issues.

The category breakdown is more revealing. It retains all synthetic correctness findings but detects only half the synthetic performance findings and three quarters of the synthetic security findings.

The case-level paired bootstrap gives an illustrative 95% interval for the optimized-minus-baseline recall difference of approximately **-30.2 to -6.4 percentage points**.

This result was deliberately created by the fixture. It tells us that the scoring code can expose the quality tradeoff it was designed to reveal. It is **not a measured performance comparison between actual models**.

The practical lesson is immediate: an agent can have a better precision score, a lower invoice, and still be unacceptable for your workflow.

This is particularly important for security and reliability reviewers, where the cost of missing one severe issue may dwarf savings across thousands of ordinary reviews.

## 15. Profiling Real Request Logs Without Leaking Prompts

Synthetic experiments teach the mechanics. To optimize a real deployment, we need actual usage traces.

The companion `trace_analysis.py` module consumes a normalized JSON Lines file with one record per model request.

The minimum useful record contains a job identifier, reviewer name, model identifier, tier, duration, and the four non-overlapping token buckets.

```json
{
  "job_id": "review-001",
  "model": "example-model",
  "reviewer": "quality",
  "tier": "full",
  "duration_ms": 1834,
  "usage": {
    "regular_input": 2800,
    "cache_write": 12000,
    "cache_read": 0,
    "output": 410
  }
}
```

This is an example normalized record, not the literal schema of Anthropic, OpenAI, or another provider.

Real provider schemas differ. Some include cached tokens within the broader input count. Some report cache creation tokens under a separate field. Some expose usage only after a streaming response is complete. Others report reasoning tokens differently.

Create a provider-specific adapter that turns its actual response usage into a well-defined internal schema. Test the adapter with captured, redacted provider responses so that billing never double-counts the same token.

Then run:

```bash
python -m agent_cost_lab.trace_analysis \
    examples/synthetic_normalized_trace.jsonl \
    --out results/trace_summary.json
```

The local example contains 90 synthetic requests across 30 jobs. The script reports costs by job, reviewer, model, and tier, plus a cache-read token share.

The lab also provides a **paired comparison** command for two normalized JSONL files representing the same job population:

```bash
python -m agent_cost_lab.compare_traces \
    examples/synthetic_before.jsonl \
    examples/synthetic_after.jsonl \
    --out results/paired_cost_comparison.json
```

It matches the 96 synthetic job identifiers, aggregates every request within a job, and uses paired resampling of jobs to produce an illustrative uncertainty interval for the total cost reduction. It reports unpaired job counts so that missing tasks are not silently treated as savings.

Under the simulated prefix-cache conditions, the paired comparison reports **69.1% modeled savings** and a bootstrap interval of approximately **67.7% to 69.9%**. The interval describes variation in this deterministic fixture under resampling. It does **not** establish a causal result or uncertainty interval for any real provider.


Its example output totals **$1.7694** in modeled spending, or **$0.05898 per job on average**, using the same illustrative rates as our cache experiment.

That is another **synthetic accounting result**, not real provider spend.

### Instrument the right events

For production, add at least these fields to the event model:

- An opaque job ID that can join model requests to the overall task.
- A per-request sequence number and parent span ID.
- Specialist name, risk tier, model name, model revision, and effort setting.
- Prompt and tool-schema version identifiers.
- Cache-read and cache-write token counts from the provider.
- Input and output token counts after correct normalization.
- Actual provider bill, if available, alongside independently computed estimate.
- Request start and end timestamps, and separately the job start and end.
- Tool name, result type, error class, and retry reason.
- Whether the job finished with a verdict, timed out, was canceled, or failed.
- Stable fingerprints of selected static prompt regions, scoped to the appropriate security boundary.

Do not log raw customer source code or complete prompts merely to improve observability. In sensitive deployments, even a hash of a predictable secret may enable guessing. Use keyed fingerprints and appropriate access controls when the threat model requires them.

A cost profiler should be useful without becoming a new data-leakage surface.

## 16. Designing a Rigorous, Fair Production Experiment

Cloudflare's report is unusually candid about measurement limitations. It describes staggered releases, effort changes, and some randomized splits, rather than claiming every percentage came from one controlled experiment. [1]

We should be similarly careful.

A stronger independent study would use the following design.

### Freeze the task population

Select a representative set of merge requests spanning trivial, lite, and full review tiers. Keep the original code revisions fixed during evaluation.

If you compare last week's tiny documentation patches with this week's critical distributed-systems changes, the results are not comparable.

### Control model and prompt changes

Change one variable at a time where practical: prefix layout, tool availability, concurrency, stopping conditions, model, or effort.

If several changes must be released together, label the resulting experiment as a bundled intervention.

### Use paired or randomized assignments

Run both configurations against the same eligible cases in shadow mode, or randomize traffic with stable stratification by repository and risk tier.

For interactive agents, watch out for time-varying provider load, rate limits, hot caches, and differences in available tools.

If one arm benefits from a warm cache while the other consistently starts cold, the comparison may reflect cache order rather than superior request design.

### Capture a complete outcome

For each job, measure total billable cost, overall wall-clock duration, model calls, tool calls, failures, retries, cache behavior, and validated findings.

Measure p50, p95, and p99 job latency, not only average call latency.

Include canceled and failed jobs in the spend accounting. A system that occasionally burns tokens for ten minutes and returns no verdict can look artificially cheap if the dashboard only counts successful reviews.

### Evaluate quality with independent labels

Use expert-reviewed known defects, blinded model judging where appropriate, and actual test outcomes. Do not let the candidate agent grade itself without checks.

Report precision, recall, important findings per review, missed high-severity issues, and severity inflation. Break those metrics down by defect class and review tier.

### Express uncertainty honestly

Compute paired confidence intervals over review cases, not over each generated comment as if all comments were independent.

For small samples and rare critical defects, acknowledge that confidence intervals may be wide. A short experiment that finds zero misses does not prove zero risk.

### Define the launch rule in advance

Before running the test, define acceptable changes to cost, latency, false positives, severe-defect recall, and timeout rate.

A configuration should not automatically ship because cost fell 20% if it also violates a predeclared security-recall threshold.

This is cost optimization as an engineering experiment, not model shopping by instinct.

## 17. Latency, Retries, and Cancellation as Cost Controls

Cost and latency are related, but they are not interchangeable.

Fewer model calls often mean less waiting, especially when calls are sequential. However, cached-prefix coordination may add a delay to allow one request to populate a shared entry before other reviewers start.

Cloudflare described delaying secondary reviewers for a short period so they could reuse the first reviewer's cached prefix. That modest coordination overhead saved money under its workload. [1]

The right decision depends on a product's latency budget. For a background code-review job, a brief delay may be acceptable. For an interactive autocomplete operation, the same delay could be unacceptable.

Retry policy also matters.

A network transport failure, a provider rate limit, a model refusal, and a genuine computation timeout are different events. Retrying them identically can create retry storms.

A robust agent orchestrator should cap retries, classify failure reasons, avoid replaying non-idempotent side effects without checks, and propagate cancellation to underlying model requests where supported.

Even if you have a tight maximum request duration, you need a job-level deadline. Otherwise, a specialist can repeatedly consume its full allowance until the overall workflow is far slower and more expensive than expected.

Cloudflare's reported reliability changes provide a useful motivation. The rate of reviews ending without a verdict decreased, and security-review retries fell substantially after its changes. [1]

Do not treat reliability improvements as bonus metrics. They change the real cost per useful result.

## 18. Security, Privacy, and Isolation in Prompt Caching

It is tempting to maximize cache reuse at any cost. That is a mistake.

A production agent may process private source code, customer-specific policies, internal incidents, access tokens, and security findings. Those materials may have different retention rules and access boundaries.

The stable prefix should contain only information legitimately reusable at the intended scope.

A multi-tenant application must never infer that two customers should share privileged context just because a few initial instructions have the same text.

Our toy cache key explicitly includes tenant and model identifiers to reinforce that boundary. Provider implementations have their own isolation guarantees, which must be verified from current documentation. [3][4]

### Cache consistency is not semantic safety

A perfectly cached prompt can still contain stale or inappropriate instructions.

If organizational policy changes but the cache entry is reused under the old prompt version, the agent may operate on outdated constraints. Version prompt policy and invalidation explicitly.

Likewise, if a repository-specific rule is omitted from the job suffix because a developer wanted a stable prefix, the agent may miss a requirement.

Preserve policy correctness before optimizing token reuse.

### Cache a trusted prefix, not untrusted authority

Repository text, issue descriptions, and tool output should remain lower-trust data. A malicious comment in a source file must not become privileged system instructions because it was concatenated into a global prompt.

Cache design, prompt-injection defense, and tool-authorization design are parts of the same boundary problem.

The safest structure keeps trusted policy stable, attaches untrusted material with clear provenance, and enforces tool permissions outside the model.

## 19. A Practical Optimization Sequence

For a real team, an efficient order of operations is:

**First, establish honest telemetry.** Capture token usage by bucket, request count, actual job cost, latency percentiles, timeout rates, and quality outcomes. Without this, optimization is guesswork.

**Second, inspect the first model request.** Hash the exact stable prefix, check that it is followed by a provider-supported breakpoint, and verify that real provider usage records show cache reads for new sessions.

**Third, clean the execution graph.** Remove tools a reviewer cannot use, eliminate unnecessary final turns, batch safe parallel file reads, and stop canceled work. Measure whether the resulting workflow still covers the required evidence.

**Fourth, evaluate model and effort settings.** Keep the task set fixed, compare accepted findings, and increase effort where important recall degrades. Do not require every specialist to run at the same effort level.

**Fifth, optimize routing and re-review behavior.** Reuse validated results when inputs are unchanged and skip a specialist only when the eligibility rule has demonstrated acceptable miss rates.

**Sixth, rerun quality and reliability checks after every important change.** A new provider model can alter tool-use patterns even if the prompt has not changed.

**Finally, add ongoing guardrails.** Alert on sudden increases in calls per review, disappearance of first-request cache reads, rising timeout costs, shifts in model mix, or falling recall on a fixed canary suite.

This order matters because the highest-value optimizations may be structural, not incremental edits to prompt wording.

## 20. What Cloudflare's 68% Does and Does Not Establish

The production report establishes that Cloudflare observed a large reduction in average review cost and latency after a sustained optimization program. It describes concrete engineering interventions and provides additional observations about quality and reliability. [1]

It does **not** establish that every company can cut its agent bill by 68% by enabling caching.

The team changed prompt layout, agent tools, context delivery, re-review logic, model choices, effort levels, orchestration, and cancellation behavior.

Its reported attribution to code changes versus model changes is useful operational accounting, but it is not a clean causal decomposition from one randomized factorial experiment.

The model and workload ecosystem also changed during the measurement period. A result on a multi-specialist code-review system may not transfer to a customer-support chatbot, a long-running research agent, or an interactive IDE completion system.

And there is a quality caveat. In Cloudflare's own analysis, an optimized configuration missed meaningful performance findings at one effort level. The organization responded by investigating stronger settings rather than treating the lower invoice as definitive success. [1]

Our lab establishes narrower things:

1. Under a fixed synthetic billing model, moving variable data after a simulated cache breakpoint can significantly reduce first-call costs.
2. Under another fixed synthetic model, removing interactions from a growing conversation can reduce modeled spending.
3. A labeled quality-evaluation fixture can detect a reviewer that is more precise but misses more important issues.
4. The supplied code correctly computes those scenarios under the stated assumptions, as checked by its local tests.

We did **not** run live model API benchmarks, gather real provider cache usage, or reproduce Cloudflare's private reviewer.

A reproducible systems experiment begins with that distinction.

## 21. Conclusion: Optimize the Workflow, Then the Model

The expensive part of an AI coding agent is not always its language model.

It may be a cache breakpoint in the wrong position. A tool definition that should not exist. A file read repeated over multiple turns. A reviewer that runs again when nothing relevant changed. A canceled job that continues until completion. Or a model that takes too many steps to do something another model can do in one.

Those problems are invisible if your only metric is the advertised price per million tokens.

Cloudflare's report is valuable because it follows the actual execution path, from the first prompt bytes to the final review verdict, and asks which operations contribute to useful work.

But its quality analysis is just as important as its savings headline.

**The optimal agent is not the one that spends the least. It is the one that achieves the required standard of evidence and reliability at the lowest sustainable cost.**

The next time an AI agent bill rises, resist the urge to change models immediately.

Inspect the first request. Measure the cache. Count the calls. Trace the failures. Audit the tools. Then test whether the cheaper version still catches the bugs you actually care about.

That is where agent optimization becomes engineering rather than guesswork.

---

## 22. References

**[1] Ryan Skidmore, Cloudflare.** *Cutting our inference bill by 68% by being obsessive about optimization.* October 19, 2026. Primary case study, measurements, implementation changes, model/effort comparisons, and quality analysis.

https://blog.cloudflare.com/obsessive-inference-optimization/

**[2] Cloudflare Engineering.** *Orchestrating AI Code Review at scale.* April 2026. Background on its coordinator, reviewer workflow, and system design.

https://blog.cloudflare.com/ai-code-review/

**[3] Anthropic.** *Prompt caching.* Official provider documentation covering breakpoints, cache pricing, TTLs, usage reporting, and cache construction.

https://platform.claude.com/docs/en/build-with-claude/prompt-caching

**[4] OpenAI.** *Prompt caching.* Official provider documentation covering prefix matching, cache read and write accounting, TTLs, and request usage reporting. Provider behavior differs from Anthropic's, so examples should not be mixed indiscriminately.

https://developers.openai.com/api/docs/guides/prompt-caching

**[5] Cloudflare.** *Coding agents with Cloudflare AI Gateway.* Guidance for centralizing visibility over coding-agent request volume, latency, model selection, and cost.

https://developers.cloudflare.com/ai-gateway/integrations/coding-agents/

**[6] OpenTelemetry.** *Tracing concepts.* Background on spans, parent/child relationships, and measuring distributed execution paths.

https://opentelemetry.io/docs/concepts/signals/traces/

**[7] OpenTelemetry.** *Semantic conventions for generative AI systems.* Reference for representing model calls and usage as structured telemetry. Check current stability and provider-specific support before adopting any field names.

https://opentelemetry.io/docs/specs/semconv/gen-ai/

**[8] Sebastian Ruder.** *An Overview of Multi-Task Learning in Deep Neural Networks.* Background on balancing distinct learning objectives, useful context for why different specialists and quality dimensions should not be collapsed into a single cost metric.

https://arxiv.org/abs/1706.05098

---

*Reproducibility and attribution note, October 30, 2026: Cloudflare numbers in this article come from its published case study. The companion project is an independent standard-library Python laboratory. Its default token prices, call counts, cache TTL, artificial findings, and simulated workloads are educational assumptions. The lab ran locally and passed automated tests. It does not make live model API requests. All quality and cost results from the lab are labeled synthetic and must not be presented as Cloudflare measurements or an evaluation of actual code-review models.*
