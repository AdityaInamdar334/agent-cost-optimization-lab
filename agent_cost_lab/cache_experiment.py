"""Synthetic prompt-prefix caching simulator.

This models ONLY the accounting consequences of a matching, live prefix,
not actual provider internals. A cache key includes tenant and model,
and a timestamp-limited TTL is refreshed on every successful read.
"""
import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .pricing import Rates, Usage, price

SHARED_PROMPT = {
    "tools": ["read", "grep", "search", "diff"],
    "policy": "review source for correctness, security, and performance",
    "schema": "finding id, file, line, severity, explanation",
    "version": 1,
}


def canonical_bytes(payload: dict) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def prefix_key(tenant: str, model: str, payload: dict) -> str:
    digest = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    return f"{tenant}:{model}:{digest}"


@dataclass
class Cache:
    ttl_seconds: int = 300

    def __post_init__(self):
        if self.ttl_seconds <= 0:
            raise ValueError("TTL must be positive")
        self._seen: dict[str, int] = {}

    def access(self, key: str, at_seconds: int) -> bool:
        expiry = self._seen.get(key)
        hit = expiry is not None and at_seconds < expiry
        self._seen[key] = at_seconds + self.ttl_seconds
        return hit


def run_scenario(stable_prefix: bool, jobs: int = 96,
                 repository_count: int = 16,
                 interval_seconds: int = 45,
                 ttl_seconds: int = 300,
                 tenant: str = "example-tenant") -> dict:
    """First-request-only experiment. No live API calls or actual tokenization.

    All jobs have same logical shared policy. The *bad* prompt puts changing
    environment metadata before the cache breakpoint. The *good* prompt puts
    the breakpoint immediately after shared context.
    """
    if jobs <= 0 or repository_count <= 0 or interval_seconds < 0:
        raise ValueError("Invalid simulation setup")
    cache = Cache(ttl_seconds)
    rates = Rates()
    rows = []
    for job in range(jobs):
        repo_id = job % repository_count
        timestamp = job * interval_seconds
        env = {"repo": f"repo-{repo_id}", "date_bucket": timestamp // 86400}
        prefix = SHARED_PROMPT if stable_prefix else {**SHARED_PROMPT, "env": env}
        hit = cache.access(prefix_key(tenant, "example-model", prefix), timestamp)
        # Simulated token counts supplied by experiment, NOT model tokenizer.
        shared_tokens = 12_000
        dynamic_tokens = 2_500
        output_tokens = 450
        usage = Usage(
            regular_input=dynamic_tokens,
            cache_write=0 if hit else shared_tokens,
            cache_read=shared_tokens if hit else 0,
            output=output_tokens,
        )
        rows.append({
            "job_id": f"job-{job:03d}", "repo": env["repo"],
            "at_seconds": timestamp, "cache_hit": hit,
            "usage": usage.__dict__, "usd": round(price(usage, rates), 9),
        })
    total = sum(r["usd"] for r in rows)
    return {
        "scenario": "stable-prefix" if stable_prefix else "dynamic-before-breakpoint",
        "assumptions": {
            "jobs": jobs, "repositories": repository_count,
            "interval_seconds": interval_seconds, "ttl_seconds": ttl_seconds,
            "shared_prefix_tokens": 12_000, "dynamic_tokens": 2_500,
            "output_tokens": 450,
            "rates_usd_per_million": rates.__dict__,
            "tokenizer": "synthetic token counts; no provider calls",
        },
        "cache_hits": sum(r["cache_hit"] for r in rows),
        "first_call_cache_hit_rate": sum(r["cache_hit"] for r in rows) / jobs,
        "total_usd": round(total, 9),
        "average_usd_per_first_call": round(total / jobs, 9),
        "events": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=96)
    parser.add_argument("--repos", type=int, default=16)
    parser.add_argument("--interval", type=int, default=45)
    parser.add_argument("--ttl", type=int, default=300)
    parser.add_argument("--out", type=Path, default=Path("results/cache_comparison.json"))
    args = parser.parse_args()
    before = run_scenario(False, args.jobs, args.repos, args.interval, args.ttl)
    after = run_scenario(True, args.jobs, args.repos, args.interval, args.ttl)
    comparison = {
        "baseline": before, "optimized": after,
        "modeled_cost_reduction_pct": round(100 * (1 - after["total_usd"] / before["total_usd"]), 3),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    for item in (before, after):
        print(f"{item['scenario']}: first-call hits={item['cache_hits']}/{args.jobs}, "
              f"total modeled USD=${item['total_usd']:.4f}")
    print(f"Modeled reduction for these FIRST CALLS ONLY: "
          f"{comparison['modeled_cost_reduction_pct']:.1f}%")
    print("Model prices, token counts, and cache behavior here are assumptions.")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
