"""Deterministic cost comparison for changing agent call count.

This is a deliberately simplified scenario. Providers can cache conversation
history as well as the static system prompt; actual token usage must be read
from real provider responses. It does not measure wall-clock latency.
"""
import argparse
import json
from pathlib import Path
from .pricing import Rates, Usage, price


def model_workflow(calls: int, shared_tokens: int = 12000,
                   dynamic_tokens: int = 1500, history_growth: int = 1100,
                   output_per_call: int = 350) -> dict:
    if calls < 1 or min(shared_tokens, dynamic_tokens,
                        history_growth, output_per_call) < 0:
        raise ValueError("Invalid counts")
    charges = []
    for i in range(calls):
        usage = Usage(
            cache_write=shared_tokens if i == 0 else 0,
            cache_read=shared_tokens if i > 0 else 0,
            regular_input=dynamic_tokens + history_growth * i,
            output=output_per_call,
        )
        charges.append(price(usage, Rates()))
    return {"calls": calls, "usd": round(sum(charges), 7),
            "per_call_usd": [round(x, 7) for x in charges],
            "scenario_only": "Stable 12k-token cache, otherwise uncached growing history"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-calls", type=int, default=16)
    p.add_argument("--optimized-calls", type=int, default=10)
    p.add_argument("--out", type=Path, default=Path("results/workflow_comparison.json"))
    a = p.parse_args()
    before = model_workflow(a.baseline_calls)
    after = model_workflow(a.optimized_calls)
    out = {"before": before, "after": after,
           "modeled_cost_reduction_pct": round((1-after["usd"]/before["usd"])*100, 3),
           "warning": "Synthetic pricing and token growth. Do not cite as Cloudflare measurements."}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2)+"\n", encoding="utf-8")
    print(f"{before['calls']} calls: ${before['usd']:.4f}")
    print(f"{after['calls']} calls: ${after['usd']:.4f}")
    print(f"Illustrative reduction: {out['modeled_cost_reduction_pct']:.1f}%")


if __name__ == '__main__':
    main()
