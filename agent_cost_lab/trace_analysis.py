"""Aggregate normalized per-call traces. Supports JSONL (one request per line).

Expected fields: job_id, model, reviewer, tier, duration_ms, and usage containing
regular_input, cache_write, cache_read, output. Do not upload raw prompts.
"""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from .pricing import Rates, Usage, price


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * p
    left = int(pos)
    right = min(left + 1, len(ordered) - 1)
    return ordered[left] + (ordered[right] - ordered[left]) * (pos - left)


def analyze(rows: list[dict], rates: Rates = Rates()) -> dict:
    if not rows:
        raise ValueError("Need at least one request")
    job_map = defaultdict(list)
    group_map = defaultdict(list)
    tokens = dict(regular_input=0, cache_write=0, cache_read=0, output=0)
    for row in rows:
        if "job_id" not in row or not row["job_id"]:
            raise ValueError("Missing job_id")
        usage = Usage(**row["usage"])
        if row.get("duration_ms", 0) < 0:
            raise ValueError("Negative duration_ms")
        for key in tokens:
            tokens[key] += getattr(usage, key)
        detail = {"job_id": row["job_id"], "cost_usd": price(usage, rates),
                  "duration_ms": float(row.get("duration_ms", 0))}
        job_map[row["job_id"]].append(detail)
        group_map[(row.get("model", "unknown"), row.get("reviewer", "unknown"),
                   row.get("tier", "unknown"))].append(detail)
    jobs = [sum(item["cost_usd"] for item in group) for group in job_map.values()]
    # Sum durations as an illustrative upper bound, NOT the wall-clock job time
    # when a real workflow runs agent requests in parallel.
    total_cost = sum(jobs)
    cache_den = tokens["regular_input"] + tokens["cache_write"] + tokens["cache_read"]
    groups = []
    for (model, reviewer, tier), group in sorted(group_map.items()):
        groups.append({"model": model, "reviewer": reviewer, "tier": tier,
                       "calls": len(group), "usd": round(sum(r["cost_usd"] for r in group), 7)})
    return {
        "requests": len(rows), "jobs": len(job_map),
        "total_usd": round(total_cost, 7),
        "mean_usd_per_job": round(statistics.mean(jobs), 7),
        "p50_usd_per_job": round(percentile(jobs, .5), 7),
        "p95_usd_per_job": round(percentile(jobs, .95), 7),
        "token_totals": tokens,
        "cache_read_token_share": round(tokens["cache_read"] / cache_den, 5) if cache_den else None,
        "groups": groups,
        "notice": "No wall-clock job latency estimated; requests may overlap. Prices are scenario assumptions.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Normalized JSONL request trace")
    parser.add_argument("--out", type=Path, default=Path("results/trace_summary.json"))
    args = parser.parse_args()
    lines = [line for line in args.path.read_text(encoding="utf-8").splitlines() if line.strip()]
    report = analyze([json.loads(line) for line in lines])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
