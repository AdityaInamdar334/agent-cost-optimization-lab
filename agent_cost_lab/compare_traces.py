"""Paired comparison of two normalized agent-request JSONL logs.

Pairing uses identical job IDs. Results concern token cost only; no inferences
about correctness or wall-clock duration are made from request timestamps.
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from .pricing import Rates, Usage, price


def cost_per_job(rows: list[dict], rates: Rates = Rates()) -> dict[str, float]:
    result = defaultdict(float)
    for row in rows:
        job_id = row.get('job_id')
        if not isinstance(job_id, str) or not job_id:
            raise ValueError('Every row must have a nonempty job_id')
        result[job_id] += price(Usage(**row['usage']), rates)
    return dict(result)


def compare(before: list[dict], after: list[dict],
            iterations: int = 2000, seed: int = 19) -> dict:
    a = cost_per_job(before)
    b = cost_per_job(after)
    ids = sorted(set(a) & set(b))
    if not ids:
        raise ValueError('No matching job IDs')
    diffs = [b[j]-a[j] for j in ids]
    total_a = sum(a[j] for j in ids)
    total_b = sum(b[j] for j in ids)
    rnd = random.Random(seed)
    samples = []
    for _ in range(iterations):
        drawn = [rnd.choice(ids) for _ in ids]
        baseline = sum(a[j] for j in drawn)
        optimized = sum(b[j] for j in drawn)
        if baseline:
            samples.append(100 * (1-optimized/baseline))
    samples.sort()
    return {
        'paired_jobs': len(ids),
        'unpaired_before_jobs': len(set(a)-set(b)),
        'unpaired_after_jobs': len(set(b)-set(a)),
        'baseline_usd_matched': round(total_a, 7),
        'optimized_usd_matched': round(total_b, 7),
        'total_cost_reduction_pct': round(100*(1-total_b/total_a), 3) if total_a else None,
        'mean_cost_difference_after_minus_before_usd': round(sum(diffs)/len(diffs), 7),
        'paired_bootstrap_95pct_cost_reduction': [
            round(samples[int(.025*(len(samples)-1))], 3),
            round(samples[int(.975*(len(samples)-1))], 3)
        ] if samples else None,
        'warning': 'Causal claims require controlled comparable traffic; interval is descriptive.',
    }


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('before', type=Path)
    parser.add_argument('after', type=Path)
    parser.add_argument('--out',type=Path,default=Path('results/paired_cost_comparison.json'))
    args=parser.parse_args()
    report=compare(load_jsonl(args.before), load_jsonl(args.after))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
