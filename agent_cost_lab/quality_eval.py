"""Evaluate pre-labeled code-review findings by stable finding IDs.

Only use gold finding IDs created by reviewers and validated on a fixed suite.
Do not infer correctness from the number or severity of model assertions.
"""
import argparse
import json
import random
from pathlib import Path


def metrics(cases: list[dict], arm: str) -> dict:
    tp = fp = fn = 0
    by_cat = {}
    for case in cases:
        gold = {f["id"]: f["category"] for f in case["gold"]}
        predictions = case[arm]
        if len(predictions) != len(set(predictions)):
            raise ValueError("Duplicate prediction IDs")
        detected = set(predictions)
        gold_ids = set(gold)
        tp += len(detected & gold_ids)
        fp += len(detected - gold_ids)
        fn += len(gold_ids - detected)
        for f_id, category in gold.items():
            stat = by_cat.setdefault(category, {"gold": 0, "found": 0})
            stat["gold"] += 1
            stat["found"] += int(f_id in detected)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision,
            "recall": recall,
            "f1": (2 * precision * recall / (precision + recall)
                   if precision is not None and recall is not None and precision + recall else None),
            "recall_by_category": {key: round(v["found"] / v["gold"], 4)
                                   for key, v in sorted(by_cat.items()) if v["gold"]}}


def paired_bootstrap_recall(cases: list[dict], a: str, b: str,
                            iterations: int = 2000, seed: int = 10) -> dict:
    """Illustrative paired bootstrap resampling cases, not findings."""
    if len(cases) < 2:
        return {"ci95": None, "iterations": 0}
    rng = random.Random(seed)
    diffs = []
    for _ in range(iterations):
        sample = [rng.choice(cases) for _ in range(len(cases))]
        ma, mb = metrics(sample, a), metrics(sample, b)
        if ma["recall"] is not None and mb["recall"] is not None:
            diffs.append(mb["recall"] - ma["recall"])
    if not diffs:
        return {"ci95": None, "iterations": iterations}
    diffs.sort()
    return {"ci95": [round(diffs[int(.025 * (len(diffs) - 1))], 4),
                     round(diffs[int(.975 * (len(diffs) - 1))], 4)],
            "iterations": iterations,
            "unit": "review case, paired resampling"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results/quality_summary.json"))
    args = parser.parse_args()
    cases = json.loads(args.path.read_text(encoding="utf-8"))["cases"]
    report = {
        "baseline": metrics(cases, "baseline"),
        "optimized": metrics(cases, "optimized"),
        "optimized_minus_baseline_recall_ci": paired_bootstrap_recall(cases, "baseline", "optimized"),
        "warning": "Fixture is SYNTHETIC, manually labeled; this is not a model comparison.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
