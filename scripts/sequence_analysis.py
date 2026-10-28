#!/usr/bin/env python3
"""Paired bootstrap and sign-flip inference for one score per independent video."""

import argparse
import csv
import json

import numpy as np
from scipy.stats import wilcoxon

from elstnet.utils import save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with open(args.pairs, encoding="utf-8", newline="") as handle:
        entries = list(csv.DictReader(handle))
    a = np.array([float(row[args.baseline]) for row in entries])
    b = np.array([float(row[args.candidate]) for row in entries])
    if len(a) < 5:
        raise ValueError("At least five independently paired units are required")
    differences = a - b
    rng = np.random.default_rng(args.seed)
    sampled = differences[rng.integers(0, len(differences), size=(args.resamples, len(differences)))]
    means = sampled.mean(1)
    bounds = np.quantile(means, [0.025, 0.975])
    signs = rng.choice((-1, 1), size=(args.resamples, len(differences)))
    null_dist = (differences * signs).mean(1)
    p_perm = (1 + np.sum(np.abs(null_dist) >= abs(differences.mean()))) / (args.resamples + 1)
    p_wilcoxon = float(wilcoxon(differences).pvalue) if np.any(differences) else 1.0
    report = {"independent_pairs": int(len(differences)),
              "mean_baseline": float(a.mean()), "mean_candidate": float(b.mean()),
              "mean_difference": float(differences.mean()),
              "bootstrap_95_low": float(bounds[0]), "bootstrap_95_high": float(bounds[1]),
              "permutation_p": float(p_perm), "wilcoxon_p": p_wilcoxon,
              "resamples": args.resamples, "seed": args.seed}
    save_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
