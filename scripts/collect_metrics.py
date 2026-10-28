#!/usr/bin/env python3
"""Aggregate numeric score files without adjusting or replacing measured values."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--group-key", default="model")
    parser.add_argument("--metric", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    by_group = defaultdict(list)
    for glob_pattern in args.inputs:
        for path in Path().glob(glob_pattern):
            if path.suffix == ".json":
                item = json.loads(path.read_text(encoding="utf-8"))
                by_group[item.get(args.group_key, path.stem)].append(float(item[args.metric]))
            elif path.suffix == ".csv":
                with path.open(newline="", encoding="utf-8") as handle:
                    for item in csv.DictReader(handle):
                        by_group[item[args.group_key]].append(float(item[args.metric]))
    if not by_group:
        raise ValueError("No matching metric records were found")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[args.group_key, "count", "mean", "std"])
        writer.writeheader()
        for name, values in sorted(by_group.items()):
            writer.writerow({args.group_key: name, "count": len(values),
                             "mean": float(np.mean(values)),
                             "std": float(np.std(values, ddof=1)) if len(values) > 1 else ""})


if __name__ == "__main__":
    main()
