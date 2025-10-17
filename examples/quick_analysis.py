"""Quick analysis: load decisions.jsonl from a run and plot average reported confidence per tick."""

import json
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt


def load_jsonl(path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def plot_avg_confidence(decisions_path: str, out_path: str):
    by_tick = defaultdict(list)
    for rec in load_jsonl(decisions_path):
        tick = rec.get("tick", 0)
        conf = rec.get("reported_conf", None)
        if conf is not None:
            by_tick[tick].append(conf)

    ticks = sorted(by_tick.keys())
    avg = [sum(by_tick[t]) / len(by_tick[t]) for t in ticks]

    plt.figure()
    plt.plot(ticks, avg, marker="o")
    plt.xlabel("Tick")
    plt.ylabel("Average Reported Confidence")
    plt.title("Average Reported Confidence over Time")
    plt.grid(True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    print("Saved plot to", out_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python examples/quick_analysis.py <path-to-run-folder>")
        sys.exit(1)
    run_dir = sys.argv[1]
    decisions = os.path.join(run_dir, "decisions.jsonl")
    out = os.path.join(run_dir, "quick_plots", "avg_confidence.png")
    plot_avg_confidence(decisions, out)
