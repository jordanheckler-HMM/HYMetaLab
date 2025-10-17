#!/usr/bin/env python3
"""
Create golden reference file for Reality Loop Lite core module.
Run this when you intentionally update the mathematical model.

Usage: python tests/create_golden.py
"""

import sys

sys.path.insert(0, "apps")

import json
import random

from loop_lite_core import compute_once


def main():
    # Set deterministic seed
    random.seed(42)

    # Run with fixed inputs (Team Cohesion scenario)
    trust, hope, meaning = 0.70, 0.80, 0.90
    sensitivity = 1.8

    dcci, dhazard = compute_once(
        trust, hope, meaning, sensitivity=sensitivity, noise=0.0  # deterministic
    )

    # Create golden output
    golden = {
        "seed": 42,
        "inputs": {
            "trust": trust,
            "hope": hope,
            "meaning": meaning,
            "sensitivity": sensitivity,
        },
        "outputs": {"dcci": dcci, "dhazard": dhazard},
        "metadata": {
            "version": "1.0",
            "module": "apps.loop_lite_core",
            "function": "compute_once",
            "description": "Team Cohesion scenario golden reference",
        },
    }

    with open("tests/golden_run_seed42.json", "w") as f:
        json.dump(golden, f, indent=2)

    print("✅ Created tests/golden_run_seed42.json")
    print(f"   ΔCCI:    {dcci:.6f}")
    print(f"   Δhazard: {dhazard:.6f}")
    print()
    print("⚠️  Remember: Only update golden files when you intentionally")
    print("   change the mathematical model!")


if __name__ == "__main__":
    main()
