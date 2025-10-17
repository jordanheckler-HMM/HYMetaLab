#!/usr/bin/env python3
"""
Verify current implementation matches golden reference.
Run this on every check to ensure deterministic behavior.

Usage: python tests/verify_golden.py

Exit codes:
  0 = PASS (all tests match golden reference)
  1 = FAIL (tests don't match golden reference)
"""

import sys

sys.path.insert(0, "apps")

import json
import random

from loop_lite_core import compute_once


def close(a, b, tol=1e-6):
    """Check if two values are close within tolerance."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return abs(a - b) <= tol


def main():
    # Load golden reference
    try:
        with open("tests/golden_run_seed42.json") as f:
            gold = json.load(f)
    except FileNotFoundError:
        print("❌ FAIL: Golden reference file not found!")
        print("   Run 'python tests/create_golden.py' first.")
        return 1

    # Run current implementation with same seed
    random.seed(gold["seed"])
    inputs = gold["inputs"]

    dcci_cur, dhazard_cur = compute_once(
        inputs["trust"],
        inputs["hope"],
        inputs["meaning"],
        sensitivity=inputs["sensitivity"],
        noise=0.0,
    )

    # Current output
    cur = {"dcci": dcci_cur, "dhazard": dhazard_cur}

    # Compare with tolerance
    dcci_match = close(cur["dcci"], gold["outputs"]["dcci"], 1e-6)
    dhazard_match = close(cur["dhazard"], gold["outputs"]["dhazard"], 1e-6)

    ok = dcci_match and dhazard_match

    # Display results
    print("=" * 60)
    print("GOLDEN TEST VERIFICATION")
    print("=" * 60)
    print(f"Status: {'✅ PASS' if ok else '❌ FAIL'}")
    print()
    print("Current Run:")
    print(f"  ΔCCI:    {cur['dcci']:.6f}")
    print(f"  Δhazard: {cur['dhazard']:.6f}")
    print()
    print("Golden Reference:")
    print(f"  ΔCCI:    {gold['outputs']['dcci']:.6f}")
    print(f"  Δhazard: {gold['outputs']['dhazard']:.6f}")
    print()
    print("Match Results:")
    print(f"  ΔCCI:    {'✅ PASS' if dcci_match else '❌ FAIL'}")
    print(f"  Δhazard: {'✅ PASS' if dhazard_match else '❌ FAIL'}")
    print("=" * 60)

    if not ok:
        print()
        print("⚠️  Tests failed! Possible causes:")
        print("   • Unintended change in loop_lite_core.py")
        print("   • Random seed not properly set")
        print("   • Floating point precision issue")
        print()
        print("   If the change is intentional:")
        print("   Run 'python tests/create_golden.py' to update reference")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
