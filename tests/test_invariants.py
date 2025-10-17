#!/usr/bin/env python3
"""
Test invariants for Reality Loop Lite core module.
Verifies outputs stay within expected ranges across multiple seeds.

Usage: python tests/test_invariants.py

Exit codes:
  0 = PASS (all invariants hold)
  1 = FAIL (some invariants violated)
"""

import sys

sys.path.insert(0, "apps")

import random

from loop_lite_core import compute_once

# Expected ranges
DCCI_MIN = -0.001
DCCI_MAX = 1.0
DHAZARD_MIN = -1.0
DHAZARD_MAX = 0.05


def test_scenario(
    name, trust, hope, meaning, sensitivity=1.8, seeds=range(1, 6), noise=0.0
):
    """Test a scenario across multiple seeds."""
    print(f"\n{name}:")
    print("-" * 60)

    all_pass = True

    for s in seeds:
        random.seed(s)
        dcci, dhz = compute_once(
            trust, hope, meaning, sensitivity=sensitivity, noise=noise
        )

        # Check invariants
        dcci_ok = DCCI_MIN <= dcci <= DCCI_MAX
        dhz_ok = DHAZARD_MIN <= dhz <= DHAZARD_MAX

        seed_pass = dcci_ok and dhz_ok

        if not seed_pass:
            status = "❌ FAIL"
            print(f"Seed {s}: {status}")
            print(f"  ΔCCI:    {dcci:.6f}  {'✅' if dcci_ok else '❌ OUT OF RANGE'}")
            print(f"  Δhazard: {dhz:.6f}  {'✅' if dhz_ok else '❌ OUT OF RANGE'}")
            all_pass = False

    if all_pass:
        print(f"✅ All {len(list(seeds))} seeds passed")

    return all_pass


def main():
    print("=" * 60)
    print("INVARIANT TESTS")
    print("=" * 60)
    print("\nExpected Ranges:")
    print(f"  ΔCCI:    [{DCCI_MIN}, {DCCI_MAX}]")
    print(f"  Δhazard: [{DHAZARD_MIN}, {DHAZARD_MAX}]")

    all_tests_pass = True

    # Test scenarios (deterministic, noise=0.0)
    print("\n" + "=" * 60)
    print("DETERMINISTIC TESTS (noise=0.0)")
    print("=" * 60)

    tests = [
        ("Crisis", 0.15, 0.20, 0.30),
        ("Neutral", 0.50, 0.50, 0.50),
        ("Optimistic", 0.85, 0.90, 0.85),
        ("Team Cohesion", 0.70, 0.80, 0.90),
    ]

    for name, trust, hope, meaning in tests:
        if not test_scenario(name, trust, hope, meaning, noise=0.0):
            all_tests_pass = False

    # Test with noise (stochastic)
    print("\n" + "=" * 60)
    print("STOCHASTIC TESTS (noise=0.02, 20 seeds)")
    print("=" * 60)

    for name, trust, hope, meaning in tests:
        if not test_scenario(
            f"{name} (with noise)", trust, hope, meaning, noise=0.02, seeds=range(1, 21)
        ):
            all_tests_pass = False

    # Summary
    print("\n" + "=" * 60)
    print(f"OVERALL: {'✅ PASS' if all_tests_pass else '❌ FAIL'}")
    print("=" * 60)

    return 0 if all_tests_pass else 1


if __name__ == "__main__":
    sys.exit(main())
