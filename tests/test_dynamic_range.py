#!/usr/bin/env python3
"""
Test dynamic range for Reality Loop Lite core module.
Verifies the system responds meaningfully to different input conditions.

Usage: python tests/test_dynamic_range.py

Exit codes:
  0 = PASS (sufficient dynamic range)
  1 = FAIL (insufficient dynamic range)
"""

import sys

sys.path.insert(0, "apps")

from loop_lite_core import compute_once


def run_seedless(trust, hope, meaning, sensitivity=1.8):
    """
    Run computation without seed dependency.
    Using noise=0.0 makes it fully deterministic.
    """
    dcci, dhazard = compute_once(
        trust, hope, meaning, sensitivity=sensitivity, noise=0.0
    )
    return {"dcci": dcci, "dhazard": dhazard}


def main():
    print("=" * 60)
    print("SEEDLESS DYNAMIC RANGE TEST")
    print("=" * 60)
    print()

    # Test high and low states
    hi = run_seedless(0.9, 0.9, 0.9)
    lo = run_seedless(0.1, 0.1, 0.1)

    print("HIGH state (0.9, 0.9, 0.9):")
    print(f"  ΔCCI:    {hi['dcci']:.6f}")
    print(f"  Δhazard: {hi['dhazard']:.6f}")
    print()
    print("LOW state (0.1, 0.1, 0.1):")
    print(f"  ΔCCI:    {lo['dcci']:.6f}")
    print(f"  Δhazard: {lo['dhazard']:.6f}")
    print()
    print("=" * 60)
    print("DYNAMIC RANGE CHECKS:")
    print("=" * 60)

    dcci_lift = hi["dcci"] - lo["dcci"]
    risk_reduction_lift = hi["dhazard"] - lo["dhazard"]

    # Targets:
    # - ΔCCI should increase significantly from low to high (>= 0.03)
    # - Risk reduction should increase from low to high (lift >= 0.01)
    # FIXED: Core module now returns positive values (higher = safer)
    dcci_ok = dcci_lift >= 0.03
    risk_ok = risk_reduction_lift >= 0.01

    print(
        f"ΔCCI lift:            {dcci_lift:+.6f}  (target: >= 0.03)  {'✅ PASS' if dcci_ok else '❌ FAIL'}"
    )
    print(
        f"Risk reduction lift:  {risk_reduction_lift:+.6f}  (target: >= 0.01)  {'✅ PASS' if risk_ok else '❌ FAIL'}"
    )
    print()

    # Additional diagnostics
    print("=" * 60)
    print("DIAGNOSTICS:")
    print("=" * 60)
    print(f"ΔCCI range:           {lo['dcci']:.6f} → {hi['dcci']:.6f}")
    print(f"Risk reduction range: {lo['dhazard']:.6f} → {hi['dhazard']:.6f}")
    print()
    print("Expected behavior:")
    print("  • ΔCCI should increase with trust/hope/meaning")
    print("  • Risk reduction should increase with improvement")
    print("    (higher = safer, more risk reduced)")
    print()

    if risk_reduction_lift < 0:
        print("⚠️  WARNING: Risk reduction direction may be inverted!")
        print("   LOW state has MORE risk reduction than HIGH state.")
        print("   This suggests the calculation may need review.")
        print()

    print("=" * 60)

    overall_pass = dcci_ok and risk_ok
    print(f"OVERALL: {'✅ PASS' if overall_pass else '❌ FAIL'}")
    print("=" * 60)

    if not overall_pass:
        print()
        print("⚠️  Dynamic range check failed!")
        print("   Review loop_lite_core.py risk reduction calculation.")
        print("   System may not provide meaningful user feedback.")

    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
