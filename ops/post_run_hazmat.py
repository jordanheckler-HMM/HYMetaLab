import sys

from ops.validator import hazmat_sweep

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python ops/post_run_hazmat.py <study_id>")
        sys.exit(2)
    acts = hazmat_sweep(f"exports/{sys.argv[1]}")
    for a in acts:
        print("[HAZMAT]", a)
