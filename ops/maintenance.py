import os
import shutil
import sys

from ops.validator import hazmat_sweep

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
END = "\033[0m"


def check_env():
    probs = []
    for d in ["studies", "adapters", "exports", "data", "ops"]:
        if not os.path.isdir(d):
            probs.append(f"missing dir: {d}")
    try:
        import adapters.sim_adapter as a

        if not hasattr(a, "run_sim"):
            probs.append("adapters/sim_adapter.py missing run_sim()")
    except Exception as e:
        probs.append(f"adapter import error: {e}")
    py = os.environ.get("PYTHONPATH", "")
    if os.getcwd() not in py:
        probs.append(
            "PYTHONPATH missing repo root (export PYTHONPATH=$PWD:$PYTHONPATH)"
        )
    return probs


def vacuum_exports():
    root = "exports"
    if not os.path.isdir(root):
        return 0
    removed = 0
    for dirpath, dirnames, filenames in os.walk(root):
        if not dirnames and not filenames:
            shutil.rmtree(dirpath)
            removed += 1
        for f in filenames:
            if f.endswith(".tmp"):
                os.remove(os.path.join(dirpath, f))
                removed += 1
    return removed


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    if cmd == "health":
        probs = check_env()
        if probs:
            print(RED + "Healthcheck FAIL:" + END)
            [print(" -", p) for p in probs]
            sys.exit(1)
        print(GREEN + "Healthcheck OK" + END)
    elif cmd == "hazmat":
        if len(sys.argv) < 3:
            print("usage: python ops/maintenance.py hazmat <study_id>")
            sys.exit(2)
        acts = hazmat_sweep(f"exports/{sys.argv[2]}")
        for a in acts:
            print("[HAZMAT]", a)
    elif cmd == "vacuum":
        n = vacuum_exports()
        print(f"Cleaned {n} artifacts")
    else:
        print("commands: health | hazmat <study_id> | vacuum")


if __name__ == "__main__":
    main()
