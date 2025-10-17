# Uses your existing run_sim but enforces a timeout.
from adapters.sim_adapter import run_sim as _run_sim

from ops.watchdog import run_with_timeout


def run_sim(**kwargs):
    # Adjust timeout_s to your typical per-run time (seconds)
    return run_with_timeout(_run_sim, kwargs=kwargs, timeout_s=90, name="run_sim")
