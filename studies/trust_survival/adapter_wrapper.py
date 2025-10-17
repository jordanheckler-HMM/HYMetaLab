from typing import Any

# Delegates single-run calls to adapter_trust which was written to run whole study grids.
try:
    from studies.trust_survival import adapter_trust as at
except Exception:
    # try relative import fallback
    import studies.trust_survival.adapter_trust as at


def run(config: dict[str, Any], seed: int = 0):
    """Compatibility wrapper: expected signature run(config, seed) -> DataFrame
    If config contains a grid-like structure (list of seeds), call adapter_trust.run_study.
    Otherwise, run one simulation and return its trajectory.
    """
    # If config looks like the study config, run entire study
    if "prereg" in config or "design" in config:
        # adapter_trust.run_study expects the full study config
        res = at.run_study(config)
        # return a single summary DataFrame for compatibility; callers expect a trajectory
        # We'll return the concatenated trajectories as DataFrame-like object (pandas)
        return res.get("runs_summary")

    # Otherwise, run single sim using the fallback internal sim
    # adapter_trust exposes _fallback_sim_run and returns DataFrame
    if hasattr(at, "_fallback_sim_run"):
        return at._fallback_sim_run(config, seed=seed)
    raise RuntimeError("No available simulation function in adapter_trust")
