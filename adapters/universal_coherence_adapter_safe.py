from adapters.universal_coherence_adapter import run_agents as _run_agents
from adapters.universal_coherence_adapter import run_civ as _run_civ
from ops.watchdog import run_with_timeout


def run_agents(cfg):
    return run_with_timeout(
        _run_agents, kwargs={"cfg": cfg}, timeout_s=120, name="universal_run_agents"
    )


def run_civ(cfg):
    return run_with_timeout(
        _run_civ, kwargs={"cfg": cfg}, timeout_s=120, name="universal_run_civ"
    )
