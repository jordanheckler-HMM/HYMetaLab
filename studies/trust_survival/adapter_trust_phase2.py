from typing import Any

try:
    from studies.trust_survival import adapter_trust as at
except Exception:
    import studies.trust_survival.adapter_trust as at


def inject_betrayal_event(
    cfg: dict[str, Any], run_cfg: dict[str, Any]
) -> dict[str, Any]:
    betr = (
        cfg.get("prereg", {})
        .get("constants", {})
        .get("derived_events", {})
        .get("betrayal_shock")
    )
    if not betr:
        return run_cfg
    run_cfg = dict(run_cfg)
    run_cfg["dynamic_events"] = [
        {
            "epoch": int(betr.get("at_epoch", 250)),
            "delta": {
                "defection_noise": float(betr.get("delta_defection_noise", 0.25))
            },
            "duration": int(betr.get("duration", 40)),
            "repair_rate": float(betr.get("repair_rate", 0.05)),
            "label": "betrayal_shock",
        }
    ]
    return run_cfg


def run(config: dict[str, Any], seed: int = 0):
    """Compatibility run(config, seed) for OpenLaws. If config looks like a study config,
    delegate to adapter_trust.run_study after injecting events into the grid.
    Otherwise, inject events into the single-run config and call fallback sim.
    """
    # If it's the full study config, modify grid
    if "prereg" in config or "design" in config:
        # build grid, inject events for each grid element and delegate to adapter_trust.run_study
        # adapter_trust.run_study expects the study config; we'll call it directly
        # but ensure its build_grid will see the derived events via config.prereg.constants
        return at.run_study(config)

    # otherwise, it's a per-run config
    run_cfg = inject_betrayal_event(
        {"prereg": {"constants": config.get("derived_events", {})}}, config
    )
    if hasattr(at, "_fallback_sim_run"):
        return at._fallback_sim_run(run_cfg, seed=seed)
    raise RuntimeError("No fallback simulator available in adapter_trust")
