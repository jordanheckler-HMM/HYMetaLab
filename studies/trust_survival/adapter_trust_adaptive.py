from typing import Any

try:
    # prefer simcore runner helpers if available
    from simcore.runner import (
        bootstrap_summary,
        plot_standard,
        save_exports,
        sweep_and_collect,
    )
except Exception:
    # fallback to local adapter_trust helpers
    try:
        from studies.trust_survival.adapter_trust import (
            bootstrap_summary,
            plot_standard,
            save_exports,
            sweep_and_collect,
        )
    except Exception as e:
        raise ImportError(
            "Wire adapter_trust_adaptive.py to simcore.* helpers or ensure studies.trust_survival.adapter_trust is importable: "
            + str(e)
        )


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


def _map_static(trust_tau, base_coord, base_noise):
    coord = _clip(base_coord + 0.6 * (trust_tau - 0.5), 0.20, 0.90)
    defe = max(0.0, base_noise * (1.0 - 0.9 * (coord - 0.5)))
    return coord, defe


def _inject_adaptive_logic(run_cfg: dict[str, Any], study_cfg: dict[str, Any]):
    rule = study_cfg.get("adaptive_trust", {})
    if run_cfg.get("mode", "STATIC") != "ADAPTIVE":
        return
    lr = run_cfg.get("lr_trust", 0.0)
    mom = run_cfg.get("momentum", 0.0)
    base_noise = study_cfg.get("controls", {}).get("base_noise", 0.1)

    run_cfg.setdefault("hooks", {})
    # The runner/adapter_trust fallback understands a hooks dict with on_epoch
    run_cfg["hooks"]["on_epoch"] = {
        "name": "adaptive_trust_update",
        "args": {
            "lr_trust": lr,
            "momentum": mom,
            "inverse_coupling_k": 0.9,
            "base_noise": base_noise,
            "clip_lo": 0.20,
            "clip_hi": 0.90,
            "signal": {"blend": {"CCI_norm": 0.6, "survival_rate": 0.4}},
        },
    }


def build_grid(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    grid = []
    factors = cfg.get("design", {}).get("factors", {})
    for mode in factors.get("mode", ["STATIC"]):
        for tau0 in factors.get("trust_tau0", [0.5]):
            for g in factors.get("goal_inequality", [0.5]):
                for shock in factors.get("shock_severity", [0.5]):
                    for lr in factors.get("lr_trust", [0.0]):
                        for m in factors.get("momentum", [0.0]):
                            coord, defe = _map_static(
                                tau0,
                                cfg.get("controls", {}).get("base_coord", 0.5),
                                cfg.get("controls", {}).get("base_noise", 0.1),
                            )
                            rcfg = {
                                "label": f"{mode}_tau{tau0:.2f}_g{g:.2f}_shock{shock:.2f}_lr{lr:.2f}_m{m:.2f}",
                                "mode": mode,
                                "trust_tau0": tau0,
                                "epochs_cap": cfg.get("design", {}).get(
                                    "epochs_cap", 400
                                ),
                                "agents": cfg.get("design", {}).get("agents", 120),
                                "seeds": cfg.get("design", {}).get("seeds", []),
                                "coordination_strength": coord,
                                "defection_noise": defe,
                                "goal_inequality": g,
                                "shock_severity": shock,
                                "lr_trust": lr,
                                "momentum": m,
                                "hygiene": cfg.get("controls", {}).get("hygiene", {}),
                                "log_thin": 1,
                            }
                            _inject_adaptive_logic(rcfg, cfg)
                            grid.append(rcfg)
    return grid


def run_study(study_cfg: dict[str, Any]) -> dict[str, Any]:
    grid = build_grid(study_cfg)
    runs_summary, trajectories_long = sweep_and_collect(grid)
    summary = bootstrap_summary(
        runs_summary,
        factors=[
            "mode",
            "trust_tau0",
            "lr_trust",
            "momentum",
            "goal_inequality",
            "shock_severity",
        ],
        metrics=[
            "survival_rate",
            "collapse_risk",
            "hazard",
            "CCI",
            "t_recover_CCI_0_50",
            "t_recover_hazard_0_20",
        ],
    )
    figs = plot_standard(
        runs_summary,
        trajectories_long,
        outdir=study_cfg.get("exports", {}).get("figs_dir", "./figs"),
        extra=["uplift_grid", "stability_window", "interaction_maps"],
    )
    save_exports(
        study_cfg,
        runs_summary,
        trajectories_long,
        summary,
        figures=figs,
        study_name=study_cfg.get("name", "adaptive_study"),
    )
    return {"runs_summary": runs_summary, "summary": summary}
