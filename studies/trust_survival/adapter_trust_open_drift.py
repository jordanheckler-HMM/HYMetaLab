import random
from typing import Any

try:
    from simcore.runner import (
        bootstrap_summary,
        plot_standard,
        save_exports,
        sweep_and_collect,
    )
except Exception:
    raise ImportError("Wire adapter_trust_open_drift.py to simcore.* helpers")


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


def _map_from_tau(tau, base_coord, base_noise):
    coord = _clip(base_coord + 0.6 * (tau - 0.5), 0.10, 0.95)
    defe = max(0.0, base_noise * (1.0 - 0.9 * (coord - 0.5)))
    return coord, defe


def _inherit(parent: dict[str, float], cfg: dict[str, Any]) -> dict[str, float]:
    H = cfg["cultural_trust"]["heritability"]
    b = cfg["cultural_trust"]["bounds"]
    mut = cfg["cultural_trust"]["mutation"]
    child = {}
    # Gaussian mutation around parent with heritability
    child["tau0"] = _clip(
        random.gauss(H * parent["tau0"] + (1 - H) * 0.60, mut["tau0_sigma"]), *b["tau0"]
    )
    child["lr_trust"] = _clip(
        abs(random.gauss(H * parent["lr_trust"], mut["lr_trust_sigma"])), *b["lr_trust"]
    )
    child["momentum"] = _clip(
        abs(random.gauss(H * parent["momentum"], mut["momentum_sigma"])), *b["momentum"]
    )
    return child


def _init_traits(mode: str, cfg: dict[str, Any], tau0: float) -> dict[str, float]:
    if mode == "CULTURAL":
        return {
            "tau0": tau0,
            "lr_trust": cfg["cultural_trust"]["lr_trust_init"],
            "momentum": cfg["cultural_trust"]["momentum_init"],
        }
    # ADAPTIVE/META/STATIC fallback (reuse Phase 6/7 semantics)
    return {"tau0": tau0, "lr_trust": 0.05, "momentum": 0.00}


def _attach_hooks(run_cfg: dict[str, Any], study_cfg: dict[str, Any], mode: str):
    base_noise = study_cfg["controls"]["base_noise"]
    run_cfg.setdefault("hooks", {})

    # Trust updates inside epochs (ADAPTIVE/META/CULTURAL behave similarly,
    # but CULTURAL traits can change at generation boundaries via inheritance)
    if mode in ["ADAPTIVE", "META", "CULTURAL"]:
        run_cfg["hooks"]["on_epoch"] = {
            "name": "adaptive_trust_update",
            "args": {
                "lr_trust_key": "traits.lr_trust",
                "momentum_key": "traits.momentum",
                "inverse_coupling_k": 0.9,
                "base_noise": base_noise,
                "clip_lo": 0.10,
                "clip_hi": 0.95,
                "signal": {"blend": {"CCI_norm": 0.6, "survival_rate": 0.4}},
            },
        }

    # Generation boundary: selection, inheritance, mutation, and open flow
    if study_cfg.get("generations", {}).get("enabled", False):
        gen_cfg = study_cfg["generations"]
        flow = study_cfg.get("open_flow", {})
        run_cfg["hooks"]["on_epoch_end"] = {
            "name": "cultural_generation_step",
            "args": {
                "period": gen_cfg["period"],
                "fitness_formula": gen_cfg["fitness"],
                "survival_pressure": gen_cfg["survival_pressure"],
                "offspring_per_parent": gen_cfg["offspring_per_parent"],
                "cull_fraction": gen_cfg["cull_fraction"],
                "inherit_fn": "traits_inherit_gaussian",  # implemented in runner using _inherit semantics
                "inherit_cfg": study_cfg["cultural_trust"],
                "open_flow": {
                    "enabled": flow.get("enabled", False),
                    "influx_rate": flow.get("influx_rate", 0.0),
                    "efflux_rate": flow.get("efflux_rate", 0.0),
                    "influx_tau_dist": flow.get("influx_trust_init", "N(0.60,0.10)"),
                },
            },
        }


def build_grid(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    grid = []
    for mode in cfg["design"]["factors"]["mode"]:
        for tau0 in cfg["design"]["factors"]["trust_tau0"]:
            for g in cfg["design"]["factors"]["goal_inequality"]:
                for shock in cfg["design"]["factors"]["shock_severity"]:
                    traits = _init_traits(mode, cfg, tau0)
                    coord, defe = _map_from_tau(
                        traits["tau0"],
                        cfg["controls"]["base_coord"],
                        cfg["controls"]["base_noise"],
                    )
                    rcfg = {
                        "label": f"{mode}_tau{tau0:.2f}_g{g:.2f}_shock{shock:.2f}",
                        "mode": mode,
                        "epochs_cap": cfg["design"]["epochs_cap"],
                        "agents": cfg["design"]["agents"],
                        "seeds": cfg["design"]["seeds"],
                        "goal_inequality": g,
                        "shock_severity": shock,
                        "coordination_strength": coord,
                        "defection_noise": defe,
                        "traits": traits,
                        "log_thin": 1,
                    }
                    _attach_hooks(rcfg, cfg, mode)
                    grid.append(rcfg)
    return grid


def run_study(study_cfg: dict[str, Any]) -> dict[str, Any]:
    grid = build_grid(study_cfg)
    runs_summary, trajectories_long = sweep_and_collect(grid)
    summary = bootstrap_summary(
        runs_summary,
        factors=["mode", "trust_tau0", "goal_inequality", "shock_severity"],
        metrics=[
            "survival_rate",
            "collapse_risk",
            "hazard",
            "CCI",
            "t_recover_CCI_0_50",
            "t_recover_hazard_0_20",
            "episodes_collapse_per_1k",
            "gen_trust_entropy",
            "gen_mean_tau",
        ],
    )
    figs = plot_standard(
        runs_summary,
        trajectories_long,
        outdir=study_cfg["exports"]["figs_dir"],
        extra=["uplift_grid", "stability_window", "trait_drift", "collapse_episodes"],
    )
    save_exports(
        study_cfg,
        runs_summary,
        trajectories_long,
        summary,
        figures=figs,
        study_name=study_cfg["name"],
    )
    return {"runs_summary": runs_summary, "summary": summary}
