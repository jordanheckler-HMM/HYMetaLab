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
    raise ImportError("Wire adapter_trust_env_feedback.py to simcore.* helpers")


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


def _instability(env: dict[str, float], cfg: dict[str, Any]) -> float:
    # env contains rolling-window stats computed by runner hooks
    w_h = cfg["env_feedback"]["hazard_weight"]
    w_e = cfg["env_feedback"]["entropy_weight"]
    # both normalized to 0..1 by runner; combine & clip
    s = w_h * env.get("hazard_mean", 0.0) + w_e * env.get("entropy_slope_pos", 0.0)
    return _clip(s, 0.0, 1.0)


def _scale_from_instability(s: float, lo: float, hi: float) -> float:
    return lo + s * (hi - lo)


def _inherit_gauss(parent, base_cfg, sigmas, scale=1.0):
    # Gaussian mutation with scaled sigma
    child = {}
    child["tau0"] = _clip(
        random.gauss(parent["tau0"], sigmas["tau0_sigma"] * scale),
        *base_cfg["bounds"]["tau0"],
    )
    child["lr_trust"] = _clip(
        abs(random.gauss(parent["lr_trust"], sigmas["lr_trust_sigma"] * scale)),
        *base_cfg["bounds"]["lr_trust"],
    )
    child["momentum"] = _clip(
        abs(random.gauss(parent["momentum"], sigmas["momentum_sigma"] * scale)),
        *base_cfg["bounds"]["momentum"],
    )
    return child


def _attach_hooks(run_cfg: dict[str, Any], study_cfg: dict[str, Any], mode: str):
    base_noise = study_cfg["controls"]["base_noise"]
    run_cfg.setdefault("hooks", {})

    # rolling environment stats
    run_cfg["hooks"]["on_epoch"] = {
        "name": "env_window_update",
        "args": {"window": study_cfg["env_feedback"]["window"]},
    }

    if mode in ["META", "CULTURAL_FB"]:
        # trust adaptation each epoch (uses traits.* keys; META updates via meta layer if present)
        run_cfg["hooks"]["on_epoch_trust"] = {
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

    # generation step: inheritance + env-responsive mutation + ε-greedy exploration
    if study_cfg.get("generations", {}).get("enabled", False):
        fb = study_cfg["env_feedback"]
        base = study_cfg["cultural_trust_fb"]["base"]
        sig = study_cfg["cultural_trust_fb"]["mutation_sigmas"]
        run_cfg["hooks"]["on_epoch_end"] = {
            "name": "cultural_generation_step_env_feedback",
            "args": {
                "period": study_cfg["generations"]["period"],
                "fitness_formula": study_cfg["generations"]["fitness"],
                "survival_pressure": study_cfg["generations"]["survival_pressure"],
                "offspring_per_parent": study_cfg["generations"][
                    "offspring_per_parent"
                ],
                "cull_fraction": study_cfg["generations"]["cull_fraction"],
                "inherit_fn": "traits_inherit_env_scaled",  # implemented in runner using logic below
                "inherit_cfg": {
                    "base": base,
                    "mutation_sigmas": sig,
                    "mut_scale_min": fb["mut_scale_min"],
                    "mut_scale_max": fb["mut_scale_max"],
                    "eps_min": fb["eps_min"],
                    "eps_max": fb["eps_max"],
                },
                "env_keys": {
                    "hazard_mean": "env.hazard_mean",
                    "entropy_slope_pos": "env.entropy_slope_pos",
                },
                "anneal": {
                    "after_epoch": fb["anneal_after_epoch"],
                    "gamma": fb["anneal_gamma"],
                },
            },
        }


def build_grid(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    grid = []
    for mode in cfg["design"]["factors"]["mode"]:
        for tau0 in cfg["design"]["factors"]["trust_tau0"]:
            for g in cfg["design"]["factors"]["goal_inequality"]:
                for shock in cfg["design"]["factors"]["shock_severity"]:
                    rcfg = {
                        "label": f"{mode}_tau{tau0:.2f}_g{g:.2f}_shock{shock:.2f}",
                        "mode": mode,
                        "epochs_cap": cfg["design"]["epochs_cap"],
                        "agents": cfg["design"]["agents"],
                        "seeds": cfg["design"]["seeds"],
                        "traits": {"tau0": tau0, "lr_trust": 0.05, "momentum": 0.10},
                        "goal_inequality": g,
                        "shock_severity": shock,
                        "log_thin": 1,
                    }
                    _attach_hooks(rcfg, cfg, mode)
                    grid.append(rcfg)
    return grid


def run_study(study_cfg: dict[str, Any]) -> dict[str, Any]:
    runs_summary, trajectories_long = sweep_and_collect(build_grid(study_cfg))
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
        extra=[
            "uplift_grid",
            "stability_window",
            "trait_drift",
            "collapse_episodes",
            "env_instability_vs_mutation",
        ],
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
