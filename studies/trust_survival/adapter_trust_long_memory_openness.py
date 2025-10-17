from typing import Any

try:
    from simcore.runner import (
        bootstrap_summary,
        plot_standard,
        save_exports,
        sweep_and_collect,
    )
except Exception:
    raise ImportError("Wire adapter_trust_long_memory_openness.py to simcore.* helpers")


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


def _attach_base_trust(run_cfg: dict[str, Any], base_noise: float):
    run_cfg.setdefault("hooks", {})
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


def _attach_long_memory(run_cfg: dict[str, Any], cfg: dict[str, Any]):
    lm = cfg["openness_long_memory"]
    run_cfg.setdefault("state", {})
    run_cfg["state"]["openness_eps"] = run_cfg["eps0"]
    run_cfg.setdefault("hooks", {})
    # rolling stats
    run_cfg["hooks"]["on_epoch_env"] = {
        "name": "env_window_update",
        "args": {"window": lm["ema_window"], "track_var": True},
    }
    # ε update each epoch (with guard rails + cooldown)
    run_cfg["hooks"]["on_epoch_eps_update"] = {
        "name": "openness_eps_update_long_memory",
        "args": {
            "eps_key": "state.openness_eps",
            "k_step": lm["k_step"],
            "w_cci": lm["w_cci"],
            "w_hazard_var": lm["w_hazard_var"],
            "eps_min": lm["eps_min"],
            "eps_max": lm["eps_max"],
            "cool_down": lm["guard_rails"]["cool_down"],
            "anneal_after_epoch": lm["anneal_after_epoch"],
            "anneal_gamma": lm["anneal_gamma"],
            "env_keys": {
                "cci_ema": "env.CCI_ema",
                "hazard_var_ema": "env.hazard_var_ema",
            },
        },
    }


def _permeabilize_shock(shock: float, eps: float) -> float:
    return shock * (1.0 + 0.20 * eps)


def build_grid(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    grid = []
    for mode in cfg["design"]["factors"]["mode"]:
        for eps0 in cfg["design"]["factors"]["eps0"]:
            for tau0 in cfg["design"]["factors"]["trust_tau0"]:
                for g in cfg["design"]["factors"]["goal_inequality"]:
                    for shock in cfg["design"]["factors"]["shock_severity"]:
                        rcfg = {
                            "label": f"{mode}_eps0{eps0:.2f}_tau{tau0:.2f}_g{g:.2f}_shock{shock:.2f}",
                            "mode": mode,
                            "eps0": eps0,
                            "epochs_cap": cfg["design"]["epochs_cap"],
                            "agents": cfg["design"]["agents"],
                            "seeds": cfg["design"]["seeds"],
                            "clusters": cfg.get("clusters", 6),
                            "goal_inequality": g,
                            "shock_severity": _permeabilize_shock(shock, eps0),
                            "traits": {
                                "tau0": tau0,
                                "lr_trust": cfg["traits"]["lr_trust_init"],
                                "momentum": cfg["traits"]["momentum_init"],
                            },
                            "log_thin": 1,
                        }
                        if mode in ["META", "ADAPTIVE_OPEN_LM"]:
                            _attach_base_trust(rcfg, cfg["controls"]["base_noise"])
                        if mode == "ADAPTIVE_OPEN_LM":
                            _attach_long_memory(rcfg, cfg)
                        grid.append(rcfg)
    return grid


def run_study(study_cfg: dict[str, Any]) -> dict[str, Any]:
    runs_summary, trajectories_long = sweep_and_collect(build_grid(study_cfg))
    summary = bootstrap_summary(
        runs_summary,
        factors=["mode", "eps0", "trust_tau0", "goal_inequality", "shock_severity"],
        metrics=[
            "survival_rate",
            "collapse_risk",
            "hazard",
            "CCI",
            "t_recover_CCI_0_50",
            "t_recover_hazard_0_20",
            "trust_variance",
            "openness_eps_mean",
            "openness_eps_var",
        ],
    )
    figs = plot_standard(
        runs_summary,
        trajectories_long,
        outdir=study_cfg["exports"]["figs_dir"],
        extra=[
            "uplift_grid",
            "stability_window",
            "interaction_maps",
            "openness_timecourse",
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
