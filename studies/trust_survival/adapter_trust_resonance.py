import math
from typing import Any

try:
    from simcore.runner import (
        bootstrap_summary,
        plot_standard,
        save_exports,
        sweep_and_collect,
    )
except Exception:
    # defer import errors to wrapper/fallback which will call local helpers
    sweep_and_collect = None
    bootstrap_summary = None
    plot_standard = None
    save_exports = None


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


def _sin_eps(t, eps0, A, T, phi, lo, hi):
    val = eps0 + A * math.sin(2 * math.pi * t / float(T) + phi)
    return _clip(val, lo, hi)


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


def _attach_resonance_hooks(
    run_cfg: dict[str, Any],
    cfg: dict[str, Any],
    baseline_hazard_key="meta.baseline_hazard",
):
    rs = cfg["resonance"]
    pl = cfg["phase_lock"]
    run_cfg.setdefault("state", {})
    run_cfg.setdefault("hooks", {})

    # Rolling environment stats for lock/unlock
    run_cfg["hooks"]["on_epoch_env"] = {
        "name": "env_window_update",
        "args": {"window": pl["ema_window"], "track_var": True},
    }

    # Epsilon sinusoid each epoch with hazard guard rails
    run_cfg["hooks"]["on_epoch_eps"] = {
        "name": "openness_eps_resonant",
        "args": {
            "eps_key": "state.openness_eps",
            "eps0_key": "eps0",
            "A": rs["amplitude"],
            "T": rs["period_T"],
            "phi": rs["phase_phi"],
            "lo": rs["eps_min"],
            "hi": rs["eps_max"],
            "guard_hazard_uplift": rs["guard_hazard_uplift"],
            "baseline_hazard_key": baseline_hazard_key,
        },
    }

    # Phase-lock / desync controller on learning rate
    run_cfg["hooks"]["on_epoch_phase_lock"] = {
        "name": "phase_locked_learning",
        "args": {
            "cci_lock_threshold": pl["cci_lock_threshold"],
            "hazard_unlock_threshold": pl["hazard_unlock_threshold"],
            "ema_window": pl["ema_window"],
            "sync_gain": pl["sync_gain"],
            "desync_divergence": pl["desync_divergence"],
            "lr_floor": pl["lr_floor"],
            "lr_ceiling": pl["lr_ceiling"],
            "lr_key": "traits.lr_trust",
            "momentum_key": "traits.momentum",
        },
    }


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
                            "shock_severity": shock,
                            "traits": {
                                "tau0": tau0,
                                "lr_trust": 0.05,
                                "momentum": 0.10,
                            },
                            "state": {"openness_eps": eps0},
                            "log_thin": 1,
                        }
                        # trust learning baseline
                        if mode in ["META", "RESONANT_LOCK"]:
                            _attach_base_trust(rcfg, cfg["controls"]["base_noise"])
                        # resonance & phase-lock only for RESONANT_LOCK
                        if mode == "RESONANT_LOCK":
                            _attach_resonance_hooks(rcfg, cfg)
                        grid.append(rcfg)
    return grid


def run_study(study_cfg: dict[str, Any]) -> dict[str, Any]:
    if sweep_and_collect is None:
        raise ImportError(
            "simcore.runner helpers not available; call via wrapper fallback"
        )
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
            "episodes_collapse_per_1k",
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
            "lock_windows",
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
