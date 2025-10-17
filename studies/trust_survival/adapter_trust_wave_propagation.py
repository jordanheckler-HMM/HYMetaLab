import statistics
from typing import Any

try:
    from simcore.runner import (
        bootstrap_summary,
        plot_standard,
        save_exports,
        sweep_and_collect,
    )
except Exception:
    sweep_and_collect = None
    bootstrap_summary = None
    plot_standard = None
    save_exports = None


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


def _measure_lambda(coh: list[float], thresh=0.5):
    active = [i for i, c in enumerate(coh) if c > thresh]
    if not active:
        return 0.0
    return (max(active) - min(active) + 1) / len(coh)


def _measure_damping(amplitudes: list[float]) -> float:
    if len(amplitudes) < 2:
        return 0.0
    diffs = [abs(amplitudes[i + 1] - amplitudes[i]) for i in range(len(amplitudes) - 1)]
    return (
        1.0 - statistics.mean(diffs) / max(amplitudes) if max(amplitudes) > 0 else 0.0
    )


def _attach_wave_hooks(run_cfg: dict[str, Any], cfg: dict[str, Any]):
    wd = cfg["wave_dynamics"]
    run_cfg.setdefault("state", {})
    run_cfg.setdefault("hooks", {})
    run_cfg["hooks"]["on_epoch_wave"] = {
        "name": "coherence_wave_propagation",
        "args": {
            "seed_clusters": wd["seed_clusters"],
            "propagation_window": wd["propagation_window"],
            "damping_factor": wd["damping_factor"],
            "reinforcement_gain": wd["reinforcement_gain"],
            "noise_floor": wd["noise_floor"],
            "measure_period": wd["measure_period"],
            "guard_hazard_uplift": wd["guard_hazard_uplift"],
        },
    }


def build_grid(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    grid = []
    for mode in cfg["design"]["factors"]["mode"]:
        for topo in cfg["design"]["factors"]["network_topology"]:
            for kappa in cfg["design"]["factors"]["coupling_kappa"]:
                for phi in cfg["design"]["factors"]["phase_lag_phi"]:
                    for p_ex in cfg["design"]["factors"]["exchange_prob"]:
                        for tau0 in cfg["design"]["factors"]["trust_tau0"]:
                            for g in cfg["design"]["factors"]["goal_inequality"]:
                                for shock in cfg["design"]["factors"]["shock_severity"]:
                                    rcfg = {
                                        "label": f"{mode}_{topo}_k{kappa:.2f}_phi{phi:.2f}_pex{p_ex:.2f}_tau{tau0:.2f}_g{g:.2f}_shock{shock:.2f}",
                                        "mode": mode,
                                        "network_topology": topo,
                                        "coupling_kappa": kappa,
                                        "phase_lag_phi": phi,
                                        "exchange_prob": p_ex,
                                        "eps0": 0.50,
                                        "epochs_cap": cfg["design"]["epochs_cap"],
                                        "agents": cfg["design"]["agents"],
                                        "seeds": cfg["design"]["seeds"],
                                        "clusters": cfg["clusters"],
                                        "goal_inequality": g,
                                        "shock_severity": shock,
                                        "traits": {
                                            "tau0": tau0,
                                            "lr_trust": 0.05,
                                            "momentum": 0.10,
                                        },
                                        "state": {
                                            "openness_eps_by_cluster": [0.50]
                                            * cfg["clusters"]
                                        },
                                        "log_thin": 1,
                                    }
                                    _attach_wave_hooks(rcfg, cfg)
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
        factors=[
            "mode",
            "network_topology",
            "coupling_kappa",
            "phase_lag_phi",
            "exchange_prob",
            "trust_tau0",
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
            "episodes_collapse_per_1k",
            "trust_variance",
            "openness_eps_mean",
            "network_sync_index",
            "wave_propagation_lambda",
            "wave_damping_delta",
        ],
    )
    figs = plot_standard(
        runs_summary,
        trajectories_long,
        outdir=study_cfg["exports"]["figs_dir"],
        extra=["uplift_grid", "wavefronts", "network_sync", "propagation_curves"],
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
