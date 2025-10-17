import importlib
from typing import Any

try:
    _simcore = importlib.import_module("simcore.runner")
    run_sim = getattr(_simcore, "run_sim", None)
    sweep_and_collect = getattr(_simcore, "sweep_and_collect", None)
    save_exports = getattr(_simcore, "save_exports", None)
    plot_standard = getattr(_simcore, "plot_standard", None)
    bootstrap_summary = getattr(_simcore, "bootstrap_summary", None)
except Exception:
    sweep_and_collect = None
    bootstrap_summary = None
    plot_standard = None
    save_exports = None


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


def _attach_resonance(run_cfg: dict[str, Any], cfg: dict[str, Any]):
    rs = cfg["resonance"]
    pl = cfg["phase_lock"]
    run_cfg.setdefault("state", {})
    run_cfg.setdefault("hooks", {})
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
        },
    }
    run_cfg["hooks"]["on_epoch_pulse"] = {
        "name": "resonant_pulse",
        "args": {
            "every": rs["pulse"]["every"],
            "eps_delta": rs["pulse"]["eps_delta"],
            "lr_boost": rs["pulse"]["lr_boost"],
            "eps_key": "state.openness_eps",
        },
    }
    run_cfg["hooks"]["on_epoch_phase_lock"] = {
        "name": "phase_locked_learning",
        "args": {
            "cci_lock_threshold": pl["cci_lock_threshold"],
            "hazard_unlock_threshold": pl["hazard_unlock_threshold"],
            "ema_window": pl["ema_window"],
            "sync_gain": 1.0,
            "desync_divergence": 1.0,
            "lr_floor": 0.0,
            "lr_ceiling": 1.0,
            "lr_key": "traits.lr_trust",
            "momentum_key": "traits.momentum",
        },
    }


def _attach_arc(run_cfg: dict[str, Any], cfg: dict[str, Any]):
    arc = cfg["arc_controller"]
    run_cfg.setdefault("hooks", {})
    run_cfg["hooks"]["on_epoch_env"] = {
        "name": "env_window_update",
        "args": {"window": arc["window"], "track_var": True},
    }
    run_cfg["hooks"]["on_epoch_arc"] = {
        "name": "adaptive_resonance_control",
        "args": {
            "targets": arc["targets"],
            "gains": arc["gains"],
            "momentum_tune": arc["momentum_tune"],
            "bounds": arc["bounds"],
            "soft_guards": arc["soft_guards"],
            "keys": {
                "lr": "traits.lr_trust",
                "m": "traits.momentum",
                "cci": "env.CCI_mean",
                "survival": "env.survival_rate_mean",
                "hazard": "env.hazard_mean",
                "hazard_var": "env.hazard_var",
                "lock_flag": "env.lock_flag",
            },
        },
    }


def build_grid(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    grid = []
    for mode in cfg["design"]["factors"]["mode"]:
        for kappa in cfg["design"]["factors"]["coupling_kappa"]:
            for phi in cfg["design"]["factors"]["phase_lag_phi"]:
                for p in cfg["design"]["factors"]["exchange_prob"]:
                    for tau0 in cfg["design"]["factors"]["trust_tau0"]:
                        for g in cfg["design"]["factors"]["goal_inequality"]:
                            for shock in cfg["design"]["factors"]["shock_severity"]:
                                rcfg = {
                                    "label": f"{mode}_k{kappa:.2f}_phi{phi:.2f}_p{p:.2f}_tau{tau0:.2f}_g{g:.2f}",
                                    "mode": mode,
                                    "epochs_cap": cfg["design"]["epochs_cap"],
                                    "agents": cfg["design"]["agents"],
                                    "seeds": cfg["design"]["seeds"],
                                    "clusters": cfg.get("clusters", 8),
                                    "network_topology": cfg["design"]["factors"][
                                        "network_topology"
                                    ][0],
                                    "coupling_kappa": kappa,
                                    "phase_lag_phi": phi,
                                    "exchange_prob": p,
                                    "goal_inequality": g,
                                    "shock_severity": shock,
                                    "traits": {
                                        "tau0": tau0,
                                        "lr_trust": 0.05,
                                        "momentum": 0.10,
                                    },
                                    "state": {"openness_eps": 0.50},
                                    "log_thin": 1,
                                }
                                _attach_base_trust(rcfg, cfg["controls"]["base_noise"])
                                if mode == "ARC":
                                    _attach_resonance(rcfg, cfg)
                                    _attach_arc(rcfg, cfg)
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
            "trust_variance",
            "lr_trust_mean",
            "momentum_mean",
            "openness_eps_mean",
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
            "controller_timecourses",
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
