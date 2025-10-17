from typing import Any

try:
    from simcore.runner import (
        bootstrap_summary,
        plot_standard,
        save_exports,
        sweep_and_collect,
    )
except Exception:
    # Provide a fallback shim that raises a clear error for now.
    raise ImportError("Wire adapter_trust_meta.py to simcore.* helpers")


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


def _map_static(trust_tau, base_coord, base_noise):
    coord = _clip(base_coord + 0.6 * (trust_tau - 0.5), 0.20, 0.90)
    defe = max(0.0, base_noise * (1.0 - 0.9 * (coord - 0.5)))
    return coord, defe


def _inject_adaptive(run_cfg: dict[str, Any], study_cfg: dict[str, Any], mode: str):
    """Attach hooks for ADAPTIVE (fixed) or META (auto-tuned) trust updates."""
    base_noise = study_cfg["controls"]["base_noise"]
    meta_cfg = study_cfg.get("meta_learning", {})
    run_cfg.setdefault("hooks", {})

    if mode == "ADAPTIVE":
        run_cfg["hooks"]["on_epoch"] = {
            "name": "adaptive_trust_update",
            "args": {
                "lr_trust": run_cfg.get("lr_trust", 0.05),
                "momentum": run_cfg.get("momentum", 0.0),
                "inverse_coupling_k": 0.9,
                "base_noise": base_noise,
                "clip_lo": 0.20,
                "clip_hi": 0.90,
                "signal": {"blend": {"CCI_norm": 0.6, "survival_rate": 0.4}},
            },
        }

    if mode == "META":
        # on_epoch: normal adaptive step using current lr,m
        run_cfg["hooks"]["on_epoch"] = {
            "name": "adaptive_trust_update",
            "args": {
                "lr_trust_key": "meta.lr_current",  # runner reads from state
                "momentum_key": "meta.m_current",
                "inverse_coupling_k": 0.9,
                "base_noise": base_noise,
                "clip_lo": 0.20,
                "clip_hi": 0.90,
                "signal": {"blend": {"CCI_norm": 0.6, "survival_rate": 0.4}},
            },
        }
        # on_epoch_end: bandit-like re-tune every meta_period
        run_cfg["hooks"]["on_epoch_end"] = {
            "name": "meta_tune_trust",
            "args": {
                "meta_period": meta_cfg.get("meta_period", 40),
                "window": meta_cfg.get("window", 40),
                "candidate_lr": meta_cfg.get("candidate_lr", [0.02, 0.05, 0.08, 0.12]),
                "candidate_m": meta_cfg.get("candidate_m", [0.00, 0.30, 0.60]),
                "eps": meta_cfg.get("exploration_eps", 0.10),
                "reward_weights": meta_cfg.get(
                    "reward", {"w_dCCI": 0.6, "w_hazard": 0.4, "w_vol": 0.2}
                ),
                "state_prefix": "meta.",
            },
        }


def build_grid(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    grid = []
    for mode in cfg["design"]["factors"]["mode"]:
        for tau0 in cfg["design"]["factors"]["trust_tau0"]:
            for g in cfg["design"]["factors"]["goal_inequality"]:
                for shock in cfg["design"]["factors"]["shock_severity"]:
                    # single lr/m values in design; META ignores them (auto-tunes)
                    lr = cfg["design"]["factors"]["lr_trust"][0]
                    m = cfg["design"]["factors"]["momentum"][0]
                    coord, defe = _map_static(
                        tau0,
                        cfg["controls"]["base_coord"],
                        cfg["controls"]["base_noise"],
                    )
                    rcfg = {
                        "label": f"{mode}_tau{tau0:.2f}_g{g:.2f}_shock{shock:.2f}",
                        "mode": mode,
                        "trust_tau0": tau0,
                        "epochs_cap": cfg["design"]["epochs_cap"],
                        "agents": cfg["design"]["agents"],
                        "seeds": cfg["design"]["seeds"],
                        "coordination_strength": coord,
                        "defection_noise": defe,
                        "goal_inequality": g,
                        "shock_severity": shock,
                        "lr_trust": lr,
                        "momentum": m,
                        "log_thin": 1,
                    }
                    _inject_adaptive(rcfg, cfg, mode)
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
        ],
    )
    figs = plot_standard(
        runs_summary,
        trajectories_long,
        outdir=study_cfg["exports"]["figs_dir"],
        extra=["uplift_grid", "stability_window", "interaction_maps"],
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
