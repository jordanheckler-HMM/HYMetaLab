from typing import Any

try:
    from simcore.runner import (
        bootstrap_summary,
        plot_standard,
        save_exports,
        sweep_and_collect,
    )
except Exception:
    raise ImportError("Wire adapter_trust_adaptive_openness.py to simcore.* helpers")


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


def _eval(expr: str, **kw):
    # very small safe eval for linear forms used above
    return eval(expr, {"__builtins__": {}}, kw)


def _attach_openness_hooks(run_cfg: dict[str, Any], study_cfg: dict[str, Any]):
    eps = run_cfg["openness_eps"]
    o = study_cfg["openness"]
    exch_rate = _eval(o["exchange_rate_fn"], ε=eps)
    var_mult = _eval(o["info_inflow_fn"], ε=eps)
    run_cfg.setdefault("hooks", {})
    # cross-cluster heuristic exchange
    run_cfg["hooks"]["on_epoch_exchange"] = {
        "name": "cross_cluster_exchange",
        "args": {
            "period": o["exchange_period"],
            "rate": exch_rate,
            "keys": ["traits.lr_trust", "traits.momentum", "traits.tau0"],
        },
    }
    # variance modulation (info inflow)
    run_cfg["hooks"]["on_epoch_variance"] = {
        "name": "modulate_trust_variance",
        "args": {"sigma_mult": var_mult},
    }
    # anneal openness late game
    run_cfg["hooks"]["on_epoch_end_openness_anneal"] = {
        "name": "openness_anneal",
        "args": {
            "after_epoch": o["anneal_after_epoch"],
            "gamma": o["anneal_gamma"],
            "eps_key": "openness_eps",
        },
    }


def _permeabilize_shock(shock: float, eps: float, study_cfg: dict[str, Any]) -> float:
    o = study_cfg["openness"]
    return _eval(o["shock_permeability_fn"], shock=shock, ε=eps)


def build_grid(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    grid = []
    for mode in cfg["design"]["factors"]["mode"]:
        for eps in cfg["design"]["factors"]["openness_eps"]:
            for tau0 in cfg["design"]["factors"]["trust_tau0"]:
                for g in cfg["design"]["factors"]["goal_inequality"]:
                    for shock in cfg["design"]["factors"]["shock_severity"]:
                        rcfg = {
                            "label": f"{mode}_eps{eps:.2f}_tau{tau0:.2f}_g{g:.2f}_shock{shock:.2f}",
                            "mode": mode,
                            "openness_eps": eps,
                            "epochs_cap": cfg["design"]["epochs_cap"],
                            "agents": cfg["design"]["agents"],
                            "seeds": cfg["design"]["seeds"],
                            "clusters": cfg.get("clusters", 6),
                            "goal_inequality": g,
                            "shock_severity": _permeabilize_shock(shock, eps, cfg),
                            "traits": {
                                "tau0": tau0,
                                "lr_trust": cfg["traits"]["lr_trust_init"],
                                "momentum": cfg["traits"]["momentum_init"],
                            },
                            "log_thin": 1,
                        }
                        _attach_openness_hooks(rcfg, cfg)
                        if mode in ["META", "CULTURAL_FB", "ADAPTIVE_OPEN"]:
                            rcfg.setdefault("hooks", {})
                            rcfg["hooks"]["on_epoch_trust"] = {
                                "name": "adaptive_trust_update",
                                "args": {
                                    "lr_trust_key": "traits.lr_trust",
                                    "momentum_key": "traits.momentum",
                                    "inverse_coupling_k": 0.9,
                                    "base_noise": cfg["controls"]["base_noise"],
                                    "clip_lo": 0.10,
                                    "clip_hi": 0.95,
                                    "signal": {
                                        "blend": {"CCI_norm": 0.6, "survival_rate": 0.4}
                                    },
                                },
                            }
                        grid.append(rcfg)
    return grid


def run_study(study_cfg: dict[str, Any]) -> dict[str, Any]:
    runs_summary, trajectories_long = sweep_and_collect(build_grid(study_cfg))
    summary = bootstrap_summary(
        runs_summary,
        factors=[
            "mode",
            "openness_eps",
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
            "exchange_flow_per_epoch",
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
            "openness_tradeoffs",
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
