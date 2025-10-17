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
    # allow wrapper to fallback to local helpers if simcore not available
    sweep_and_collect = None
    bootstrap_summary = None
    plot_standard = None
    save_exports = None


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


def _build_topology(kind: str, n: int, rewire_p: float = 0.2):
    edges = []
    if kind == "fully_connected":
        for u in range(n):
            for v in range(u + 1, n):
                edges.append((u, v))
    elif kind == "ring":
        for u in range(n):
            edges.append((u, (u + 1) % n))
    elif kind == "smallworld":
        for u in range(n):
            v = (u + 1) % n
            if random.random() < rewire_p:
                v = random.randrange(n)
                while v == u:
                    v = random.randrange(n)
            edges.append((u, v))
    else:
        raise ValueError(f"Unknown topology: {kind}")
    return edges


def _attach_trust_learning(run_cfg: dict[str, Any], base_noise: float):
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


def _attach_resonance_and_network(
    run_cfg: dict[str, Any],
    cfg: dict[str, Any],
    edges: list[tuple],
    kappa: float,
    phi: float,
    p_ex: float,
):
    rs = cfg["resonance"]
    pl = cfg["phase_lock"]
    net = cfg["network"]
    run_cfg.setdefault("state", {})
    run_cfg.setdefault("hooks", {})

    # Per-cluster sinusoid with mean ε0 and local phase offsets
    run_cfg["hooks"]["on_epoch_eps"] = {
        "name": "openness_eps_resonant_network",
        "args": {
            "eps_key": "state.openness_eps_by_cluster",
            "eps0_key": "eps0",
            "A": rs.get("amplitude", 0.18),
            "T": rs.get("period_T", 120),
            "phi_global": 0.0,
            "phi_edge_mean": phi,
            "eps_lo": rs.get("eps_min", 0.05),
            "eps_hi": rs.get("eps_max", 0.85),
            "guard_hazard_uplift": net.get("guard_hazard_uplift", 0.05),
        },
    }

    run_cfg["hooks"]["on_epoch_phase_net"] = {
        "name": "phase_locked_learning_network",
        "args": {
            "edges": edges,
            "coupling_kappa": kappa,
            "exchange_prob": p_ex,
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
    n_clusters = cfg.get("clusters", 8)
    for mode in cfg["design"]["factors"]["mode"]:
        for topo in cfg["design"]["factors"]["network_topology"]:
            edges = _build_topology(
                topo, n_clusters, cfg["network"].get("rewire_p", 0.2)
            )
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
                                        "clusters": n_clusters,
                                        "goal_inequality": g,
                                        "shock_severity": shock,
                                        "traits": {
                                            "tau0": tau0,
                                            "lr_trust": 0.05,
                                            "momentum": 0.10,
                                        },
                                        "state": {
                                            "openness_eps_by_cluster": [0.50]
                                            * n_clusters
                                        },
                                        "log_thin": 1,
                                    }
                                    _attach_trust_learning(
                                        rcfg, cfg["controls"]["base_noise"]
                                    )
                                    if mode == "CRN":
                                        _attach_resonance_and_network(
                                            rcfg, cfg, edges, kappa, phi, p_ex
                                        )
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
            "openness_eps_var",
            "network_sync_index",
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
            "lock_windows",
            "network_sync",
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
