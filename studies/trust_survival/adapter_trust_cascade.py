import importlib
import random
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


def _build_supernet(kind: str, n: int) -> list[tuple[int, int]]:
    E = []
    if kind == "chain":
        for i in range(n - 1):
            E.append((i, i + 1))
    elif kind == "smallworld":
        for i in range(n):
            E.append((i, (i + 1) % n))
        for _ in range(max(1, n // 4)):
            u = random.randrange(n)
            v = random.randrange(n)
            if u != v and (u, v) not in E and (v, u) not in E:
                E.append((u, v))
    elif kind == "tree":
        nodes = list(range(n))
        random.shuffle(nodes)
        for i in range(1, n):
            E.append((nodes[i], nodes[random.randrange(0, i)]))
    else:
        raise ValueError(f"Unknown supernet topology: {kind}")
    return E


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


def _attach_arc_hooks(run_cfg: dict[str, Any], cfg: dict[str, Any]):
    arc = cfg.get("arc_controller", {})
    rs = cfg.get("resonance", {})
    run_cfg.setdefault("state", {})
    run_cfg.setdefault("hooks", {})

    run_cfg["hooks"]["on_epoch_env"] = {
        "name": "env_window_update",
        "args": {"window": arc.get("window", 60), "track_var": True},
    }
    run_cfg["hooks"]["on_epoch_eps"] = {
        "name": "openness_eps_resonant",
        "args": {
            "eps_key": "state.openness_eps",
            "eps0_key": "eps0",
            "A": rs.get("amplitude", 0.12),
            "T": rs.get("period_T", 100),
            "phi": rs.get("phase_phi", 0.0),
            "lo": rs.get("eps_min", 0.05),
            "hi": rs.get("eps_max", 0.80),
        },
    }
    run_cfg["hooks"]["on_epoch_pulse"] = {
        "name": "resonant_pulse",
        "args": {
            "every": rs.get("pulse", {}).get("every", 100),
            "eps_delta": rs.get("pulse", {}).get("eps_delta", 0.06),
            "lr_boost": rs.get("pulse", {}).get("lr_boost", 1.10),
            "eps_key": "state.openness_eps",
        },
    }
    run_cfg["hooks"]["on_epoch_phase_lock"] = {
        "name": "phase_locked_learning",
        "args": {
            "cci_lock_threshold": 0.70,
            "hazard_unlock_threshold": 0.51,
            "ema_window": 40,
            "sync_gain": 1.0,
            "desync_divergence": 1.0,
            "lr_floor": 0.0,
            "lr_ceiling": 1.0,
            "lr_key": "traits.lr_trust",
            "momentum_key": "traits.momentum",
        },
    }
    run_cfg["hooks"]["on_epoch_arc"] = {
        "name": "adaptive_resonance_control",
        "args": {
            "targets": arc.get("targets", {}),
            "gains": arc.get("gains", {}),
            "momentum_tune": arc.get("momentum_tune", {}),
            "bounds": arc.get("bounds", {}),
            "soft_guards": arc.get("soft_guards", {}),
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


def _attach_cascade_hooks(
    run_cfg: dict[str, Any],
    cfg: dict[str, Any],
    edges: list[tuple[int, int]],
    inter_kappa: float,
    inter_delay: int,
    seed_frac: float,
):
    cz = cfg.get("cascade", {})
    run_cfg.setdefault("hooks", {})
    run_cfg.setdefault("state", {})
    run_cfg["hooks"]["on_start_seed"] = {
        "name": "cascade_seed_clusters",
        "args": {
            "mode": cz.get("seed_clusters_mode", "central"),
            "seed_fraction": seed_frac,
            "threshold_cci": cz.get("threshold_cci", 0.72),
        },
    }
    run_cfg["hooks"]["on_epoch_cascade"] = {
        "name": "cascade_delayed_coupling",
        "args": {
            "edges": edges,
            "inter_kappa": inter_kappa,
            "inter_delay": inter_delay,
        },
    }
    run_cfg["hooks"]["on_epoch_measure"] = {
        "name": "cascade_measure",
        "args": {
            "every": cz.get("measure_every", 10),
            "thresholds": {
                "cci": cz.get("threshold_cci", 0.72),
                "survival": cz.get("threshold_survival", 0.80),
            },
            "sustain_window": cz.get("sustain_window", 200),
            "metrics_keys": {
                "lambda": "metrics.cascade_lambda",
                "sustain": "metrics.cascade_sustain_rate",
                "sync": "metrics.inter_sync_index",
            },
        },
    }


def build_grid(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    grid = []
    nC = cfg.get("clusters", 12)
    for mode in cfg.get("design", {}).get("factors", {}).get("mode", ["ARC"]):
        for topo in (
            cfg.get("design", {}).get("factors", {}).get("supernet_topology", ["chain"])
        ):
            edges = _build_supernet(topo, nC)
            for k in (
                cfg.get("design", {}).get("factors", {}).get("inter_kappa", [0.15])
            ):
                for dly in (
                    cfg.get("design", {}).get("factors", {}).get("inter_delay", [10])
                ):
                    for sf in (
                        cfg.get("design", {})
                        .get("factors", {})
                        .get("seed_fraction", [0.25])
                    ):
                        for tau0 in (
                            cfg.get("design", {})
                            .get("factors", {})
                            .get("trust_tau0", [0.6])
                        ):
                            for g in (
                                cfg.get("design", {})
                                .get("factors", {})
                                .get("goal_inequality", [0.2])
                            ):
                                for shock in (
                                    cfg.get("design", {})
                                    .get("factors", {})
                                    .get("shock_severity", [0.5])
                                ):
                                    rcfg = {
                                        "label": f"{mode}_{topo}_k{k:.2f}_d{dly}_sf{sf:.2f}_tau{tau0:.2f}_g{g:.2f}",
                                        "mode": mode,
                                        "epochs_cap": cfg.get("design", {}).get(
                                            "epochs_cap", 900
                                        ),
                                        "agents": cfg.get("design", {}).get(
                                            "agents", 240
                                        ),
                                        "seeds": cfg.get("design", {}).get("seeds", []),
                                        "clusters": nC,
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
                                    _attach_base_trust(
                                        rcfg,
                                        cfg.get("controls", {}).get("base_noise", 0.05),
                                    )
                                    if mode in ["ARC", "ARC_CASCADE"]:
                                        _attach_arc_hooks(rcfg, cfg)
                                    if mode == "ARC_CASCADE":
                                        _attach_cascade_hooks(
                                            rcfg, cfg, edges, k, dly, sf
                                        )
                                    grid.append(rcfg)
    return grid


def run_study(study_cfg: dict[str, Any]) -> dict[str, Any]:
    if sweep_and_collect is None:
        raise ImportError(
            "simcore.runner helpers not available; adapter expects simcore.runner or wrapper fallback"
        )
    runs_summary, trajectories_long = sweep_and_collect(build_grid(study_cfg))
    summary = bootstrap_summary(
        runs_summary,
        factors=[
            "mode",
            "supernet_topology",
            "inter_kappa",
            "inter_delay",
            "seed_fraction",
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
            "cascade_lambda",
            "cascade_sustain_rate",
            "inter_sync_index",
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
            "cascade_curves",
            "supernet_sync",
        ],
    )
    save_exports(
        study_cfg,
        runs_summary,
        trajectories_long,
        summary,
        figures=figs,
        study_name=study_cfg.get("name", "phase19_cascade"),
    )
    return {"runs_summary": runs_summary, "summary": summary}
