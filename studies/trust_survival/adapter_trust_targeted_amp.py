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


def _attach_arc_capacitor(run_cfg: dict[str, Any], cfg: dict[str, Any]):
    arc, rs = cfg.get("arc_controller", {}), cfg.get("resonance", {})
    run_cfg.setdefault("state", {})
    run_cfg.setdefault("hooks", {})
    run_cfg["state"]["openness_eps"] = 0.50
    run_cfg["state"]["reservoir"] = 0.0
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
            "targets": cfg.get("arc_controller", {}).get("targets", {}),
            "gains": cfg.get("arc_controller", {}).get("gains", {}),
            "momentum_tune": cfg.get("arc_controller", {}).get("momentum_tune", {}),
            "bounds": cfg.get("arc_controller", {}).get("bounds", {}),
            "soft_guards": cfg.get("arc_controller", {}).get("soft_guards", {}),
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
    run_cfg["hooks"]["on_epoch_contain"] = {
        "name": "containment_absorb",
        "args": {
            "trigger": cfg.get("containment", {}).get("trigger", {}),
            "absorption": cfg.get("containment", {}).get("absorption", {}),
            "safety": {
                "cap_reservoir": cfg.get("containment", {})
                .get("reservoir", {})
                .get("cap", 1.8),
                "freeze_pulses": True,
            },
            "keys": {
                "eps": "state.openness_eps",
                "reservoir": "state.reservoir",
                "hazard": "env.hazard_mean",
                "hazard_var": "env.hazard_var",
            },
        },
    }
    run_cfg["hooks"]["on_epoch_reinject"] = {
        "name": "containment_reinject_adaptive",
        "args": {
            "min_delay": cfg.get("reinjection_adaptive", {}).get("min_delay", 25),
            "ema_smoothing": cfg.get("reinjection_adaptive", {}).get(
                "ema_smoothing", 0.8
            ),
            "gate": cfg.get("reinjection_adaptive", {}).get("gate", {}),
            "neighborhood": {
                "edges": [],
                "kappa_bias": cfg.get("reinjection_adaptive", {})
                .get("neighborhood", {})
                .get("kappa_bias", 0.6),
                "deficit_floor": cfg.get("reinjection_adaptive", {})
                .get("neighborhood", {})
                .get("deficit_floor", 0.05),
            },
            "schedule": cfg.get("reinjection_adaptive", {}).get("schedule", {}),
            "effects": cfg.get("reinjection_adaptive", {}).get("effects", {}),
            "keys": {
                "reservoir": "state.reservoir",
                "eps": "state.openness_eps",
                "lr": "traits.lr_trust",
                "cci_by_cluster": "env.cci_by_cluster",
                "survival_by_cluster": "env.survival_by_cluster",
                "hazard": "env.hazard_mean",
            },
        },
    }


def _attach_taf(
    run_cfg: dict[str, Any], cfg: dict[str, Any], edges: list[tuple[int, int]]
):
    taf = cfg.get("taf", {})
    if not taf.get("enabled", False):
        return
    run_cfg.setdefault("hooks", {})
    run_cfg["hooks"]["on_epoch_taf"] = {
        "name": "targeted_amplification_feedback",
        "args": {
            "momentum_window": taf.get("momentum_window", 30),
            "smoothing": taf.get("smoothing", 0.85),
            "gain": taf.get("gain", {}),
            "limiter": taf.get("limiter", {}),
            "routing": {
                "edges": edges,
                "top_k": taf.get("routing", {}).get("top_k", 3),
                "spillover": taf.get("routing", {}).get("spillover", 0.15),
            },
            "keys": {
                "reservoir": "state.reservoir",
                "eps": "state.openness_eps",
                "lr": "traits.lr_trust",
                "hazard": "env.hazard_mean",
                "inter_sync": "env.inter_sync_index",
                "cci_by_cluster": "env.cci_by_cluster",
                "survival_by_cluster": "env.survival_by_cluster",
            },
        },
    }


def build_grid(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    grid = []
    nC = cfg.get("clusters", 12)
    for mode in (
        cfg.get("design", {}).get("factors", {}).get("mode", ["ARC_CAP_REINJECT"])
    ):
        for topo in (
            cfg.get("design", {})
            .get("factors", {})
            .get("supernet_topology", ["smallworld"])
        ):
            edges = _build_supernet(topo, nC)
            for k in (
                cfg.get("design", {}).get("factors", {}).get("inter_kappa", [0.18])
            ):
                for d in (
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
                                        "label": f"{mode}_{topo}_k{k:.2f}_d{d}_sf{sf:.2f}_tau{tau0:.2f}_g{g:.2f}",
                                        "mode": mode,
                                        "epochs_cap": cfg.get("design", {}).get(
                                            "epochs_cap", 950
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
                                        "state": {
                                            "openness_eps": 0.50,
                                            "reservoir": 0.0,
                                        },
                                        "log_thin": 1,
                                    }
                                    _attach_base_trust(
                                        rcfg,
                                        cfg.get("controls", {}).get("base_noise", 0.05),
                                    )
                                    if mode in ["ARC_CAP_REINJECT", "ARC_TAF"]:
                                        _attach_arc_capacitor(rcfg, cfg)
                                    if mode == "ARC_TAF":
                                        _attach_taf(rcfg, cfg, edges)
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
            "containment_active_ratio",
            "reservoir_mean",
            "reinject_efficiency",
            "reinject_precision",
            "taf_gain_mean",
            "taf_focus_score",
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
            "smart_capacitor",
            "taf_heatmaps",
        ],
    )
    save_exports(
        study_cfg,
        runs_summary,
        trajectories_long,
        summary,
        figures=figs,
        study_name=study_cfg.get("name", "phase22_taf"),
    )
    return {"runs_summary": runs_summary, "summary": summary}
