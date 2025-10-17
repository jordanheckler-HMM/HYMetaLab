import pandas as pd

from studies.trust_survival import adapter_trust_adaptive


def run(config: dict, seed: int = None):
    """Run a single config (or single-cell grid) to match openlaws_automation's expected adapter signature.

    The automation expects an epoch-wise DataFrame per call. Build a one-item grid and call sweep_and_collect,
    then return the epoch trajectories for the requested seed (so the automation can compute the stability window).
    """
    cfg = dict(config)
    # ensure seeds in design are a single-element list with the requested seed
    if seed is not None:
        cfg.setdefault("design", {})["seeds"] = [seed]

    # build a one-item grid entry matching build_grid logic
    grid = []
    run_conf = {}
    # copy fields expected by adapter_trust_adaptive.build_grid
    for k in [
        "mode",
        "trust_tau0",
        "epochs_cap",
        "agents",
        "seeds",
        "coordination_strength",
        "defection_noise",
        "goal_inequality",
        "shock_severity",
        "lr_trust",
        "momentum",
        "hygiene",
        "label",
        "log_thin",
        "epochs",
    ]:
        if k in config:
            run_conf[k] = config[k]

    # fallback: if not present, try to fill from prereg constants
    prereg = cfg.get("prereg", {}).get("constants", {})
    run_conf.setdefault("mode", prereg.get("mode", "STATIC"))
    run_conf.setdefault(
        "trust_tau0",
        (
            prereg.get("trust_tau0", [0.5])[0]
            if isinstance(prereg.get("trust_tau0", None), list)
            else prereg.get("trust_tau0", 0.5)
        ),
    )
    run_conf.setdefault(
        "epochs_cap",
        cfg.get("design", {}).get(
            "epochs_cap", cfg.get("prereg", {}).get("constants", {}).get("epochs", 400)
        ),
    )
    run_conf.setdefault(
        "epochs",
        run_conf.get(
            "epochs_cap", cfg.get("prereg", {}).get("constants", {}).get("epochs", 400)
        ),
    )
    run_conf.setdefault(
        "agents",
        cfg.get("design", {}).get(
            "agents", cfg.get("prereg", {}).get("constants", {}).get("agents", 120)
        ),
    )
    run_conf.setdefault(
        "seeds", cfg.get("design", {}).get("seeds", [seed] if seed is not None else [])
    )

    # compute coordination_strength and defection_noise from trust_tau0 if missing
    def _clip(x, lo, hi):
        return max(lo, min(hi, x))

    if run_conf.get("coordination_strength") is None:
        tau = run_conf.get("trust_tau0", 0.5)
        base_coord = cfg.get("controls", {}).get("base_coord", 0.5)
        coord = _clip(base_coord + 0.6 * (tau - 0.5), 0.30, 0.70)
        run_conf["coordination_strength"] = coord
    if run_conf.get("defection_noise") is None:
        tau = run_conf.get("trust_tau0", 0.5)
        base_noise = cfg.get("controls", {}).get("base_noise", 0.1)
        defe = max(0.0, base_noise * (1.0 - 0.8 * (tau - 0.5)))
        run_conf["defection_noise"] = defe

    run_conf.setdefault(
        "goal_inequality",
        config.get("goal_inequality", prereg.get("goal_inequality", [0.5])[0]),
    )
    run_conf.setdefault(
        "shock_severity",
        config.get("shock_severity", prereg.get("shock_severity", [0.5])[0]),
    )
    run_conf.setdefault(
        "lr_trust", config.get("lr_trust", prereg.get("lr_trust", [0.0])[0])
    )
    run_conf.setdefault(
        "momentum", config.get("momentum", prereg.get("momentum", [0.0])[0])
    )
    run_conf.setdefault(
        "hygiene", config.get("hygiene", cfg.get("controls", {}).get("hygiene", {}))
    )
    run_conf.setdefault("log_thin", config.get("log_thin", 1))

    grid.append(run_conf)
    # use sweep_and_collect to run the single config
    runs_summary, trajectories_long = adapter_trust_adaptive.sweep_and_collect(grid)

    # prefer to return epoch-level trajectories for the requested seed
    try:
        if not trajectories_long.empty:
            if seed is not None:
                traj = trajectories_long[trajectories_long["seed"] == int(seed)].copy()
            else:
                traj = trajectories_long
            # ensure expected metric columns exist
            for col in ["epoch", "CCI", "survival_rate", "hazard", "collapse_risk"]:
                if col not in traj.columns:
                    traj[col] = pd.NA
            # return epoch-level df sorted by epoch
            traj = traj.sort_values("epoch")
            return traj.reset_index(drop=True)
    except Exception:
        pass

    # fallback: if no trajectories available, synthesize an epoch series from runs_summary row
    if not runs_summary.empty:
        row = runs_summary.iloc[0]
        E = int(run_conf.get("epochs", run_conf.get("epochs_cap", 400)))
        epochs = list(range(E))
        df = pd.DataFrame(
            {
                "epoch": epochs,
                "CCI": [row.get("CCI", float("nan"))] * E,
                "survival_rate": [row.get("survival_rate", float("nan"))] * E,
                "hazard": [row.get("hazard", float("nan"))] * E,
                "collapse_risk": [row.get("collapse_risk", float("nan"))] * E,
            }
        )
        return df

    # optimized fallback: empty epoch frame
    return pd.DataFrame(
        columns=["epoch", "CCI", "survival_rate", "hazard", "collapse_risk"]
    )
