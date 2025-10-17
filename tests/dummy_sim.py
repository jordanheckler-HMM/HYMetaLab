def run_sim(seed=0, config=None):
    # simple deterministic fake sim for testing
    return {
        "run_id": f"sim_seed{seed}",
        "label": config.get("label", "dummy"),
        "seed": seed,
        "epochs_cap": config.get("epochs_cap", 100),
        "stability": {"CCI_mean": 0.9, "hazard_mean": 0.05, "CCI_slope": 0.001},
        "peak_CCI": 0.95,
        "final_CCI": 0.88,
        "hazard_peak": 0.1,
        "early_stopped": False,
        "collapse_flag": False,
        "lifespan_epochs": config.get("epochs_cap", 100),
        "reservoirs_ok": True,
        "time_sec": 0.0,
        "traj_rows": [],
        "openness_rows": [],
        "shock_rows": [],
    }
