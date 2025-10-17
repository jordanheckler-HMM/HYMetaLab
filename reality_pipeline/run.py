import os

import yaml

from .ingest import load_sources
from .normalize_map import (
    build_param_pack,
    map_calibration,
    map_goals,
    map_gravity,
    map_shocks,
    map_survival,
)
from .sim_adapter import SimAdapter
from .utils import log, safe_mkdir, write_csv, write_json


def main(auto: bool = True):
    cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    outdir = cfg.get("output_dir", "discovery_results")
    safe_mkdir(outdir)
    safe_mkdir(os.path.join(outdir, "data"))

    # 1) Ingest
    surv_df = load_sources(cfg["survival"])
    shocks_df = load_sources(cfg["shocks"])
    goals_df = load_sources(cfg["goals"])
    calib_df = load_sources(cfg["calibration"])
    grav_df = load_sources(cfg["gravity"])

    # 2) Persist raw
    write_csv(surv_df, os.path.join(outdir, "data", "survival_raw.csv"))
    write_csv(shocks_df, os.path.join(outdir, "data", "shocks_raw.csv"))
    write_csv(goals_df, os.path.join(outdir, "data", "goals_raw.csv"))
    write_csv(calib_df, os.path.join(outdir, "data", "calibration_raw.csv"))
    write_csv(grav_df, os.path.join(outdir, "data", "gravity_raw.csv"))

    # 3) Map → params (pass tuning block)
    params = build_param_pack(
        map_survival(surv_df),
        map_shocks(shocks_df),
        map_goals(goals_df, tuning=cfg.get("tuning", {})),
        map_calibration(calib_df, tuning=cfg.get("tuning", {})),
        map_gravity(grav_df, tuning=cfg.get("tuning", {})),
    )
    write_json(params, os.path.join(outdir, "real_world_param_pack.json"))
    log.info(f"Param pack saved → {os.path.join(outdir,'real_world_param_pack.json')}")

    # 4) Run sims if present (non-destructive)
    SimAdapter().run_all(params, outdir=outdir)


if __name__ == "__main__":
    main()
