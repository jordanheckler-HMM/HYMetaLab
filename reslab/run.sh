#!/usr/bin/env bash
set -e
python3 -m pip install -r reslab/requirements.txt
# Run experiments
python3 -m reslab.experiments --config configs/coord_sweep.yaml
python3 -m reslab.experiments --config configs/ineq_ramp.yaml
python3 -m reslab.experiments --config configs/shock_chaos.yaml
python3 -m reslab.experiments --config configs/stability_grid.yaml
# Analyze & export a single bundle
python3 -m reslab.analyze