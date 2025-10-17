[//]: # (Merged and trimmed copilot instructions for quick AI productivity)
## Quick orientation — Reality → Sim

This repo is a Reality→Sim pipeline plus many standalone experiment scripts. The pipeline ingests CSVs, builds a "param-pack" (JSON), and calls sim modules; outputs live under `discovery_results/<name>_<YYYYmmdd_HHMMSS>/`.

Key entrypoints
- `reality_pipeline/run.py` — canonical ingest → map → SimAdapter.run_all
- `reality_pipeline/v2_runner.py` — v2 reports / theme reruns (themes 3/6/9)
- `smoke_run.py` — lightweight CI/smoke runner used by `.github/workflows/ci.yml`

Discovery and calling conventions
- `reality_pipeline/sim_adapter.py` auto-discovers modules listed in `MODULE_GROUPS`. Use those filenames when adding new sim modules.
- Preferred function signature: `fn(params, outdir=...)`. `utils.try_call` inspects for `outdir` and will pass it when present.
- Fallback: if no accepted callable is found, the adapter runs the file as `python path/to/script.py`. Make sure such scripts are runnable.

Data and param-pack notes
- Param-pack sample: `real_world_param_pack.json` keys include `calibration_params`, `gravity_params`, `goals_params`, `shock_params`, `survival_params`. `v2_runner.py` composes these for themed reports.
- Time-scaling: many experiments accept `time_scale`/`shock_window` to shrink legacy epoch ranges (see `meaning_experiment.py`). Preserve this handling when editing or calling experiments.

Developer workflows (concrete)
- Setup: `python -m venv .venv && . .venv/bin/activate` then `pip install -r requirements.txt`
- Run pipeline v1: `python -m reality_pipeline.run --auto`
- Run v2 themes: `python -m reality_pipeline.v2_runner`
- Quick smoke: `python3 smoke_run.py` (CI uses Python 3.10)
- Sweep scripts & sanitize tasks live under `scripts/` (e.g. `scripts/run_phase31b_sweep.py`, `scripts/sanitize_and_report.py`); tasks are defined in the workspace task list.

Conventions and expectations
- Outputs are idempotent per-run under `discovery_results/<name>_<stamp>/` and commonly include `results.csv`, `summary.md`/`REPORT.md`, and `*_SUMMARY.json` copied to repo root.
- Look for constants `OUT_DIR`, `ROOT`, `STAMP`, `RUN_STAMP` when tracing where a script writes outputs.
- Use `REALITY_PIPELINE_LOG` env var to change log level during debugging (`reality_pipeline/utils.py`).

Checklist for adding/updating a sim module
1. Add file with a recognized name (or add it to `MODULE_GROUPS` in `sim_adapter.py`).
2. Implement a callable like `def run(params, outdir=None):` or another `ENTRYPOINT` from `sim_adapter.py`.
3. Ensure the module is runnable as `python script.py` (fallback case).
4. Preserve `time_scale`/`shock_window` handling when applicable and write outputs into `outdir` or `OUT_DIR`.
5. Keep outputs idempotent and include a small summary (`*_SUMMARY.json` / `results.csv`) for downstream tools.

Files to inspect when making changes
- `reality_pipeline/run.py`, `reality_pipeline/v2_runner.py`, `reality_pipeline/sim_adapter.py`, `reality_pipeline/utils.py`, `reality_pipeline/normalize_map.py`
- Example modules: `meaning_experiment.py`, `baby_universe_lab.py`, `gravity_analysis.py`, `goal_externalities.py`

If anything here is unclear or you'd like short example code (entrypoint skeleton, param-pack sample, or a checklist turned into a PR template), tell me which section to expand.
