import os

from .utils import import_from_path, log, rglob_module_paths, try_call

ENTRYPOINTS = [
    ("run_shock_sweep", True),
    ("run_survival", True),
    ("run_belief_experiment", True),
    ("run_calibration", True),
    ("run_goal_sweep", True),
    ("run_gravity_experiments", True),
    ("main", True),
    ("main", False),
]

MODULE_GROUPS = {
    "shock": ["shock_resilience.py", "shock_sweep.py"],
    "survival": ["survival_experiment.py", "lesion_experiment.py"],
    "belief": ["belief_experiment.py", "meaning_experiment.py"],
    "calibration": ["calibration_experiment.py"],
    "goals": ["goal_externalities.py", "goal_externalities_sweep.py"],
    "gravity": ["gravity_analysis.py", "gravity_nbody.py"],
}


class SimAdapter:
    def __init__(self, search_roots=None):
        self.roots = search_roots or [os.getcwd()]
        self.paths = {k: self._find_first(v) for k, v in MODULE_GROUPS.items()}
        self.mods = {
            k: (import_from_path(p) if p else None) for k, p in self.paths.items()
        }

    def _find_first(self, filenames):
        for name in filenames:
            hits = rglob_module_paths(name, self.roots)
            if hits:
                log.info(f"Found {name} at {hits[0]}")
                return hits[0]
        return None

    def available(self):
        return {k: bool(v) for k, v in self.paths.items()}

    def run_all(self, params, outdir: str):
        avail = self.available()
        log.info(f"Detected modules: {avail}")
        ran_any = False
        for key in MODULE_GROUPS.keys():
            mod = self.mods.get(key)
            path = self.paths.get(key)
            if not (mod or path):
                continue
            ran_any = True
            domain_params = (
                params.get("shock_params")
                if key == "shock"
                else (
                    params.get("survival_params")
                    if key == "survival"
                    else (
                        params.get("goals_params")
                        if key in ("belief", "goals")
                        else (
                            params.get("calibration_params")
                            if key == "calibration"
                            else (
                                params.get("gravity_params") if key == "gravity" else {}
                            )
                        )
                    )
                )
            )
            try_call(mod if mod else None, ENTRYPOINTS, domain_params, outdir, path)

        if not ran_any:
            log.info(
                "No sim modules detected; completed in NO-SIM mode (data prepared + params exported)."
            )
