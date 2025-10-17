from __future__ import annotations

import importlib.util
import inspect
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class SimResult:
    resilience: float
    survival_rate: float
    hazard: float
    cci: float
    eta: float


def _find_candidate() -> Path:
    root = Path(".").resolve()
    override = os.environ.get("SIM_MAIN_PATH")
    if override:
        p = (root / override).resolve()
        if not p.exists():
            raise FileNotFoundError(f"SIM_MAIN_PATH points to missing file: {p}")
        return p
    ignore = {
        "legacy",
        ".git",
        ".venv",
        "venv",
        "env",
        "node_modules",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".idea",
        ".vscode",
        "adapters",
        "tools",
    }
    newest = None
    newest_mtime = -1.0
    for p in root.rglob("main.py"):
        if any(part in ignore for part in p.parts):
            continue
        try:
            mt = p.stat().st_mtime
        except Exception:
            continue
        if mt > newest_mtime:
            newest_mtime, newest = mt, p
    if not newest:
        raise RuntimeError(
            "No candidate main.py found. Set SIM_MAIN_PATH env to point to your sim's main.py."
        )
    return newest


def _import_by_path(py_path: Path):
    spec = importlib.util.spec_from_file_location(py_path.stem, py_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {py_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[py_path.stem] = mod
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def _try_call(obj, **kw):
    sig = inspect.signature(obj)
    filt = {k: v for k, v in kw.items() if k in sig.parameters}
    return obj(**filt)


def _extract(out: Any) -> SimResult:
    if isinstance(out, dict):
        get = out.get
    else:
        get = lambda k: getattr(out, k)
    # Raise if any required field missing—better to fail than fabricate
    required = ["resilience", "survival_rate", "hazard", "cci", "eta"]
    missing = [k for k in required if (get(k) is None)]
    if missing:
        raise KeyError(f"Simulation output missing required fields: {missing}")
    return SimResult(
        resilience=float(get("resilience")),
        survival_rate=float(get("survival_rate")),
        hazard=float(get("hazard")),
        cci=float(get("cci")),
        eta=float(get("eta")),
    )


def run_simulation(
    *, epsilon: float, seed: int, epochs: int, cci_target: float, eta_target: float
) -> SimResult:
    """
    Programmatic entrypoint expected by the Phase 31b adapter.
    It imports the newest main.py (or SIM_MAIN_PATH) and tries common entrypoints.
    Absolutely no synthetic math here—results come only from your real sim.
    """
    main_path = _find_candidate()
    mod = _import_by_path(main_path)

    # Try function-style first
    for fn_name in ("run_simulation", "simulate", "run"):
        fn = getattr(mod, fn_name, None)
        if callable(fn):
            out = _try_call(
                fn,
                epsilon=epsilon,
                seed=seed,
                epochs=epochs,
                cci_target=cci_target,
                eta_target=eta_target,
            )
            return _extract(out)

    # Try class-style patterns
    for cls_name in ("Simulation", "Simulator", "World"):
        cls = getattr(mod, cls_name, None)
        if inspect.isclass(cls):
            inst = cls()
            run_m = getattr(inst, "run", None)
            if callable(run_m):
                out = _try_call(
                    run_m,
                    epsilon=epsilon,
                    seed=seed,
                    epochs=epochs,
                    cci_target=cci_target,
                    eta_target=eta_target,
                )
                return _extract(out)

    # If we got here, we couldn’t find a callable. Tell the user exactly what to do.
    raise RuntimeError(
        f"No callable entrypoint found in {main_path}.\n"
        "Expected one of:\n"
        " - run_simulation(epsilon, seed, epochs, cci_target, eta_target)\n"
        " - simulate(...)\n"
        " - run(...)\n"
        " - Simulation()/Simulator()/World().run(...)\n"
        "Please expose one of these, or set SIM_MAIN_PATH to a module that does."
    )
