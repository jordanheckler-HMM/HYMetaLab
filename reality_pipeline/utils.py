import importlib.util
import json
import logging
import os
import pathlib
import shlex
import subprocess
import sys
from typing import Any

import pandas as pd


def get_logger():
    lvl = os.environ.get("REALITY_PIPELINE_LOG", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, lvl, logging.INFO), format="%(levelname)s | %(message)s"
    )
    return logging.getLogger("reality_pipeline")


log = get_logger()


def safe_mkdir(path: str):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def write_json(obj, path: str):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def write_csv(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)


def rglob_module_paths(filename: str, roots=None):
    roots = roots or [os.getcwd()]
    out = []
    for root in roots:
        for p in pathlib.Path(root).rglob(filename):
            if p.name == filename:
                out.append(str(p))
    return out


def import_from_path(file_path: str):
    name = pathlib.Path(file_path).stem
    spec = importlib.util.spec_from_file_location(name, file_path)
    if not spec or not spec.loader:
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def try_call(
    mod,
    preferred: list[tuple[str, bool]],
    params: dict[str, Any],
    outdir: str,
    fallback_path: str | None,
):
    """
    preferred: list of (func_name, accepts_params) in priority order.
    """
    for fname, accepts_params in preferred:
        fn = getattr(mod, fname, None)
        if callable(fn):
            log.info(
                f"Calling {mod.__name__}.{fname} (accepts_params={accepts_params})"
            )
            try:
                if accepts_params:
                    return (
                        fn(params, outdir=outdir)
                        if "outdir" in fn.__code__.co_varnames
                        else fn(params)
                    )
                else:
                    return fn()
            except Exception as e:
                log.warning(f"Function {fname} raised: {e}")

    # fallback: subprocess
    if fallback_path:
        try:
            cmd = [sys.executable, fallback_path]
            log.info(
                f"Fallback subprocess: {' '.join(shlex.quote(arg) for arg in cmd)}"
            )
            subprocess.run(cmd, check=False)
        except Exception as e:
            log.warning(f"Subprocess fallback failed: {e}")
