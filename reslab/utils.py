import ast
import glob
import importlib
import json
import os
import random

import numpy as np


def ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def save_json(obj, path):
    ensure_dir(path)
    json.dump(obj, open(path, "w"), indent=2)


def load_yaml(path):
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)


def merge_cfg(paths):
    cfg = {}
    for p in paths:
        c = load_yaml(p)
        if "include" in c:
            cfg |= load_yaml(c["include"])
            del c["include"]
        cfg |= c
    return cfg


def find_candidates():
    hits = []
    for py in glob.glob("**/*.py", recursive=True):
        if py.startswith("reslab/"):
            continue
        try:
            with open(py, encoding="utf-8") as f:
                import ast as _ast

                tree = _ast.parse(f.read(), filename=py)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    name = node.name.lower()
                    if any(
                        k in name
                        for k in ["run_sim", "simulate", "run", "step", "main"]
                    ):
                        hits.append((py, node.name))
        except Exception:
            pass
    return hits


def import_attr(module_path, attr_path):
    mod = importlib.import_module(module_path)
    obj = mod
    for part in attr_path.split("."):
        obj = getattr(obj, part)
    return obj
