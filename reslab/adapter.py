"""
Adapter for YOUR sim without touching it.
1) Use configs/adapter_mapping.yaml if filled.
2) Else, auto-detect a likely function (run_sim/simulate/run/step/main) and pass accepted kwargs.
We write a small report to reslab/exports/adapter_report.json describing what was used.
"""

import inspect

from .utils import ensure_dir, find_candidates, import_attr, load_yaml, save_json


def _from_mapping():
    try:
        m = load_yaml("configs/adapter_mapping.yaml")
    except Exception:
        return None
    if not m or not m.get("module") or not m.get("func"):
        return None
    return m


def _extract_returns(obj, retmap):
    if hasattr(obj, "keys"):
        get = lambda k: obj.get(k, None)
    else:
        get = lambda k: getattr(obj, k, None)
    return {
        "survived": get(retmap.get("survived", "survived")),
        "calibration": get(retmap.get("calibration", "calibration")),
        "coherence": get(retmap.get("coherence", "coherence")),
        "emergence": get(retmap.get("emergence", "emergence")),
        "noise": get(retmap.get("noise", "noise")),
        "misalign_rate": get(retmap.get("misalign_rate", "misalign_rate")),
    }


def run_sim(**params):
    ensure_dir("reslab/exports/dummy.txt")
    # path only to ensure folder exists
    mapping = _from_mapping()
    report = {}
    if mapping:
        fn = import_attr(mapping["module"], mapping["func"])
        sig = inspect.signature(fn)
        kwargs = {
            mapping["argmap"].get(k, k): v
            for k, v in params.items()
            if mapping["argmap"].get(k, k) in sig.parameters
        }
        out = fn(**kwargs)
        ret = _extract_returns(out, mapping.get("returns", {}))
        report = {
            "mode": "mapping",
            "module": mapping["module"],
            "func": mapping["func"],
            "accepted_kwargs": list(kwargs.keys()),
            "returns_found": {k: (ret[k] is not None) for k in ret},
        }
    else:
        cands = find_candidates()
        if not cands:
            raise SystemExit(
                "Adapter could not auto-detect a sim entrypoint. Fill configs/adapter_mapping.yaml."
            )
        py, name = cands[0]
        mod = py[:-3].replace("/", ".")
        fn = import_attr(mod, name)
        sig = inspect.signature(fn)
        kwargs = {k: v for k, v in params.items() if k in sig.parameters}
        out = fn(**kwargs)
        ret = _extract_returns(out, {})
        report = {
            "mode": "autodetect",
            "module": mod,
            "func": name,
            "accepted_kwargs": list(kwargs.keys()),
            "returns_found": {k: (ret[k] is not None) for k in ret},
        }
    save_json(report, "reslab/exports/adapter_report.json")
    return ret
