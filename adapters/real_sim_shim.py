from __future__ import annotations

import csv
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
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


def _run(
    cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None
) -> str:
    p = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    if p.returncode != 0:
        raise RuntimeError(
            f"Command failed ({p.returncode}): {' '.join(cmd)}\n--- OUTPUT ---\n{p.stdout}"
        )
    return p.stdout


def _parse_json(p: Path) -> dict[str, Any]:
    return json.loads(p.read_text())


def _parse_csv(p: Path) -> dict[str, Any]:
    with p.open() as f:
        r = list(csv.DictReader(f))
    if not r:
        raise RuntimeError(f"CSV {p} has no rows")
    return r[0]


def _pick(keys: dict[str, Any], *candidates: str, required: bool = True, cast=float):
    for k in candidates:
        if k in keys and keys[k] not in (None, ""):
            try:
                return cast(keys[k])
            except Exception:
                return cast(str(keys[k]).strip())
    if required:
        raise KeyError(f"Missing required metric; tried keys: {candidates}")
    return None


def run_simulation(
    *, epsilon: float, seed: int, epochs: int, cci_target: float, eta_target: float
) -> SimResult:
    """Run the real sim via subprocess and return normalized metrics.

    Environment variables honored:
      SIM_PYTHON, SIM_MAIN_PATH, SIM_ARGS_TEMPLATE, SIM_ANALYZE_CMD,
      SIM_EXPECT_JSON, SIM_SUMMARY_BASENAME, SIM_KEY_*

    Behavior:
      - If SIM_ARGS_TEMPLATE contains placeholders for epsilon/cci/eta we format them in.
      - Otherwise we build a safe argv list and append --epsilon/--cci/--eta. If the sim
        rejects those arguments we retry without them and inject SIM_EPSILON/SIM_CCI_TARGET/SIM_ETA_TARGET env vars.
      - We then attempt to parse a JSON or CSV summary; if keys are missing we compute a best-effort fallback from CSV logs.
    """
    py = os.environ.get("SIM_PYTHON", "").strip() or sys.executable
    main_path = Path(os.environ.get("SIM_MAIN_PATH", "main.py")).resolve()
    if not main_path.exists():
        raise FileNotFoundError(f"SIM main not found: {main_path}")

    session_dir = Path(tempfile.mkdtemp(prefix="sim31b_"))
    outdir = session_dir / "logs"
    outdir.mkdir(parents=True, exist_ok=True)

    args_tpl = os.environ.get(
        "SIM_ARGS_TEMPLATE",
        "{python} {main} --seed {seed} --ticks {epochs} --log-dir {outdir}",
    )

    fmt_kwargs = dict(
        python=py, main=str(main_path), seed=seed, epochs=epochs, outdir=str(outdir)
    )
    expects_params = False
    if "{epsilon" in args_tpl:
        fmt_kwargs["epsilon"] = epsilon
        expects_params = True
    if "{cci_target" in args_tpl:
        fmt_kwargs["cci_target"] = cci_target
        expects_params = True
    if "{eta_target" in args_tpl:
        fmt_kwargs["eta_target"] = eta_target
        expects_params = True

    if expects_params:
        cmd_str = args_tpl.format(**fmt_kwargs)
        base_cmd = shlex.split(cmd_str)
        extra: list[str] = []
    else:
        base_cmd = [
            py,
            str(main_path),
            "--seed",
            str(seed),
            "--ticks",
            str(epochs),
            "--log-dir",
            str(outdir),
        ]
        extra = [
            "--epsilon",
            str(epsilon),
            "--cci",
            str(cci_target),
            "--eta",
            str(eta_target),
        ]

    full_cmd = base_cmd + extra
    try:
        print("[shim] CMD:", " ".join(full_cmd), flush=True)
    except Exception:
        pass

    sim_cwd = main_path.parent
    try:
        sim_out = _run(full_cmd, cwd=sim_cwd)
    except RuntimeError as e:
        out = str(e)
        if (
            "unrecognized arguments" in out
            or "unknown arguments" in out
            or "usage:" in out
        ):
            # Retry without tuned CLI flags; pass targets via env
            retry_base = base_cmd
            env = os.environ.copy()
            env.update(
                {
                    "SIM_EPSILON": str(epsilon),
                    "SIM_CCI_TARGET": str(cci_target),
                    "SIM_ETA_TARGET": str(eta_target),
                }
            )
            sim_out = _run(retry_base, cwd=sim_cwd, env=env)
        else:
            raise

    analyze_tpl = os.environ.get("SIM_ANALYZE_CMD", "").strip()
    summary_base = os.environ.get("SIM_SUMMARY_BASENAME", "summary")
    prefer_json = os.environ.get("SIM_EXPECT_JSON", "").lower() in ("1", "true", "yes")

    if analyze_tpl:
        analyze_cmd = analyze_tpl.format(python=py, outdir=str(outdir))
        _run(shlex.split(analyze_cmd), cwd=sim_cwd)

    candidates: list[Path] = []
    if prefer_json:
        candidates += [outdir / f"{summary_base}.json", outdir / "metrics.json"]
    candidates += [
        outdir / "metrics.json",
        outdir / f"{summary_base}.json",
        outdir / f"{summary_base}.csv",
    ]

    found: Path | None = None
    for p in candidates:
        if p.exists():
            found = p
            break
    if not found:
        for p in outdir.glob("*"):
            if p.suffix.lower() in (".json", ".csv"):
                found = p
                break
    if not found:
        raise FileNotFoundError(
            f"No summary found in {outdir}. Provide SIM_ANALYZE_CMD or make sim write summary.json/csv."
        )

    if found.suffix.lower() == ".json":
        d = _parse_json(found)
    else:
        d = _parse_csv(found)

    if isinstance(d.get("metrics"), dict):
        d = d["metrics"]
    if isinstance(d.get("summary"), dict):
        d = d["summary"]

    def _compute_from_logs(logdir: Path) -> dict[str, float]:
        outvals = {
            "resilience": 0.0,
            "survival_rate": 0.0,
            "hazard": 0.0,
            "cci": 0.0,
            "eta": 0.0,
        }
        wt = logdir / "world_tick.csv"
        if wt.exists():
            try:
                with wt.open() as fh:
                    r = list(csv.DictReader(fh))
                if r:
                    nums = []
                    for row in r:
                        v = row.get("num_agents") or row.get("num_agents", 0)
                        try:
                            nums.append(float(v))
                        except Exception:
                            nums.append(0.0)
                    ticks = list(range(len(nums)))
                    initial = nums[0] if nums else 0.0
                    final = nums[-1] if nums else 0.0
                    outvals["survival_rate"] = (final / initial) if initial else 0.0
                    if len(ticks) > 1 and initial:
                        x = ticks
                        y = nums
                        n = len(x)
                        sx = sum(x)
                        sy = sum(y)
                        sxy = sum(a * b for a, b in zip(x, y))
                        sx2 = sum(a * a for a in x)
                        denom = n * sx2 - sx * sx
                        slope = (n * sxy - sx * sy) / denom if denom else 0.0
                        outvals["resilience"] = slope / initial
            except Exception:
                pass
        ev = logdir / "events.csv"
        death_count = 0
        if ev.exists():
            try:
                with ev.open() as fh:
                    for row in csv.DictReader(fh):
                        t = row.get("type") or row.get("event") or ""
                        if str(t).lower().strip() == "death":
                            death_count += 1
            except Exception:
                pass
        if death_count and wt.exists():
            try:
                with wt.open() as fh:
                    rows = list(csv.DictReader(fh))
                ticks = len(rows) or 1
                initial = float(rows[0].get("num_agents") or 0) if rows else 0.0
                outvals["hazard"] = (
                    (death_count / (initial * ticks))
                    if initial
                    else (death_count / ticks)
                )
            except Exception:
                outvals["hazard"] = float(death_count)

        # cci/eta detection across CSVs
        for p in logdir.glob("*.csv"):
            try:
                with p.open() as fh:
                    rdr = csv.DictReader(fh)
                    cols = rdr.fieldnames or []
                    lower = [c.lower() for c in cols]
                    if any("cci" in c or "coherence" in c for c in lower):
                        vals = []
                        for row in rdr:
                            for k in cols:
                                if "cci" in (k.lower() or "") or "coherence" in (
                                    k.lower() or ""
                                ):
                                    try:
                                        vals.append(float(row.get(k) or 0.0))
                                    except Exception:
                                        pass
                        if vals:
                            outvals["cci"] = sum(vals) / len(vals)
                            break
            except Exception:
                continue
        for p in logdir.glob("*.csv"):
            try:
                with p.open() as fh:
                    rdr = csv.DictReader(fh)
                    cols = rdr.fieldnames or []
                    if any(
                        "eta" in (c.lower() or "") or "entropy" in (c.lower() or "")
                        for c in cols
                    ):
                        vals = []
                        for row in rdr:
                            for k in cols:
                                if "eta" in (k.lower() or "") or "entropy" in (
                                    k.lower() or ""
                                ):
                                    try:
                                        vals.append(float(row.get(k) or 0.0))
                                    except Exception:
                                        pass
                        if vals:
                            outvals["eta"] = sum(vals) / len(vals)
                            break
            except Exception:
                continue
        return outvals

    R_key = os.environ.get("SIM_KEY_RESILIENCE", "resilience")
    S_key = os.environ.get("SIM_KEY_SURVIVAL", "survival_rate")
    H_key = os.environ.get("SIM_KEY_HAZARD", "hazard")
    CCI_key = os.environ.get("SIM_KEY_CCI", "cci")
    ETA_key = os.environ.get("SIM_KEY_ETA", "eta")

    try:
        res = SimResult(
            resilience=_pick(d, R_key, "R", "res", cast=float),
            survival_rate=_pick(d, S_key, "survival", "S", cast=float),
            hazard=_pick(d, H_key, "H", cast=float),
            cci=_pick(d, CCI_key, "CCI", "coherence_index", cast=float),
            eta=_pick(d, ETA_key, "entropy_coupling", "eta_c", cast=float),
        )
    except KeyError:
        computed = _compute_from_logs(outdir)
        res = SimResult(
            resilience=float(computed.get("resilience") or 0.0),
            survival_rate=float(computed.get("survival_rate") or 0.0),
            hazard=float(computed.get("hazard") or 0.0),
            cci=float(computed.get("cci") or 0.0),
            eta=float(computed.get("eta") or 0.0),
        )

    try:
        shutil.rmtree(session_dir)
    except Exception:
        pass
    return res

    # adapter can continue. This helps with sims that don't emit a tidy JSON
    # summary but do emit detailed CSV logs.
    def _compute_from_logs(logdir: Path) -> dict[str, float]:
        out: dict[str, float] = {
            "resilience": 0.0,
            "survival_rate": 0.0,
            "hazard": 0.0,
            "cci": 0.0,
            "eta": 0.0,
        }
        # world_tick.csv -> num_agents over time
        wt = logdir / "world_tick.csv"
        if wt.exists():
            try:
                with wt.open() as fh:
                    r = list(csv.DictReader(fh))
                if r:
                    nums = [
                        float(row.get("num_agents") or row.get("num_agents", 0))
                        for row in r
                    ]
                    ticks = [int(row.get("tick") or i) for i, row in enumerate(r)]
                    initial = nums[0] if nums else 0.0
                    final = nums[-1] if nums else 0.0
                    out["survival_rate"] = (final / initial) if initial else 0.0
                    # linear slope (num_agents vs tick) normalized by initial size
                    if len(ticks) > 1 and initial:
                        x = ticks
                        y = nums
                        n = len(x)
                        sx = sum(x)
                        sy = sum(y)
                        sxy = sum(a * b for a, b in zip(x, y))
                        sx2 = sum(a * a for a in x)
                        denom = n * sx2 - sx * sx
                        if denom:
                            slope = (n * sxy - sx * sy) / denom
                        else:
                            slope = 0.0
                        out["resilience"] = slope / initial
            except Exception:
                pass

        # events.csv -> count deaths
        ev = logdir / "events.csv"
        death_count = 0
        if ev.exists():
            try:
                with ev.open() as fh:
                    for row in csv.DictReader(fh):
                        t = row.get("type") or row.get("event") or ""
                        if str(t).lower().strip() == "death":
                            death_count += 1
            except Exception:
                pass
        if death_count and wt.exists():
            try:
                with wt.open() as fh:
                    rows = list(csv.DictReader(fh))
                ticks = len(rows) or 1
                initial = float(rows[0].get("num_agents") or 0) if rows else 0.0
                out["hazard"] = (
                    (death_count / (initial * ticks))
                    if initial
                    else (death_count / ticks)
                )
            except Exception:
                out["hazard"] = float(death_count)

        # Try to discover CCI/ETA-like columns in any CSV and average them
        for p in logdir.glob("*.csv"):
            try:
                with p.open() as fh:
                    rdr = csv.DictReader(fh)
                    cols = rdr.fieldnames or []
                    lower = [c.lower() for c in cols]
                    # look for candidate columns
                    if any("cci" in c for c in lower) or any(
                        "coherence" in c for c in lower
                    ):
                        vals = []
                        for row in rdr:
                            for k in cols:
                                if "cci" in (k.lower() or "") or "coherence" in (
                                    k.lower() or ""
                                ):
                                    try:
                                        vals.append(float(row.get(k) or 0.0))
                                    except Exception:
                                        pass
                        if vals:
                            out["cci"] = sum(vals) / len(vals)
                            break
            except Exception:
                continue

        # ETA-like search
        for p in logdir.glob("*.csv"):
            try:
                with p.open() as fh:
                    rdr = csv.DictReader(fh)
                    cols = rdr.fieldnames or []
                    lower = [c.lower() for c in cols]
                    if any("eta" in c for c in lower) or any(
                        "entropy" in c for c in lower
                    ):
                        vals = []
                        for row in rdr:
                            for k in cols:
                                if "eta" in (k.lower() or "") or "entropy" in (
                                    k.lower() or ""
                                ):
                                    try:
                                        vals.append(float(row.get(k) or 0.0))
                                    except Exception:
                                        pass
                        if vals:
                            out["eta"] = sum(vals) / len(vals)
                            break
            except Exception:
                continue

        return out

    R_key = os.environ.get("SIM_KEY_RESILIENCE", "resilience")
    S_key = os.environ.get("SIM_KEY_SURVIVAL", "survival_rate")
    H_key = os.environ.get("SIM_KEY_HAZARD", "hazard")
    CCI_key = os.environ.get("SIM_KEY_CCI", "cci")
    ETA_key = os.environ.get("SIM_KEY_ETA", "eta")

    if isinstance(d.get("metrics"), dict):
        d = d["metrics"]
    if isinstance(d.get("summary"), dict):
        d = d["summary"]

    try:
        res = SimResult(
            resilience=_pick(d, R_key, "R", "res", cast=float),
            survival_rate=_pick(d, S_key, "survival", "S", cast=float),
            hazard=_pick(d, H_key, "H", cast=float),
            cci=_pick(d, CCI_key, "CCI", "coherence_index", cast=float),
            eta=_pick(d, ETA_key, "entropy_coupling", "eta_c", cast=float),
        )
    except KeyError:
        # Best-effort fallback: compute from CSV logs in outdir
        computed = _compute_from_logs(outdir)
        res = SimResult(
            resilience=float(computed.get("resilience") or 0.0),
            survival_rate=float(computed.get("survival_rate") or 0.0),
            hazard=float(computed.get("hazard") or 0.0),
            cci=float(computed.get("cci") or 0.0),
            eta=float(computed.get("eta") or 0.0),
        )
    try:
        shutil.rmtree(session_dir)
    except Exception:
        pass
    return res
