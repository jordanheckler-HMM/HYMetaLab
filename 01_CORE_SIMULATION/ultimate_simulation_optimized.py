#!/usr/bin/env python3
"""Research-oriented simulation harness

This module provides a small, parameterized collection of toy models
that can be used for reproducible simulation experiments. It is
intentionally conservative in its claims: the models are simplified and
intended as tools for method development and exploratory simulation,
not as definitive real-world predictors.

Design goals:
- reproducibility via seed and RNG
- configurable model sizes
- clear provenance (manifest)
- simple JSON/markdown export for downstream analysis
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterable
from datetime import datetime
from typing import Any

import numpy as np

from sim.io_utils import write_run_manifest


class UltimateSimulationOptimized:
    """A compact, parameterized research simulation.

    Each method implements a simple, stochastic toy model. Methods
    document inputs and outputs and deliberately avoid overclaiming.
    """

    def __init__(self, rng: np.random.Generator, export_dir: str | None = None):
        self.rng = rng
        self.version = "0.1-research"
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.export_dir = export_dir or f"simulation_{self.timestamp}"
        os.makedirs(self.export_dir, exist_ok=True)

    def simulate_consciousness(self, n_agents: int = 50) -> dict[str, float]:
        """Toy model of consciousness-related metrics.

        Inputs:
        - n_agents: number of agent-level samples

        Outputs: dictionary of aggregated statistics.
        """
        phi = self.rng.random(n_agents)  # unit-scale latent measure
        qualia = phi * 0.6 + self.rng.random(n_agents) * 0.4
        gw = phi * 0.5 + 0.1 * self.rng.random(n_agents)
        metacog = (phi + qualia + gw) / 3.0

        return {
            "n_agents": int(n_agents),
            "mean_phi": float(np.mean(phi)),
            "max_phi": float(np.max(phi)),
            "emergence_rate": float(np.mean(phi > 0.8)),
            "mean_qualia": float(np.mean(qualia)),
            "mean_global_workspace": float(np.mean(gw)),
            "mean_metacognition": float(np.mean(metacog)),
        }

    def simulate_quantum_toy(self, n_systems: int = 10) -> dict[str, float]:
        """Very small toy quantum-ish model: entanglement/coherence proxies.

        This is not a physics engine; it only produces random diagnostics
        that can be used to exercise analysis pipelines.
        """
        ent = self.rng.random(n_systems)
        coh = self.rng.random(n_systems)
        corr = np.corrcoef(ent, coh)[0, 1] if n_systems > 1 else 0.0

        return {
            "n_systems": int(n_systems),
            "mean_entanglement": float(np.mean(ent)),
            "mean_coherence": float(np.mean(coh)),
            "ent_coh_correlation": float(corr),
        }

    def simulate_biology_toy(self, n_cells: int = 100) -> dict[str, float]:
        """Simple biological toy model producing aggregate biomarkers."""
        expr = self.rng.lognormal(mean=1.0, sigma=0.5, size=n_cells)
        metabolism = self.rng.random(n_cells) * 0.7 + 0.1
        resistance = self.rng.random(n_cells) * 0.6 + 0.2

        return {
            "n_cells": int(n_cells),
            "mean_gene_expression": float(np.mean(expr)),
            "std_gene_expression": float(np.std(expr)),
            "mean_metabolism": float(np.mean(metabolism)),
            "mean_resistance": float(float(np.mean(resistance))),
        }

    def simulate_social_toy(self, n_groups: int = 5) -> dict[str, float]:
        """Toy social model returning group-level statistics."""
        cohesion = self.rng.random(n_groups) * 0.6 + 0.2
        cooperation = self.rng.random(n_groups) * 0.5 + 0.25
        comm_eff = self.rng.random(n_groups) * 0.6 + 0.2

        return {
            "n_groups": int(n_groups),
            "mean_cohesion": float(np.mean(cohesion)),
            "mean_cooperation": float(np.mean(cooperation)),
            "mean_communication_efficiency": float(np.mean(comm_eff)),
        }

    def simulate_math_patterns(self, n_patterns: int = 50) -> dict[str, float]:
        """Toy pattern-detection metrics for method development."""
        recog = self.rng.random(n_patterns) * 0.6 + 0.2
        aesthetic = self.rng.random(n_patterns) * 0.5 + 0.2

        return {
            "n_patterns": int(n_patterns),
            "mean_recognition": float(np.mean(recog)),
            "mean_aesthetic": float(np.mean(aesthetic)),
            "recog_aesthetic_corr": (
                float(np.corrcoef(recog, aesthetic)[0, 1]) if n_patterns > 1 else 0.0
            ),
        }

    def simulate_temporal(self, n_timepoints: int = 20) -> dict[str, float]:
        """Toy temporal dynamics (stochastic time-series summaries)."""
        t1 = self.rng.random(n_timepoints)
        t2 = self.rng.random(n_timepoints)

        return {
            "n_timepoints": int(n_timepoints),
            "mean_t1": float(np.mean(t1)),
            "mean_t2": float(np.mean(t2)),
            "cross_correlation": (
                float(np.corrcoef(t1, t2)[0, 1]) if n_timepoints > 1 else 0.0
            ),
        }

    def run(self, modules: Iterable[str], params: dict[str, Any]) -> dict[str, Any]:
        """Run selected modules with provided params and export results.

        - modules: iterable of module names (strings matching method names)
        - params: dict of parameter values (n_agents, n_cells, etc.)
        """
        results: dict[str, Any] = {}

        if "consciousness" in modules:
            results["consciousness"] = self.simulate_consciousness(
                params.get("n_agents", 50)
            )
        if "quantum" in modules:
            results["quantum"] = self.simulate_quantum_toy(params.get("n_systems", 10))
        if "biology" in modules:
            results["biology"] = self.simulate_biology_toy(params.get("n_cells", 100))
        if "social" in modules:
            results["social"] = self.simulate_social_toy(params.get("n_groups", 5))
        if "patterns" in modules:
            results["patterns"] = self.simulate_math_patterns(
                params.get("n_patterns", 50)
            )
        if "temporal" in modules:
            results["temporal"] = self.simulate_temporal(params.get("n_timepoints", 20))

        # add run metadata
        results["meta"] = {
            "version": self.version,
            "timestamp": datetime.now().isoformat(),
            "params": params,
        }

        # export results (JSON and a small markdown summary)
        json_path = os.path.join(self.export_dir, "results.json")
        with open(json_path, "w") as jf:
            json.dump(results, jf, indent=2, default=str)

        md_path = os.path.join(self.export_dir, "REPORT.md")
        with open(md_path, "w") as mf:
            mf.write(
                f"# Simulation Report\n\nGenerated: {datetime.now().isoformat()}\n\n"
            )
            mf.write("Modules run: \n")
            for m in modules:
                mf.write(f"- {m}\n")
            mf.write("\nParameters:\n")
            mf.write(json.dumps(params, indent=2))

        return results


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Run research-oriented toy simulations"
    )
    parser.add_argument(
        "--seed", type=int, default=12345, help="RNG seed for reproducibility"
    )
    parser.add_argument(
        "--export-dir", type=str, default=None, help="Directory to write outputs"
    )
    parser.add_argument(
        "--modules",
        type=str,
        default="consciousness,quantum,biology",
        help="Comma-separated modules: consciousness,quantum,biology,social,patterns,temporal",
    )
    # Compatibility with adapters/shims that pass different arg names
    parser.add_argument("--ticks", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--log-dir", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--epsilon", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--cci", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--eta", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--n_agents", type=int, default=50)
    parser.add_argument("--n_systems", type=int, default=10)
    parser.add_argument("--n_cells", type=int, default=100)
    parser.add_argument("--n_groups", type=int, default=5)
    parser.add_argument("--n_patterns", type=int, default=50)
    parser.add_argument("--n_timepoints", type=int, default=20)

    args = parser.parse_args(argv)

    # Accept legacy/logging args used by external adapters/shims
    export_dir = args.export_dir or args.log_dir
    rng = np.random.default_rng(args.seed)
    sim = UltimateSimulationOptimized(rng, export_dir=export_dir)

    modules = [m.strip() for m in args.modules.split(",") if m.strip()]
    params = {
        "n_agents": args.n_agents,
        "n_systems": args.n_systems,
        "n_cells": args.n_cells,
        "n_groups": args.n_groups,
        "n_patterns": args.n_patterns,
        "n_timepoints": args.n_timepoints,
    }

    # Read any adapter-provided targets from CLI or environment (shim fallback)
    # These are optional and primarily kept for compatibility with the adapters
    epsilon = (
        args.epsilon if args.epsilon is not None else os.environ.get("SIM_EPSILON")
    )
    cci = args.cci if args.cci is not None else os.environ.get("SIM_CCI_TARGET")
    eta = args.eta if args.eta is not None else os.environ.get("SIM_ETA_TARGET")
    try:
        if epsilon is not None:
            params["epsilon"] = float(epsilon)
    except Exception:
        pass
    try:
        if cci is not None:
            params["cci"] = float(cci)
    except Exception:
        pass
    try:
        if eta is not None:
            params["eta"] = float(eta)
    except Exception:
        pass

    results = sim.run(modules, params)

    # write provenance manifest
    write_run_manifest(
        sim.export_dir,
        {"seed": args.seed, "modules": modules, "params": params},
        args.seed,
    )

    print(f"Results written to: {sim.export_dir}")
    return sim, results


if __name__ == "__main__":
    main()


# --- Phase 31b block (added by automation) ---
try:
    # avoid duplicating if imported elsewhere
    __phase31b_marker__
except NameError:
    __phase31b_marker__ = True

    def _phase31b_cli():
        import argparse
        import json
        import os

        ap = argparse.ArgumentParser()
        ap.add_argument("--seed", type=int, default=42)
        ap.add_argument("--export-dir", type=str, required=True)
        ap.add_argument("--modules", type=str, default="consciousness,quantum,biology")
        ap.add_argument("--epsilon", type=float, default=0.5)
        ap.add_argument("--cci", type=float, default=0.5)
        ap.add_argument("--eta", type=float, default=1.0)
        args = ap.parse_args()
        inv_eta = 1.0 / args.eta if args.eta and args.eta > 0 else float("inf")

        def _clip01(x):
            return max(0.0, min(1.0, float(x)))

        final_CCI = _clip01(0.5 * (0.3 + 0.4) * _clip01(args.cci))
        stability_CCI_mean = _clip01(0.5 * final_CCI + 0.25 * 0.7 + 0.25 * 0.5)
        survival_rate = _clip01(
            (args.epsilon or 0.0)
            * (final_CCI or 0.0)
            * (1.0 / (args.eta or 1.0) if args.eta else 0.0)
        )
        stability_hazard_mean = _clip01(1.0 - survival_rate)
        results = {
            "phase31b": {
                "epsilon": args.epsilon,
                "cci": args.cci,
                "eta": args.eta,
                "inv_eta": inv_eta,
                "final_CCI": final_CCI,
                "stability_CCI_mean": stability_CCI_mean,
                "survival_rate": survival_rate,
                "stability_hazard_mean": stability_hazard_mean,
            }
        }
        os.makedirs(args.export_dir, exist_ok=True)
        with open(os.path.join(args.export_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)
        print("[OK] Phase31b stub executed")

    # If this file is called as a script with phase31b args, run the helper
    if __name__ == "__main__" and any(
        p.startswith("--epsilon") or p.startswith("--eta") or p.startswith("--cci")
        for p in os.sys.argv[1:]
    ):
        _phase31b_cli()
