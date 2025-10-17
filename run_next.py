import datetime
import json
from pathlib import Path

from sim_ext.extended_sweep import run_extended

ROOT = Path("discovery_results")
ROOT.mkdir(exist_ok=True, parents=True)


def stamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


RUNS = []

# 0) FAST VALIDATION
RUNS.append(
    {
        "name": "validation_fast",
        "n_agents": [50],
        "timesteps": 300,
        "dt": 1.0,
        "shocks": [{"severity": 0.3, "timing": 150, "type": "external"}],
        "disease": {"R0": 1.5, "IFR": 0.6},
        "energy": {"softening": 0.05, "heat_loss": 0.02},
        "enable_uq": False,
        "enable_bayes": False,
        "seeds": [1],
    }
)

# A) MEDICAL REALISM (lesion, disease, repair)
RUNS.append(
    {
        "name": "med_realism_sweep_v1",
        "n_agents": [100, 200],
        "timesteps": 1000,  # Reduced for stability
        "dt": 1.0,
        "shocks": [
            {"severity": 0.2, "timing": 400, "type": "external"},
            {"severity": 0.5, "timing": 800, "type": "combo"},
        ],
        "disease": {"R0": [1.2, 2.0], "IFR": [0.2, 0.6]},  # Simplified
        "energy": {"softening": [0.01, 0.05], "heat_loss": [0.01, 0.05]},
        "noise": [0.0, 0.1],
        "enable_uq": True,
        "enable_bayes": False,  # Disabled for now
        "seeds": [111, 222],
    }
)

# B) CONSTRUCTIVE SHOCK THRESHOLDS (ethics & valence)
RUNS.append(
    {
        "name": "constructive_thresholds_v2",
        "n_agents": [100],
        "timesteps": 1000,  # Reduced for stability
        "dt": 1.0,
        "shocks": [
            {"severity": 0.1, "timing": 300, "type": "external"},
            {"severity": 0.3, "timing": 600, "type": "combo"},
        ],
        "ethics": {
            "rule_set": ["utilitarian", "deontic"],
            "mutation_rate": [0.0, 0.02],
        },  # Simplified
        "info": {"misinfo_rate": [0.0, 0.1], "trust_decay": [0.001, 0.01]},
        "multiscale": {"coupling_coeff": [0.2, 0.6], "cell_agents": 16},  # Simplified
        "valence_weighting": [0.3, 0.6],
        "enable_uq": True,
        "seeds": [7, 8],
    }
)

# C) INFO × NORMS (misinfo/ethics interaction)
RUNS.append(
    {
        "name": "info_ethics_interaction_v1",
        "n_agents": [200],
        "timesteps": 800,  # Reduced for stability
        "dt": 1.0,
        "info": {
            "misinfo_rate": [0.0, 0.1],
            "trust_decay": [0.001, 0.02],
        },  # Simplified
        "ethics": {
            "rule_set": ["utilitarian", "deontic"],
            "mutation_rate": [0.0, 0.02],
        },  # Simplified
        "shocks": [{"severity": 0.5, "timing": 400, "type": "external"}],
        "energy": {"softening": 0.01},
        "enable_uq": True,
        "seeds": [101, 202],
    }
)

# D) PHENOMENOLOGY / SELF-MODELING
RUNS.append(
    {
        "name": "phenomenology_identity_v1",
        "n_agents": [100],
        "timesteps": 1000,  # Reduced for stability
        "dt": 1.0,
        "shocks": [
            {"severity": 0.2, "timing": 300, "type": "external"},
            {"severity": 0.8, "timing": 600, "type": "combo"},
        ],
        "valence_weighting": [0.3, 0.6],  # Simplified
        "multiscale": {"coupling_coeff": [0.2, 0.9], "cell_agents": 8},
        "enable_uq": False,
        "seeds": [42, 43],
    }
)


def main():
    index = {}
    findings = {}

    for cfg in RUNS:
        print(f"🚀 Running experiment: {cfg['name']}")
        try:
            out = run_extended(cfg)
            if isinstance(out, dict) and "output_dir" in out:
                out_dir = Path(out["output_dir"])
            else:
                candidates = sorted(ROOT.glob(f"*_{cfg['name']}"))
                out_dir = candidates[-1] if candidates else ROOT
            index[cfg["name"]] = str(out_dir)

            # Extract findings from summary
            if isinstance(out, dict) and "summary" in out:
                summary = out["summary"]
                findings[cfg["name"]] = {
                    "simulations": summary.get("n_simulations", 0),
                    "successful": summary.get("successful_simulations", 0),
                    "energy_valid": summary.get("energy_valid_simulations", 0),
                    "avg_survival": summary.get("avg_survival_rate", 0),
                    "avg_cci": summary.get("avg_cci", 0),
                    "avg_valence": summary.get("avg_valence", 0),
                    "energy_drift": summary.get("avg_energy_drift", 0),
                }
            else:
                findings[cfg["name"]] = {
                    "status": "completed",
                    "details": "No summary available",
                }

            print(f"✅ Completed: {cfg['name']} → {out_dir}")

        except Exception as e:
            print(f"❌ Failed: {cfg['name']} - {e}")
            index[cfg["name"]] = "FAILED"
            findings[cfg["name"]] = {"status": "failed", "error": str(e)}

    # Save index
    with open(ROOT / "_index_next.json", "w") as f:
        json.dump(index, f, indent=2)

    # Build master report
    lines = ["# NEXT MASTER REPORT\n"]
    lines.append(f"_Generated: {datetime.datetime.now().isoformat()}_\n")
    lines.append("## Runs\n")
    for name, p in index.items():
        lines.append(f"- **{name}** → `{p}`")

    lines.append("\n## Findings Summary\n")
    for name, finding in findings.items():
        lines.append(f"### {name}")
        if finding.get("status") == "failed":
            lines.append(f"- ❌ **FAILED**: {finding.get('error', 'Unknown error')}")
        else:
            lines.append(f"- 📊 Simulations: {finding.get('simulations', 0)}")
            lines.append(f"- ✅ Successful: {finding.get('successful', 0)}")
            lines.append(f"- ⚡ Energy Valid: {finding.get('energy_valid', 0)}")
            lines.append(f"- 💚 Avg Survival: {finding.get('avg_survival', 0):.3f}")
            lines.append(f"- 🧠 Avg CCI: {finding.get('avg_cci', 0):.3f}")
            lines.append(f"- 😊 Avg Valence: {finding.get('avg_valence', 0):.3f}")
            lines.append(f"- ⚡ Energy Drift: {finding.get('energy_drift', 0):.3f}%")
        lines.append("")

    lines.append("## Figure Checklist\n")
    lines.append("- KM & hazard\n")
    lines.append("- R_eff\n")
    lines.append("- Energy drift & entropy flux\n")
    lines.append("- Ethics vs Collapse\n")
    lines.append("- Info accuracy phase\n")
    lines.append("- Multiscale coherence\n")
    lines.append("- Valence vs CCI\n")
    lines.append("- UQ tornado / CIs\n")

    (ROOT / "next_master_report.md").write_text("\n".join(lines))

    # Print final summary
    print("\n" + "=" * 60)
    print("🎯 EXPERIMENT SUITE COMPLETED")
    print("=" * 60)
    print(f"📁 Master Report: {ROOT.absolute()}/next_master_report.md")
    print(f"📊 Index File: {ROOT.absolute()}/_index_next.json")
    print("\n📋 FINDINGS SUMMARY:")
    for name, finding in findings.items():
        if finding.get("status") == "failed":
            print(f"  ❌ {name}: FAILED - {finding.get('error', 'Unknown error')}")
        else:
            print(
                f"  ✅ {name}: {finding.get('successful', 0)}/{finding.get('simulations', 0)} successful"
            )
            print(f"     💚 Survival: {finding.get('avg_survival', 0):.3f}")
            print(f"     🧠 CCI: {finding.get('avg_cci', 0):.3f}")
            print(f"     ⚡ Energy Drift: {finding.get('energy_drift', 0):.3f}%")


if __name__ == "__main__":
    main()
