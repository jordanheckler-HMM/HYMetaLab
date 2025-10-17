import yaml

from adapters.universal_coherence_adapter import run_agents, run_civ

cfg = yaml.safe_load(open("studies/phase31b_universal_coherence_law.yml"))
common = cfg["common"]
suite = cfg["tests"]
for dom, fn in [("agents", run_agents), ("civ", run_civ)]:
    print(f"[UCLV] Running domain: {dom}", flush=True)
    fn(
        {
            "seeds": common["seeds"],
            "epochs": common["epochs"],
            "bootstrap": common["bootstrap"],
            "suite": suite,
        }
    )
print("[UCLV] Fallback run complete.")
