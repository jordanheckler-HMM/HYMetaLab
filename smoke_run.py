"""Smoke-run: deterministic tiny simulation that writes a decisions.jsonl and run_manifest.json"""

import os
import random
from datetime import datetime

from sim.io_utils import validate_jsonl_against_schema, write_jsonl, write_run_manifest

ROOT = os.path.dirname(__file__)


def tiny_simulation(seed: int = 12345, export_dir: str = None):
    random.seed(seed)
    if export_dir is None:
        export_dir = os.path.join(
            ROOT, "outputs", f'smoke_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        )

    # config placeholder
    config = {"n_agents": 3, "ticks": 5, "seed": seed}
    manifest_path = write_run_manifest(export_dir, config, seed)

    # create tiny decisions log
    records = []
    for tick in range(5):
        for agent in range(3):
            rec = {
                "tick": tick,
                "agent_id": f"A-{agent}",
                "workspace_reads": random.randint(0, 2),
                "workspace_writes": random.randint(0, 1),
                "lookahead_depth": 0,
                "candidates": [{"action": "noop", "score": 0.5}],
                "chosen_action": "noop",
                "reported_conf": round(random.random(), 3),
                "outcome_reward": 0.0,
                "prediction_model_p": 0.5,
                "rng_seed_local": random.randint(0, 2**31 - 1),
            }
            records.append(rec)

    decisions_path = os.path.join(export_dir, "decisions.jsonl")
    write_jsonl(decisions_path, records)

    # validate
    schema_path = os.path.join(
        os.path.dirname(__file__), "schema", "decisions.schema.json"
    )
    validate_results = validate_jsonl_against_schema(decisions_path, schema_path)

    print("Manifest written to:", manifest_path)
    print("Decisions written to:", decisions_path)
    print("Validation:", validate_results)
    return {
        "manifest": manifest_path,
        "decisions": decisions_path,
        "validation": validate_results,
    }


if __name__ == "__main__":
    tiny_simulation()
