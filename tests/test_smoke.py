import os

from smoke_run import tiny_simulation


def test_smoke_run_creates_manifest_and_valid_jsonl(tmp_path):
    export = tmp_path / "out"
    out = tiny_simulation(seed=999, export_dir=str(export))
    assert os.path.exists(out["manifest"])
    assert os.path.exists(out["decisions"])
    # validation should have at least some valid lines and zero invalid
    val = out["validation"]
    assert val["total"] == val["valid"]
    assert val["invalid"] == 0
