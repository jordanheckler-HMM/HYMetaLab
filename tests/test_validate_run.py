from sim.validate_run import validate_run
from smoke_run import tiny_simulation


def test_validate_run_smoke(tmp_path):
    tmp = tmp_path / "run"
    tmp.mkdir()
    out = tiny_simulation(seed=123, export_dir=str(tmp))
    res = validate_run(str(tmp))
    # decisions should be present and valid
    assert "decisions.jsonl" in res
    assert res["decisions.jsonl"]["total"] == res["decisions.jsonl"]["valid"]
