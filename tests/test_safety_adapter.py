from adapters import safety_cci_adapter as adapter


def test_rejects_without_allow_mocks(tmp_path, monkeypatch):
    cfg = {
        "sweep": {
            "agent_count": [2],
            "noise": [0.02],
            "shock_severity": [0.1],
            "goal_diversity": [1],
            "openness": [{"label": "openA", "mechanism": "test", "epsilon": 0.01}],
        },
        "seed_policy": {"seeds": [1]},
    }
    out_dir = tmp_path / "out"
    try:
        adapter.run_study(cfg, out_dir=str(out_dir))
        assert False, "Expected RuntimeError when mocks are disabled"
    except RuntimeError:
        pass


def test_allows_with_allow_mocks(tmp_path):
    cfg = {
        "sweep": {
            "agent_count": [2],
            "noise": [0.02],
            "shock_severity": [0.1],
            "goal_diversity": [1],
            "openness": [{"label": "openA", "mechanism": "test", "epsilon": 0.01}],
        },
        "seed_policy": {"seeds": [1]},
        "allow_mocks": True,
    }
    out_dir = tmp_path / "out"
    res = adapter.run_study(cfg, out_dir=str(out_dir))
    # provenance should exist and show used_mocks true
    prov = out_dir / "data" / "provenance.json"
    assert prov.exists()
    import json

    p = json.loads(prov.read_text())
    assert p.get("used_mocks", False) is True


def test_uses_sim_entry(tmp_path, monkeypatch):
    # point to our dummy sim
    cfg = {
        "sweep": {
            "agent_count": [2],
            "noise": [0.02],
            "shock_severity": [0.1],
            "goal_diversity": [1],
            "openness": [{"label": "openA", "mechanism": "test", "epsilon": 0.01}],
        },
        "seed_policy": {"seeds": [1]},
        "sim_entry": "tests.dummy_sim:run_sim",
    }
    out_dir = tmp_path / "out"
    res = adapter.run_study(cfg, out_dir=str(out_dir))
    prov = out_dir / "data" / "provenance.json"
    assert prov.exists()
    import json

    p = json.loads(prov.read_text())
    # when sim_entry is used, used_mocks should be False
    assert p.get("used_mocks", True) is False
