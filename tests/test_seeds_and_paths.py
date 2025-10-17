"""Tests for seed parsing and path resolution in openlaws_automation."""

import sys
from pathlib import Path

# Add parent directory to path to import openlaws_automation
sys.path.insert(0, str(Path(__file__).parent.parent))

from openlaws_automation import (
    DEFAULT_SEEDS,
    get_seeds,
    normalize_study_cfg,
    resolve_results_dir,
)


def test_get_seeds_from_constants():
    """Test that seeds are correctly extracted from constants.seeds."""
    cfg = {"study_id": "test_study", "constants": {"seeds": [11, 17, 23, 29]}}
    norm = normalize_study_cfg(cfg, Path("dummy.yml"))
    assert norm["__seeds"] == [
        11,
        17,
        23,
        29,
    ], f"Expected [11,17,23,29], got {norm['__seeds']}"
    print("✅ test_get_seeds_from_constants passed")


def test_get_seeds_from_top_level():
    """Test that seeds are correctly extracted from top-level seeds key."""
    cfg = {"study_id": "test_study", "seeds": [42, 43, 44]}
    seeds = get_seeds(cfg)
    assert seeds == [42, 43, 44], f"Expected [42,43,44], got {seeds}"
    print("✅ test_get_seeds_from_top_level passed")


def test_get_seeds_fallback_default():
    """Test that default seeds are used when no seeds specified."""
    cfg = {"study_id": "test_study"}
    norm = normalize_study_cfg(cfg, Path("dummy.yml"))
    assert (
        norm["__seeds"] == DEFAULT_SEEDS
    ), f"Expected {DEFAULT_SEEDS}, got {norm['__seeds']}"
    print("✅ test_get_seeds_fallback_default passed")


def test_get_seeds_priority():
    """Test that constants.seeds takes priority over top-level seeds."""
    cfg = {
        "study_id": "test_study",
        "seeds": [1, 2, 3],
        "constants": {"seeds": [11, 17, 23, 29]},
    }
    seeds = get_seeds(cfg)
    assert seeds == [11, 17, 23, 29], f"Expected [11,17,23,29], got {seeds}"
    print("✅ test_get_seeds_priority passed")


def test_resolve_results_dir_timestamped(tmp_path, monkeypatch):
    """Test that resolve_results_dir finds timestamped directories."""
    # Create a timestamped directory
    base = tmp_path / "discovery_results"
    timestamped = base / "phaseX_20251014_010203"
    timestamped.mkdir(parents=True)

    # Change to tmp directory
    monkeypatch.chdir(tmp_path)

    # Should find the timestamped directory
    # Update ROOT for this test
    import openlaws_automation

    old_root = openlaws_automation.ROOT
    openlaws_automation.ROOT = tmp_path

    try:
        p = resolve_results_dir("phaseX", expect_writable=False)
        assert "phaseX" in str(p.name), f"Expected phaseX in path, got {p}"
        print("✅ test_resolve_results_dir_timestamped passed")
    finally:
        openlaws_automation.ROOT = old_root


def test_resolve_results_dir_create_canonical(tmp_path, monkeypatch):
    """Test that resolve_results_dir creates canonical dir when expect_writable=True."""
    monkeypatch.chdir(tmp_path)

    import openlaws_automation

    old_root = openlaws_automation.ROOT
    openlaws_automation.ROOT = tmp_path

    try:
        p = resolve_results_dir("new_study", expect_writable=True)
        assert p.exists(), f"Directory should have been created: {p}"
        assert (
            p.name == "new_study"
        ), f"Expected canonical name 'new_study', got {p.name}"
        print("✅ test_resolve_results_dir_create_canonical passed")
    finally:
        openlaws_automation.ROOT = old_root


if __name__ == "__main__":

    # Run tests manually (pytest-style with fixtures)
    print("Running tests manually...")

    # Simple tests without fixtures
    test_get_seeds_from_constants()
    test_get_seeds_from_top_level()
    test_get_seeds_fallback_default()
    test_get_seeds_priority()

    # Tests with fixtures need pytest
    print("\nTests requiring pytest fixtures:")
    print("  - test_resolve_results_dir_timestamped")
    print("  - test_resolve_results_dir_create_canonical")
    print("\nRun with: pytest tests/test_seeds_and_paths.py -v")
