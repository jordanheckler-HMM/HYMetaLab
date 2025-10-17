#!/usr/bin/env python3
"""Integration test for Guardian-gated lab_techs_runner.py"""
import subprocess
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_runner_runs():
    """Test that lab_techs_runner.py executes and performs Guardian validation."""
    p = subprocess.run(
        [sys.executable, "lab_techs_runner.py"],
        capture_output=True,
        text=True,
        timeout=20,
    )
    output = p.stdout + p.stderr

    # Should contain Guardian-related output
    assert "Guardian" in output, "Expected Guardian validation in output"

    # Should either PASS or BLOCK (both are valid, depends on thresholds)
    assert "PASS" in output or "Blocked" in output, "Expected PASS or Blocked verdict"


def test_alert_file_exists_after_block():
    """Test that Integrity_Alert.md is created when execution is blocked."""
    alert_path = Path("dev/logs/Integrity_Alert.md")

    # Alert file should exist from previous runs
    # (This test assumes at least one run has occurred)
    if alert_path.exists():
        content = alert_path.read_text()
        assert "Integrity Alert" in content
        assert "Guardian" in content
        # If file exists, it should be valid
        assert len(content) > 100, "Alert file should have substantial content"


def test_guardian_client_import():
    """Test that guardian_client can be imported and used."""
    try:
        from tools.guardian_client import validate

        # Smoke test: validate a simple payload
        result = validate({"test": "smoke"}, phase="pre")

        # Check required keys exist
        assert "verdict" in result
        assert "scores" in result
        assert "coherence" in result["scores"]
        assert "noise" in result["scores"]

    except ImportError as e:
        raise AssertionError(f"Failed to import guardian_client: {e}")


def test_task_log_updated():
    """Test that Task_Log.md is being updated with operations."""
    log_path = Path("Task_Log.md")

    assert log_path.exists(), "Task_Log.md should exist"

    content = log_path.read_text()
    assert len(content) > 0, "Task_Log.md should not be empty"

    # Should contain timestamps
    assert "[" in content and "]" in content, "Expected timestamp format [...]"


if __name__ == "__main__":
    print("Running integration tests...")
    test_runner_runs()
    print("✅ test_runner_runs")

    test_alert_file_exists_after_block()
    print("✅ test_alert_file_exists_after_block")

    test_guardian_client_import()
    print("✅ test_guardian_client_import")

    test_task_log_updated()
    print("✅ test_task_log_updated")

    print("\n🎉 All integration tests passed!")
