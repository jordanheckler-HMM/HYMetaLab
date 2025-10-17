import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.guardian_stub import run_guardian_checks


def test_guardian_scores_smoke():
    env = type("E", (object,), {"payload": {"prompt": "x", "output": "y"}})()
    scores, reasons = run_guardian_checks(env)
    assert 0.7 <= scores["coherence"] <= 1.0
    assert 0.05 <= scores["noise"] <= 0.35
    assert "policy:safety.general.pass" in reasons
