import tempfile
from pathlib import Path

from experiments.belief_experiment import run_belief_simulation


def test_belief_quick(tmp_dir: Path = None):
    if tmp_dir is None:
        tmp_dir = Path(tempfile.mkdtemp())
    out = run_belief_simulation(
        n_agents=10, n_steps=10, seed=1, output_dir=Path(tmp_dir) / "belief"
    )
    assert Path(out).exists()


if __name__ == "__main__":
    test_belief_quick()
