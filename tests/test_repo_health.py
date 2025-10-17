from pathlib import Path


def test_studies_exist():
    assert Path("studies").exists()


def test_results_dirs():
    assert Path("results/discovery_results").exists()
