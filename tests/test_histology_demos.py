import shutil
import tempfile
from pathlib import Path

from experiments.histology_rework_qc import run_demo as run_rework
from experiments.histology_throughput import run_demo as run_throughput
from experiments.histology_timeline import run_demo as run_timeline


def test_run_all_histology_demos():
    tmp = Path(tempfile.mkdtemp())
    export = tmp / "histology_export"
    export.mkdir()

    t1 = Path(run_throughput(output_dir=tmp / "throughput", seed=2))
    t2 = Path(run_rework(output_dir=tmp / "rework", seed=3))
    t3 = Path(run_timeline(output_dir=tmp / "timeline", seed=4))

    # copy everything into the export folder
    for src in [t1, t2, t3]:
        dst = export / src.name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

    # basic assertions
    assert (export / t1.name / "summary.json").exists()
    assert (export / t2.name / "rework_qc_summary.json").exists()
    assert (export / t3.name / "timeline_summary.json").exists()

    print("Export created at", str(export))


if __name__ == "__main__":
    test_run_all_histology_demos()
