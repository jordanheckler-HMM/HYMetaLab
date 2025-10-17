"""Run Kepler-pair demo, parameter sweeps over softening/dt/N, and produce analysis outputs."""

import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path

# allow running from repo root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.gravity_nbody import compute_forces, kinetic_energy, run_nbody

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def run_kepler_pair_demo(params=None, outdir=None, output_dir=None):
    # Handle parameter dict if passed
    if isinstance(params, dict):
        output_dir = params.get("output_dir", output_dir)

    # Use outdir if provided
    if outdir:
        output_dir = outdir

    # Construct a hierarchical triple: two masses in tight orbit + distant small perturber
    if output_dir is None:
        out_root = Path("outputs/gravity_kepler_demo") / datetime.now().strftime(
            "run_%Y%m%d_%H%M%S"
        )
    else:
        out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # We'll create a custom initial condition by calling run_nbody with small N but modify inside
    # For simplicity call run_nbody with N=3 via a helper here: create three-body with stable binary
    N = 3
    # binary masses
    m1 = 5.0
    m2 = 3.0
    # place at +/- a along x with velocities for circular orbit
    a = 1.0
    vx1 = 0.0
    vy1 = 0.0
    vx2 = 0.0
    vy2 = 0.0
    # compute circular velocities
    G = 1.0
    v_rel = math.sqrt(G * (m1 + m2) / a)
    # set positions
    pts = [
        {
            "id": 0,
            "x": -a * m2 / (m1 + m2),
            "y": 0.0,
            "vx": 0.0,
            "vy": v_rel * m2 / (m1 + m2),
            "m": m1,
        },
        {
            "id": 1,
            "x": a * m1 / (m1 + m2),
            "y": 0.0,
            "vx": 0.0,
            "vy": -v_rel * m1 / (m1 + m2),
            "m": m2,
        },
        {"id": 2, "x": 5.0, "y": 0.5, "vx": 0.0, "vy": -0.05, "m": 0.5},
    ]

    # We'll reuse run_nbody by monkeypatching a tiny wrapper in a controlled run: write custom function here
    steps = 2000
    dt = 0.01
    eps = 0.05
    energy_records = []
    traj_records = []
    for t in range(steps):
        # half kick
        forces, phi_tmp, pot_tmp = compute_forces(pts, G=G, eps=eps)
        for i in range(len(pts)):
            ax = forces[i][0] / pts[i]["m"]
            ay = forces[i][1] / pts[i]["m"]
            pts[i]["vx"] += 0.5 * dt * ax
            pts[i]["vy"] += 0.5 * dt * ay
        # drift
        for p in pts:
            p["x"] += dt * p["vx"]
            p["y"] += dt * p["vy"]
        # full kick
        forces, phi_tmp, pot_tmp = compute_forces(pts, G=G, eps=eps)
        for i in range(len(pts)):
            ax = forces[i][0] / pts[i]["m"]
            ay = forces[i][1] / pts[i]["m"]
            pts[i]["vx"] += 0.5 * dt * ax
            pts[i]["vy"] += 0.5 * dt * ay
        ke = kinetic_energy(pts)
        energy_records.append(
            {"tick": t, "ke": ke, "pe": pot_tmp, "total": ke + pot_tmp}
        )
        if t % 10 == 0:
            for p in pts:
                traj_records.append(
                    {
                        "tick": t,
                        "id": p["id"],
                        "x": p["x"],
                        "y": p["y"],
                        "vx": p["vx"],
                        "vy": p["vy"],
                        "m": p["m"],
                    }
                )

    # write outputs
    with open(out_root / "kepler_energy.csv", "w", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=["tick", "ke", "pe", "total"])
        writer.writeheader()
        for r in energy_records:
            writer.writerow(r)
    with open(out_root / "kepler_traj.csv", "w", newline="") as cf:
        writer = csv.DictWriter(
            cf, fieldnames=["tick", "id", "x", "y", "vx", "vy", "m"]
        )
        writer.writeheader()
        for r in traj_records:
            writer.writerow(r)

    with open(out_root / "kepler_summary.json", "w") as f:
        json.dump({"N": N, "steps": steps, "dt": dt, "eps": eps}, f, indent=2)

    print("Kepler demo outputs in", str(out_root))
    return out_root


def run_sweep(
    eps_list=[0.01, 0.05, 0.1],
    dt_list=[0.02, 0.01, 0.005],
    Ns=[20, 50],
    seeds=None,
    softening_mode="uniform",
    adaptive=False,
):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path("outputs/gravity_sweeps") / f"run_{ts}"
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    if seeds is None:
        seeds = list(range(1, 11))
    for N in Ns:
        for eps in eps_list:
            for dt in dt_list:
                for seed in seeds:
                    od = out_root / f"N_{N}_eps_{eps}_dt_{dt}_s_{seed}"
                    od.mkdir(parents=True, exist_ok=True)
                    # run smaller steps for speed
                    steps = 1000 if N <= 50 else 800
                    run_nbody(
                        n=N,
                        steps=steps,
                        dt=dt,
                        seed=seed,
                        G=1.0,
                        eps=eps,
                        output_dir=od,
                        softening_mode=softening_mode,
                        adaptive=adaptive,
                    )
                    try:
                        with open(od / "gravity_summary.json") as f:
                            summ = json.load(f)
                    except Exception:
                        continue
                    rows.append(
                        {
                            "N": N,
                            "eps": eps,
                            "dt": dt,
                            "seed": seed,
                            "final_total": summ.get("final_total"),
                            "bound_fraction": summ.get("bound_fraction"),
                            "max_energy_drift": summ.get("max_energy_drift"),
                        }
                    )

    # write CSV
    csvp = out_root / "gravity_sweep_results.csv"
    with open(csvp, "w", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=rows[0].keys())
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # aggregate mean bound_fraction and mean energy drift per cell
    summary = {}
    for N in Ns:
        for eps in eps_list:
            for dt in dt_list:
                cell = [
                    r for r in rows if r["N"] == N and r["eps"] == eps and r["dt"] == dt
                ]
                if not cell:
                    continue
                mean_bound = sum(r["bound_fraction"] for r in cell) / len(cell)
                mean_drift = sum(
                    (r.get("max_energy_drift") or 0.0) for r in cell
                ) / len(cell)
                summary[f"N_{N}_eps_{eps}_dt_{dt}"] = {
                    "mean_bound_fraction": mean_bound,
                    "mean_max_energy_drift": mean_drift,
                    "n": len(cell),
                }

    with open(out_root / "gravity_sweep_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # optional plots: bound fraction vs eps and drift vs eps for each N/dt
    if plt is not None:
        for N in Ns:
            for dt in dt_list:
                xs = eps_list
                ys = [
                    summary.get(f"N_{N}_eps_{eps}_dt_{dt}", {}).get(
                        "mean_bound_fraction", 0.0
                    )
                    for eps in xs
                ]
                dr = [
                    summary.get(f"N_{N}_eps_{eps}_dt_{dt}", {}).get(
                        "mean_max_energy_drift", 0.0
                    )
                    for eps in xs
                ]
                plt.figure(figsize=(8, 3))
                plt.subplot(1, 2, 1)
                plt.plot(xs, ys, marker="o")
                plt.xlabel("eps (softening)")
                plt.ylabel("mean bound fraction")
                plt.title(f"Bound vs eps (N={N}, dt={dt})")
                plt.subplot(1, 2, 2)
                plt.plot(xs, dr, marker="o", color="r")
                plt.xlabel("eps (softening)")
                plt.ylabel("mean max energy drift")
                plt.title("Energy drift vs eps")
                plt.tight_layout()
                plt.savefig(out_root / f"bound_and_drift_N{N}_dt{dt}.png")
                plt.close()

        # create a one-page PDF summary combining a few representative plots
        try:
            from matplotlib.backends.backend_pdf import PdfPages

            pdfp = out_root / "gravity_summary.pdf"
            with PdfPages(pdfp) as pdf:
                # first page: for each N, plot bound vs eps for the smallest dt
                for N in Ns:
                    plt.figure(figsize=(6, 4))
                    for dt in dt_list:
                        xs = eps_list
                        ys = [
                            summary.get(f"N_{N}_eps_{eps}_dt_{dt}", {}).get(
                                "mean_bound_fraction", 0.0
                            )
                            for eps in xs
                        ]
                        plt.plot(xs, ys, marker="o", label=f"dt={dt}")
                    plt.xlabel("eps")
                    plt.ylabel("mean bound fraction")
                    plt.title(f"Bound fraction vs eps (N={N})")
                    plt.legend()
                    pdf.savefig()
                    plt.close()

                # second: heatmap-like table of mean drift for first N
                if Ns:
                    N = Ns[0]
                    plt.figure(figsize=(6, 4))
                    for i, eps in enumerate(eps_list):
                        vals = [
                            summary.get(f"N_{N}_eps_{eps}_dt_{dt}", {}).get(
                                "mean_max_energy_drift", 0.0
                            )
                            for dt in dt_list
                        ]
                        plt.plot(dt_list, vals, marker="o", label=f"eps={eps}")
                    plt.xlabel("dt")
                    plt.ylabel("mean max energy drift")
                    plt.title(f"Energy drift vs dt (N={N})")
                    plt.legend()
                    pdf.savefig()
                    plt.close()
        except Exception:
            # if PDF creation fails, ignore but continue
            pass

    print("Gravity sweep complete. Outputs in", str(out_root))
    return out_root


if __name__ == "__main__":
    run_kepler_pair_demo()
    run_sweep()
