"""Enhanced Newtonian N-body gravity demo (2D) with diagnostics.

Features added:
- per-particle potential (phi) so we can compute specific energy per particle
- hierarchical softening option: eps scales with (m_i*m_j)^(1/3)
- adaptive timestep (simple heuristic based on max acceleration)
- escape detection based on radius relative to COM
- energy drift tracking (max relative drift)
- returns detailed records when requested

Writes the same outputs as before and closes matplotlib figures when used.
"""

import csv
import json
import math
import random
import sys
from datetime import datetime
from pathlib import Path

# allow running from repo root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.io_utils import write_run_manifest

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def compute_forces(pts, G=1.0, eps=0.1, softening_mode="uniform"):
    """Compute pairwise forces and per-particle potential (phi: potential per unit mass).

    softening_mode: 'uniform' uses eps^2 added to r^2; 'hierarchical' scales eps by (m_i*m_j)**(1/3)
    Returns: (forces, phi, pot_total)
    - forces: [[fx,fy],...]
    - phi: per-particle potential per unit mass (so particle i has potential energy m_i * phi[i])
    - pot_total: total potential energy (sum over pairs, negative)
    """
    n = len(pts)
    forces = [[0.0, 0.0] for _ in range(n)]
    phi = [0.0 for _ in range(n)]
    pot_total = 0.0
    for i in range(n):
        xi, yi = pts[i]["x"], pts[i]["y"]
        mi = pts[i]["m"]
        for j in range(i + 1, n):
            xj, yj = pts[j]["x"], pts[j]["y"]
            mj = pts[j]["m"]
            dx = xj - xi
            dy = yj - yi
            if softening_mode == "hierarchical":
                eps_ij = eps * ((mi * mj) ** (1.0 / 3.0))
            else:
                eps_ij = eps
            r2 = dx * dx + dy * dy + eps_ij * eps_ij
            r = math.sqrt(r2)
            if r == 0:
                # identical positions; skip to avoid div-by-zero
                continue
            f = G * mi * mj / (r2)
            fx = f * dx / r
            fy = f * dy / r
            forces[i][0] += fx
            forces[i][1] += fy
            forces[j][0] -= fx
            forces[j][1] -= fy
            # potential per unit mass contributions
            phi[i] += -G * mj / r
            phi[j] += -G * mi / r
            pot_total -= G * mi * mj / r
    return forces, phi, pot_total


def leapfrog_step(pts, dt, G=1.0, eps=0.1, softening_mode="uniform"):
    """Perform a single leapfrog step and return (pot, phi) for the configuration after the step."""
    n = len(pts)
    # half kick
    forces, phi, pot = compute_forces(pts, G=G, eps=eps, softening_mode=softening_mode)
    for i in range(n):
        ax = forces[i][0] / pts[i]["m"]
        ay = forces[i][1] / pts[i]["m"]
        pts[i]["vx"] += 0.5 * dt * ax
        pts[i]["vy"] += 0.5 * dt * ay
    # drift
    for i in range(n):
        pts[i]["x"] += dt * pts[i]["vx"]
        pts[i]["y"] += dt * pts[i]["vy"]
    # full kick
    forces, phi, pot = compute_forces(pts, G=G, eps=eps, softening_mode=softening_mode)
    for i in range(n):
        ax = forces[i][0] / pts[i]["m"]
        ay = forces[i][1] / pts[i]["m"]
        pts[i]["vx"] += 0.5 * dt * ax
        pts[i]["vy"] += 0.5 * dt * ay
    return pot, phi


def kinetic_energy(pts):
    ke = 0.0
    for p in pts:
        ke += 0.5 * p["m"] * (p["vx"] ** 2 + p["vy"] ** 2)
    return ke


def run_nbody(
    n=50,
    steps=2000,
    dt=0.01,
    seed=42,
    G=1.0,
    eps=0.1,
    output_dir=None,
    sample_stride=20,
    return_data=False,
    softening_mode="uniform",
    adaptive=False,
    dt_min=1e-4,
    dt_max=0.05,
    eta=0.05,
):
    random.seed(seed)
    if output_dir is None:
        out_root = Path("outputs/gravity_nbody") / datetime.now().strftime(
            "run_%Y%m%d_%H%M%S"
        )
    else:
        out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    pts = []
    # initialize points in a disk with small random velocities
    for i in range(n):
        r = random.random() ** 0.5 * 5.0
        theta = random.random() * 2 * math.pi
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        vx = -0.2 * y + 0.1 * (random.random() - 0.5)
        vy = 0.2 * x + 0.1 * (random.random() - 0.5)
        m = 1.0 if random.random() < 0.9 else 5.0
        pts.append({"id": i, "x": x, "y": y, "vx": vx, "vy": vy, "m": m})

    # initial potential for baseline
    _, phi0, pot0 = compute_forces(pts, G=G, eps=eps, softening_mode=softening_mode)
    ke0 = kinetic_energy(pts)
    total0 = ke0 + pot0
    max_drift = 0.0
    # initial radius for escape detection (scale with initial cluster size)
    init_r = max(math.sqrt(p["x"] * p["x"] + p["y"] * p["y"]) for p in pts)
    R_escape = max(10.0, 5.0 * init_r)

    energy_records = []
    traj_records = []
    com_records = []

    t = 0
    current_dt = dt
    while t < steps:
        # adaptive dt based on max acceleration if requested
        if adaptive:
            forces_tmp, phi_tmp, pot_tmp = compute_forces(
                pts, G=G, eps=eps, softening_mode=softening_mode
            )
            a_max = 0.0
            for i in range(len(pts)):
                ax = forces_tmp[i][0] / pts[i]["m"]
                ay = forces_tmp[i][1] / pts[i]["m"]
                a = math.sqrt(ax * ax + ay * ay)
                if a > a_max:
                    a_max = a
            if a_max > 0:
                current_dt = min(dt_max, max(dt_min, eta / math.sqrt(a_max + 1e-12)))
        else:
            current_dt = dt

        pot, phi = leapfrog_step(
            pts, current_dt, G=G, eps=eps, softening_mode=softening_mode
        )
        ke = kinetic_energy(pts)
        total = ke + pot
        # record energy and drift
        energy_records.append(
            {"tick": t, "ke": ke, "pe": pot, "total": total, "dt": current_dt}
        )
        if abs(total0) > 0:
            drift = abs((total - total0) / total0)
            if drift > max_drift:
                max_drift = drift

        # sample trajectories and COM
        if t % sample_stride == 0:
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
            mx = sum(p["m"] * p["x"] for p in pts)
            my = sum(p["m"] * p["y"] for p in pts)
            mm = sum(p["m"] for p in pts)
            mvx = sum(p["m"] * p["vx"] for p in pts)
            mvy = sum(p["m"] * p["vy"] for p in pts)
            comx = mx / mm
            comy = my / mm
            comvx = mvx / mm
            comvy = mvy / mm
            com_records.append(
                {"tick": t, "comx": comx, "comy": comy, "comvx": comvx, "comvy": comvy}
            )

        t += 1

    summary = {
        "n": n,
        "steps": steps,
        "dt": dt,
        "G": G,
        "eps": eps,
        "softening_mode": softening_mode,
        "seed": seed,
        "initial_ke": ke0,
        "initial_pe": pot0,
        "initial_total": total0,
        "final_ke": energy_records[-1]["ke"] if energy_records else None,
        "final_pe": energy_records[-1]["pe"] if energy_records else None,
        "final_total": energy_records[-1]["total"] if energy_records else None,
        "max_energy_drift": max_drift,
    }

    # determine bound fraction using specific energy per unit mass in barycentric frame
    total_m = sum(p["m"] for p in pts)
    com_vx = sum(p["m"] * p["vx"] for p in pts) / total_m
    com_vy = sum(p["m"] * p["vy"] for p in pts) / total_m
    bound = 0
    escape_count = 0
    # compute phi (potential per unit mass) for final config
    _, phi_final, _ = compute_forces(pts, G=G, eps=eps, softening_mode=softening_mode)
    for i, p in enumerate(pts):
        vrel2 = (p["vx"] - com_vx) ** 2 + (p["vy"] - com_vy) ** 2
        ke_spec = 0.5 * vrel2
        phi_i = phi_final[i]
        e_spec = ke_spec + phi_i
        if e_spec < 0:
            bound += 1
        # escape detection relative to last COM sample
        last_com = com_records[-1] if com_records else {"comx": 0.0, "comy": 0.0}
        r = math.sqrt(
            (p["x"] - last_com["comx"]) ** 2 + (p["y"] - last_com["comy"]) ** 2
        )
        if r > R_escape:
            escape_count += 1
    bound_frac = bound / len(pts)
    summary["bound_fraction"] = bound_frac
    summary["escape_count"] = escape_count

    # write energy CSV
    with open(out_root / "energy_timeseries.csv", "w", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=["tick", "ke", "pe", "total", "dt"])
        writer.writeheader()
        for r in energy_records:
            writer.writerow(r)

    # write trajectories CSV (may be large)
    with open(out_root / "trajectories_sampled.csv", "w", newline="") as cf:
        writer = csv.DictWriter(
            cf, fieldnames=["tick", "id", "x", "y", "vx", "vy", "m"]
        )
        writer.writeheader()
        for r in traj_records:
            writer.writerow(r)

    # write summary
    with open(out_root / "gravity_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    write_run_manifest(
        str(out_root),
        {
            "experiment": "gravity_nbody",
            "n": n,
            "steps": steps,
            "dt": dt,
            "G": G,
            "eps": eps,
            "softening_mode": softening_mode,
        },
        seed,
    )

    # plotting
    if plt is not None:
        try:
            ticks = [r["tick"] for r in energy_records]
            ke = [r["ke"] for r in energy_records]
            pe = [r["pe"] for r in energy_records]
            tot = [r["total"] for r in energy_records]
            plt.figure()
            plt.plot(ticks, ke, label="KE")
            plt.plot(ticks, pe, label="PE")
            plt.plot(ticks, tot, label="Total")
            plt.xlabel("tick")
            plt.ylabel("energy")
            plt.legend()
            plt.title("Energy vs time")
            plt.savefig(out_root / "energy_timeseries.png")
            plt.close()

            xs = [p["x"] for p in pts]
            ys = [p["y"] for p in pts]
            ms = [p["m"] for p in pts]
            plt.figure()
            plt.scatter(xs, ys, s=[(1 + m) * 10 for m in ms], c=ms, cmap="viridis")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title("Final positions (size~mass)")
            plt.savefig(out_root / "final_positions.png")
            plt.close()
        except Exception:
            pass

    print("N-body run complete. Outputs in", str(out_root))
    if return_data:
        return out_root, pts, energy_records, traj_records, com_records
    return out_root


if __name__ == "__main__":
    # quick smoke run
    run_nbody(n=30, steps=200, dt=0.02, seed=1)
