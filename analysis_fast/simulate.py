import numpy as np
import pandas as pd


def synthesize_run(steps=24000, epochs=5, shock_every=4800, seed=42):
    rng = np.random.default_rng(seed)
    t = np.arange(steps)

    # Rc: damped cosine with low-frequency modulation + noise
    freq = 0.005
    decay = 0.00002
    rc_base = np.cos(2 * np.pi * freq * t) * np.exp(-decay * t)
    rc_noise = rng.normal(0, 0.05, size=steps)
    Rc = rc_base + rc_noise

    # epsilon: baseline with small drift and spike at shocks
    eps_base = 0.2 + 0.00001 * t
    epsilon = eps_base + rng.normal(0, 0.01, size=steps)

    # CCI: AR(1) process coupled weakly to Rc
    ar_phi = 0.995
    cci = np.empty(steps)
    cci[0] = rng.normal(0, 0.5)
    for i in range(1, steps):
        cci[i] = ar_phi * cci[i - 1] + 0.001 * Rc[i - 1] + rng.normal(0, 0.02)

    # Meaning and Coherence driven by CCI and Rc
    Meaning = 0.3 * cci + 0.2 * Rc + rng.normal(0, 0.03, size=steps)
    Coherence = 0.4 * cci + 0.1 * Rc + rng.normal(0, 0.03, size=steps)

    # Insert shocks at epoch boundaries
    shock_inds = [(i + 1) * shock_every for i in range(epochs - 1)]
    for si in shock_inds:
        if si < steps:
            Rc[si : si + 10] -= 0.6 * np.exp(-0.5 * np.arange(min(10, steps - si)))
            epsilon[si : si + 5] += 0.5

    df = pd.DataFrame(
        {
            "time": t,
            "CCI": cci,
            "Meaning": Meaning,
            "Coherence": Coherence,
            "Rc": Rc,
            "epsilon": epsilon,
        }
    )

    # Assign epochs
    epoch_edges = [0] + shock_inds + [steps]
    epochs_list = []
    for i in range(len(epoch_edges) - 1):
        start = epoch_edges[i]
        end = epoch_edges[i + 1]
        epochs_list.append((start, end))
        df.loc[start : end - 1, "epoch"] = i

    df["epoch"] = df["epoch"].astype(int)
    # Reorder columns
    df = df[["time", "epoch", "CCI", "Meaning", "Coherence", "Rc", "epsilon"]]
    return df
