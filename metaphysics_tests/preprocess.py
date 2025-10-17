import logging

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


def zscore_df(df, cols):
    out = df.copy()
    for c in cols:
        if c in out.columns:
            v = out[c].astype(float)
            out[c] = (v - v.mean()) / (v.std() if v.std() != 0 else 1.0)
    return out


def slice_epochs(df, epoch_col="epoch"):
    groups = {}
    for e, g in df.groupby(epoch_col):
        groups[int(e)] = g.sort_values("time") if "time" in g.columns else g
    return groups


def epoch_pca_summary(epoch_df, n_components=3, cols=None):
    X = epoch_df[cols].astype(float).values
    pca = PCA(n_components=n_components)
    comps = pca.fit_transform(X)
    return pca, comps


def synthesize_demo(length=6000, seed=12345):
    rng = np.random.RandomState(seed)
    t = np.arange(length)
    # damped Rc cycles
    Rc = 0.5 + 0.4 * np.sin(2 * np.pi * t / 400.0) * np.exp(-t / 4000.0)
    # CCI drives Meaning and Coherence
    CCI = 0.5 + 0.3 * np.sin(2 * np.pi * t / 200.0) + 0.05 * rng.randn(length)
    Meaning = 0.4 + 0.4 * CCI + 0.05 * rng.randn(length)
    Coherence = 0.4 + 0.3 * CCI + 0.03 * rng.randn(length)
    epsilon = 0.01 * np.ones(length)
    # periodic shocks
    for s in range(1000, length, 3000):
        Rc[s : s + 5] -= 0.6
        epsilon[s : s + 3] += 1.0
    df = pd.DataFrame(
        {
            "time": t,
            "CCI": CCI,
            "Meaning": Meaning,
            "Coherence": Coherence,
            "Rc": Rc,
            "epsilon": epsilon,
        }
    )
    # split into epochs at shocks
    df["epoch"] = 0
    spikes = np.where(epsilon > 0.5)[0]
    for i, idx in enumerate(np.unique(np.floor_divide(spikes, 3000))):
        pass
    return df
