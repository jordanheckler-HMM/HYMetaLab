import numpy as np
import pandas as pd


def phase_randomize(x, rng=None):
    # x: 1D real-valued
    n = len(x)
    Xf = np.fft.rfft(x)
    rng = np.random.default_rng() if rng is None else rng
    phases = np.exp(1j * rng.uniform(0, 2 * np.pi, size=Xf.shape))
    # preserve DC and Nyquist
    phases[0] = 1
    if Xf.size > 1:
        phases[-1] = 1
    Xf_s = Xf * phases
    xs = np.fft.irfft(Xf_s, n=n)
    # rescale to original mean/std
    xs = (xs - xs.mean()) / (xs.std() + 1e-12)
    return xs


def generate_surrogates(df, n=30, seed=42):
    rng = np.random.default_rng(seed)
    cols = ["CCI", "Meaning", "Coherence", "Rc", "epsilon"]
    X = df[cols].to_numpy(dtype=float)
    surrogates = []
    for i in range(n):
        Xs = np.zeros_like(X)
        for c in range(X.shape[1]):
            Xs[:, c] = phase_randomize(X[:, c], rng=rng)
        surrogates.append(Xs)
    return surrogates


def surrogate_rqa_compare(df, rqa_func, n=30, seed=42):
    # rqa_func expects df-like input; we will pass DataFrame made from surrogate
    sur = generate_surrogates(df, n=n, seed=seed)
    metrics = []
    for s in sur:
        sdf = df.copy()
        sdf[["CCI", "Meaning", "Coherence", "Rc", "epsilon"]] = s
        m, _, _ = rqa_func(sdf)
        metrics.append(m)
    metrics_df = pd.DataFrame(metrics)
    return metrics_df
