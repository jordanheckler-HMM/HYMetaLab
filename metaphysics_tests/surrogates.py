import numpy as np


def phase_randomized(x, rng=None):
    rng = rng or np.random.RandomState()
    X = np.asarray(x, dtype=float)
    n = len(X)
    fft = np.fft.rfft(X)
    mag = np.abs(fft)
    phase = np.angle(fft)
    # randomize phases except DC and Nyquist
    rnd = rng.uniform(0, 2 * np.pi, size=phase.shape)
    rnd[0] = phase[0]
    new = mag * np.exp(1j * rnd)
    ts = np.fft.irfft(new, n=n)
    # match original mean & std
    ts = (ts - ts.mean()) / (ts.std() if ts.std() != 0 else 1.0)
    ts = ts * (np.std(X) if np.std(X) != 0 else 1.0) + np.mean(X)
    return ts


def shuffle_surrogate(x, rng=None):
    rng = rng or np.random.RandomState()
    y = np.copy(x)
    rng.shuffle(y)
    return y


def multichannel_phase_surrogates(X, n=100, rng=None):
    rng = rng or np.random.RandomState()
    N, T = X.shape[0], X.shape[1]
    out = []
    for i in range(n):
        Y = np.zeros_like(X)
        for c in range(X.shape[1]):
            Y[:, c] = phase_randomized(X[:, c], rng)
        out.append(Y)
    return out
