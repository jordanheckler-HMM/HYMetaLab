import numpy as np


def survival_rate_from_any(x):
    if isinstance(x, (list, tuple)):
        x = np.array(x)
    if hasattr(x, "dtype"):  # array-like
        return float(np.mean(x))
    x = float(x)
    if x < 0 or x > 1:
        raise ValueError("survival_rate must be in [0,1]")
    return x


def cci(calibration=None, coherence=None, emergence=None, noise=None, eps=1e-9):
    if any(v is None for v in [calibration, coherence, emergence, noise]):
        return None
    raw = (calibration * coherence * emergence) / max(noise, eps)
    return float(max(0.0, min(1.0, raw)))


def align_index_proxy(coord, ineq):
    # 0..1 score: higher with coordination, lower with goal inequality
    coord = float(np.clip(coord, 0, 1))
    ineq = float(np.clip(ineq, 0, 1))
    return float(np.clip(0.5 * coord + 0.5 * (1.0 - ineq), 0, 1))


def bootstrap_ci(arr, B=300, ci=0.95, rng=None, stat_fn=np.mean):
    rng = np.random.default_rng() if rng is None else rng
    n = len(arr)
    reps = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        reps.append(stat_fn(arr[idx]))
    reps = np.sort(np.array(reps))
    lo = (1 - ci) / 2
    hi = 1 - lo
    return float(np.quantile(reps, lo)), float(np.quantile(reps, hi))
