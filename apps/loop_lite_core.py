import math
import random


def _sigmoid(x):
    return 1 / (1 + math.exp(-x))


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def compute_once(trust, hope, meaning, sensitivity=1.6, coupling=0.35, noise=0.00):
    """
    Inputs in [0,1]. More responsive nonlinearity:
    - amplify good regions, penalize low-low combos via coupling
    """
    t = trust
    h = hope
    m = meaning
    # Nonlinear uplift
    uplift = (
        0.45 * _sigmoid(sensitivity * (t - 0.5))
        + 0.35 * _sigmoid(sensitivity * (h - 0.5))
        + 0.20 * _sigmoid(sensitivity * (m - 0.5))
    )
    # Coupling penalty when trust & hope are BOTH low
    penalty = coupling * (1 - t) * (1 - h)

    dcci = clamp(uplift - penalty + noise * (random.uniform(-0.02, 0.02)))
    # Risk reduction increases as uplift rises; steeper with high trust
    # FIXED: Reversed calculation so high trust/hope → high risk reduction (positive)
    risk_reduction_raw = clamp((0.7 * dcci + 0.2 * t + 0.1 * h) - 0.5, 0.0, 1.0)
    # Convert to signed form: higher is better for both metrics
    dcci_signed = dcci  # 0→1 good
    risk_reduction_signed = risk_reduction_raw * 0.12  # 0→0.12 good (positive)
    return dcci_signed, risk_reduction_signed


def trajectory(trust, hope, meaning, steps=30, alpha=0.25, **kw):
    """
    Simple relaxation: state moves toward current compute_once output.
    """
    c, z = 0.02, -0.02  # start near neutral
    xs, ys = [], []
    for _ in range(steps):
        target_c, target_z = compute_once(trust, hope, meaning, **kw)
        c += alpha * (target_c - c)
        z += alpha * (target_z - z)
        xs.append(c)
        ys.append(z)
    return xs, ys
