import math


def logistic(
    x: float, x0: float, k: float, floor: float = 0.0, cap: float = 1.0
) -> float:
    """Standard logistic with optional floor/cap."""
    y = 1.0 / (1.0 + math.exp(-k * (x - x0)))
    return max(floor, min(cap, y))
