import io
import os

import pandas as pd
import requests

from .utils import log


def fetch_csv_url(url: str) -> pd.DataFrame | None:
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        return pd.read_csv(io.StringIO(r.text))
    except Exception as e:
        log.warning(f"CSV fetch failed ({url}): {e}")
        return None


def fetch_json_url(url: str) -> pd.DataFrame | None:
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        js = r.json()
        if "features" in js and isinstance(js["features"], list):
            rows = [f.get("properties", {}) for f in js["features"]]
            return pd.DataFrame(rows)
        return pd.json_normalize(js)
    except Exception as e:
        log.warning(f"JSON fetch failed ({url}): {e}")
        return None


def fetch_local_csv(path: str) -> pd.DataFrame | None:
    try:
        if os.path.exists(path):
            return pd.read_csv(path)
    except Exception as e:
        log.warning(f"Local CSV load failed ({path}): {e}")
        return None


# synthetic fallbacks
def synth_survival(n=1000):
    import numpy as np
    import pandas as pd

    age = np.clip(np.random.normal(50, 18, n), 0, 100)
    hazard = 0.0005 + 0.00007 * (age**1.2)
    return pd.DataFrame({"age": age, "hazard": hazard})


def synth_shocks(n=500):
    import numpy as np
    import pandas as pd

    mag = np.random.uniform(2.5, 7.5, n)
    depth = np.random.exponential(10, n)
    return pd.DataFrame({"magnitude": mag, "depth_km": depth})


def synth_goals(n=300):
    import numpy as np
    import pandas as pd

    gini = np.random.uniform(0.22, 0.4, n)
    goals = np.random.randint(2, 6, n)
    pop = np.random.choice([100, 300, 500], n, p=[0.4, 0.4, 0.2])
    return pd.DataFrame({"gini": gini, "goal_count": goals, "population": pop})


def synth_calibration(n=1000):
    import numpy as np
    import pandas as pd

    conf = np.clip(np.random.beta(2, 5, n), 0, 1)
    correct = (np.random.rand(n) < (0.35 + 0.5 * conf)).astype(int)
    return pd.DataFrame({"reported_confidence": conf, "correct": correct})


def synth_gravity(n=2000):
    import numpy as np
    import pandas as pd

    mass = np.random.lognormal(0, 1, n)
    ecc = np.clip(np.random.beta(2, 5, n), 0, 1)
    a = np.random.lognormal(1.5, 0.7, n)
    return pd.DataFrame({"mass": mass, "eccentricity": ecc, "a": a})


def load_sources(cfg_section: dict) -> pd.DataFrame:
    for src in cfg_section.get("sources", []):
        if src["type"] == "local_csv":
            df = fetch_local_csv(src["path"])
            if df is not None and len(df):
                return df
        elif src["type"] == "csv_url":
            df = fetch_csv_url(src["url"])
            if df is not None and len(df):
                return df
        elif src["type"] == "json_url":
            df = fetch_json_url(src["url"])
            if df is not None and len(df):
                return df
        elif src["type"] == "synthetic":
            name = src["name"]
            if "survival" in name:
                return synth_survival()
            if "shocks" in name:
                return synth_shocks()
            if "goals" in name:
                return synth_goals()
            if "calibration" in name:
                return synth_calibration()
            if "gravity" in name:
                return synth_gravity()

    return synth_survival()
