import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def normalize_cols(df):
    # map common names to canonical lowercase names
    mapping = {}
    for c in df.columns:
        key = re.sub(r"[^a-z0-9]", "", c.lower())
        mapping[c] = key
    df = df.rename(columns=mapping)
    return df


def load_csvs(glob_pattern):
    p = Path(".").glob(glob_pattern) if "*" in glob_pattern else [Path(glob_pattern)]
    frames = []
    for f in p:
        try:
            df = pd.read_csv(f)
            df = normalize_cols(df)
            df["source_file"] = str(f)
            frames.append(df)
        except Exception as e:
            logger.exception("Failed to read %s: %s", f, e)
    if not frames:
        return None
    out = pd.concat(frames, ignore_index=True, sort=False)
    return out


def infer_epochs(df):
    # If 'epoch' present, use it. Otherwise, try to detect shock markers via 'epsilon' spikes or 'time' resets.
    if "epoch" in df.columns:
        return df
    # simple heuristic: if 'epsilon' exists and has spikes, segment by large jumps
    if "epsilon" in df.columns:
        eps = df["epsilon"].fillna(0).values
        spikes = np.where(np.abs(np.diff(eps)) > 0.5)[0]
        if len(spikes) > 0:
            # split into epochs
            epoch = 0
            epochs = []
            prev = 0
            for s in spikes:
                epochs.extend([epoch] * (s - prev + 1))
                epoch += 1
                prev = s + 1
            epochs.extend([epoch] * (len(eps) - prev))
            df["epoch"] = epochs
            return df
    # fallback: single epoch 0
    df["epoch"] = 0
    return df
