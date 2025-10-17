from pathlib import Path

import matplotlib.pyplot as plt


def save_example_timeseries(df, path, length=2000, dpi=110):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 3), dpi=dpi)
    cols = ["CCI", "Rc"]
    subset = df.iloc[: min(length, len(df))]
    for c in cols:
        plt.plot(subset["time"], subset[c], label=c, linewidth=0.8)
    plt.legend()
    plt.xlabel("time")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
