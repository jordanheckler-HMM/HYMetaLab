#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

OUT_ROOT = Path("outputs/open_min_budget")
# pick latest timestamp directory
dirs = sorted([d for d in OUT_ROOT.iterdir() if d.is_dir()])
if not dirs:
    print("no outputs found")
    exit(1)
OUT = dirs[-1]
DATA = OUT / "data"
REPORT = OUT / "report"
REPORT.mkdir(parents=True, exist_ok=True)

# load predictions & observations
preds_path = DATA / "agent_predictions.csv"
obs_path = DATA / "agents_observations.csv"
if not preds_path.exists():
    print("no predictions file at", preds_path)
    exit(1)
if not obs_path.exists():
    print("no observations file at", obs_path)
    exit(1)

preds = pd.read_csv(preds_path)
obs = pd.read_csv(obs_path)

# per-agent mean absolute error
mae = (
    preds.groupby("agent_id")["abs_error"]
    .mean()
    .reset_index()
    .rename(columns={"abs_error": "mae"})
)
mae = mae.sort_values("mae")
mae.to_csv(REPORT / "agent_prediction_mae.csv", index=False)

# plot top 6 agents with lowest mae and highest mae sample
low_agents = mae.head(3)["agent_id"].tolist()
high_agents = mae.tail(3)["agent_id"].tolist()
sample_agents = low_agents + high_agents

plt.figure(figsize=(10, 6))
for a in sample_agents:
    sub = obs[obs["agent_id"] == a].groupby("epoch")["obs_CCI"].mean().reset_index()
    plt.plot(sub["epoch"], sub["obs_CCI"], label=f"agent_{a}")
plt.legend()
plt.title("Sample agent obs_CCI traces (low/high MAE)")
plt.xlabel("epoch")
plt.ylabel("obs_CCI")
plt.savefig(REPORT / "sample_agent_traces.png")
plt.close()

# small markdown summary
md = f"Agent predictor summary for outputs: {OUT.name}\n\n"
md += f"MAE summary: mean={mae['mae'].mean():.5f}, median={mae['mae'].median():.5f}\n\n"
md += "Top/Bottom agents (agent_id, mae):\n"
for _, r in mae.head(5).iterrows():
    md += f"- {int(r['agent_id'])}: {r['mae']:.6f}\n"
md += "\n...\n"
for _, r in mae.tail(5).iterrows():
    md += f"- {int(r['agent_id'])}: {r['mae']:.6f}\n"

with open(REPORT / "agent_prediction_report.md", "w") as f:
    f.write(md)

print("analysis written to", REPORT)
