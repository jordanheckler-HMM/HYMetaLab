#!/usr/bin/env python3
"""
Lightweight simulation core stubs for time metaphysics experiments.
These are intentionally simple, deterministic-enough functions to produce
plausible outputs for downstream analysis and visualization.
"""
import numpy as np


class calibration_experiment:
    @staticmethod
    def run(agents=50, noise=0.05, temporal_mode="linear"):
        """Return an array of per-agent CCI values (0..1).
        Temporal mode slightly shifts baseline coherence.
        """
        base = 0.7
        if temporal_mode == "cyclic":
            base += 0.05
        elif temporal_mode == "branching":
            base -= 0.05
        rng = np.random.RandomState(
            int((len(str(agents)) + int(noise * 1000)) * 1000) % 2**32
        )
        cci = rng.normal(loc=base, scale=noise, size=agents)
        cci = np.clip(cci, 0.01, 1.0)
        return cci


class meaning_experiment:
    @staticmethod
    def run(cci, shock=0.2, temporal_mode="linear"):
        """Return an array of 'meaning' scores correlated with CCI and shock."""
        noise = 0.05
        mod = 1.0
        if temporal_mode == "cyclic":
            mod = 1.05
        elif temporal_mode == "branching":
            mod = 0.95
        rng = np.random.RandomState(int(len(cci) + int(shock * 100)))
        meaning = cci * (1.0 - shock * 0.3) * mod + rng.normal(0, noise, size=len(cci))
        return np.clip(meaning, 0.0, 1.0)


class gravity_analysis:
    @staticmethod
    def run(mode="linear", agent_count=50):
        """Return per-agent gravity/coherence values; higher when temporal modes favor coherence."""
        base = 0.6
        if mode == "cyclic":
            base += 0.1
        elif mode == "branching":
            base -= 0.1
        rng = np.random.RandomState(agent_count * 13)
        arr = rng.normal(loc=base, scale=0.05, size=agent_count)
        return np.clip(arr, 0.0, 1.0)


class shock_resilience:
    @staticmethod
    def run(cci, shock=0.2, time_mode="linear"):
        """Return per-agent resilience scores; lower with larger shocks and with branching time."""
        base = 1.0 - shock * 0.5
        if time_mode == "cyclic":
            base += 0.1
        elif time_mode == "branching":
            base -= 0.1
        rng = np.random.RandomState(int(len(cci) + int(shock * 100)))
        res = base * cci + rng.normal(0, 0.05, size=len(cci))
        return np.clip(res, 0.0, 1.0)
