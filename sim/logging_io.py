"""Logging and I/O utilities for the consciousness simulation."""

import csv
import json
import os
from dataclasses import dataclass
from typing import Any


@dataclass
class DecisionLog:
    """Log entry for agent decisions."""

    tick: int
    agent_id: str
    state_hash: str
    workspace_reads: int
    workspace_writes: int
    lookahead_depth: int
    candidates: list[dict[str, Any]]
    chosen_action: str
    reported_conf: float
    outcome_reward: float
    prediction_model_p: float
    rng_seed_local: int


@dataclass
class IntegrationLog:
    """Log entry for workspace integration."""

    tick: int
    agent_id: str
    conflicts: int
    conflict_types: list[str]
    broadcasts: int
    resolution_time_ms: float


@dataclass
class InnovationLog:
    """Log entry for innovations."""

    tick: int
    agent_id: str
    invention_id: str
    parents: list[str]
    novelty_score: float
    utility_score: float
    adopted_by: int
    diffusion_radius: int


@dataclass
class CultureLog:
    """Log entry for cultural transmission."""

    tick: int
    agent_id: str
    meme_id: str
    action: str
    source_id: str
    trust_change: float
    reputation: float
    norm_violations: int


class Logger:
    """Handles all logging for the simulation."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Initialize log files
        self.decisions_file = open(os.path.join(output_dir, "decisions.jsonl"), "w")
        self.integration_file = open(os.path.join(output_dir, "integration.jsonl"), "w")
        self.innovations_file = open(os.path.join(output_dir, "innovations.jsonl"), "w")
        self.culture_file = open(os.path.join(output_dir, "culture.jsonl"), "w")

        # CSV files
        self.lifespans_file = open(os.path.join(output_dir, "lifespans.csv"), "w")
        self.time_series_file = open(os.path.join(output_dir, "time_series.csv"), "w")
        self.predictability_file = open(
            os.path.join(output_dir, "predictability_summary.csv"), "w"
        )

        # Initialize CSV writers
        self._init_csv_writers()

        # Data storage for CSV files
        self.agent_lifespans: dict[str, dict[str, Any]] = {}
        self.time_series_data: list[dict[str, Any]] = []

    def _init_csv_writers(self):
        """Initialize CSV writers with headers."""
        # Lifespans CSV
        self.lifespans_writer = csv.DictWriter(
            self.lifespans_file,
            fieldnames=[
                "agent_id",
                "birth_tick",
                "death_tick",
                "survival_time",
                "mean_conf",
                "conf_brier",
                "ws_reads_total",
                "inventions_made",
                "inventions_adopted",
            ],
        )
        self.lifespans_writer.writeheader()

        # Time series CSV
        self.time_series_writer = csv.DictWriter(
            self.time_series_file,
            fieldnames=[
                "tick",
                "num_agents",
                "avg_energy",
                "avg_consciousness",
                "avg_innovation",
                "avg_reputation",
                "events_this_tick",
            ],
        )
        self.time_series_writer.writeheader()

        # Predictability CSV
        self.predictability_writer = csv.DictWriter(
            self.predictability_file, fieldnames=["model", "accuracy", "logloss"]
        )
        self.predictability_writer.writeheader()

    def log_decision(
        self,
        tick: int,
        agent_id: str,
        state_hash: str,
        workspace_reads: int,
        workspace_writes: int,
        lookahead_depth: int,
        candidates: list[dict[str, Any]],
        chosen_action: str,
        reported_conf: float,
        outcome_reward: float,
        prediction_model_p: float,
        rng_seed_local: int,
    ):
        """Log an agent decision."""
        log_entry = DecisionLog(
            tick=tick,
            agent_id=agent_id,
            state_hash=state_hash,
            workspace_reads=workspace_reads,
            workspace_writes=workspace_writes,
            lookahead_depth=lookahead_depth,
            candidates=candidates,
            chosen_action=chosen_action,
            reported_conf=reported_conf,
            outcome_reward=outcome_reward,
            prediction_model_p=prediction_model_p,
            rng_seed_local=rng_seed_local,
        )

        # Write to JSONL
        json.dump(
            {
                "tick": int(log_entry.tick),
                "agent_id": str(log_entry.agent_id),
                "state_hash": str(log_entry.state_hash),
                "workspace_reads": int(log_entry.workspace_reads),
                "workspace_writes": int(log_entry.workspace_writes),
                "lookahead_depth": int(log_entry.lookahead_depth),
                "candidates": log_entry.candidates,
                "chosen_action": str(log_entry.chosen_action),
                "reported_conf": float(log_entry.reported_conf),
                "outcome_reward": float(log_entry.outcome_reward),
                "prediction_model_p": float(log_entry.prediction_model_p),
                "rng_seed_local": int(log_entry.rng_seed_local),
            },
            self.decisions_file,
        )
        self.decisions_file.write("\n")

    def log_integration(
        self,
        tick: int,
        agent_id: str,
        conflicts: int,
        conflict_types: list[str],
        broadcasts: int,
        resolution_time_ms: float,
    ):
        """Log workspace integration events."""
        log_entry = IntegrationLog(
            tick=tick,
            agent_id=agent_id,
            conflicts=conflicts,
            conflict_types=conflict_types,
            broadcasts=broadcasts,
            resolution_time_ms=resolution_time_ms,
        )

        # Write to JSONL
        json.dump(
            {
                "tick": int(log_entry.tick),
                "agent_id": str(log_entry.agent_id),
                "conflicts": int(log_entry.conflicts),
                "conflict_types": log_entry.conflict_types,
                "broadcasts": int(log_entry.broadcasts),
                "resolution_time_ms": float(log_entry.resolution_time_ms),
            },
            self.integration_file,
        )
        self.integration_file.write("\n")

    def log_innovation(
        self,
        tick: int,
        agent_id: str,
        invention_id: str,
        parents: list[str],
        novelty_score: float,
        utility_score: float,
        adopted_by: int,
        diffusion_radius: int,
    ):
        """Log innovation events."""
        log_entry = InnovationLog(
            tick=tick,
            agent_id=agent_id,
            invention_id=invention_id,
            parents=parents,
            novelty_score=novelty_score,
            utility_score=utility_score,
            adopted_by=adopted_by,
            diffusion_radius=diffusion_radius,
        )

        # Write to JSONL
        json.dump(
            {
                "tick": int(log_entry.tick),
                "agent_id": str(log_entry.agent_id),
                "invention_id": str(log_entry.invention_id),
                "parents": log_entry.parents,
                "novelty_score": float(log_entry.novelty_score),
                "utility_score": float(log_entry.utility_score),
                "adopted_by": int(log_entry.adopted_by),
                "diffusion_radius": int(log_entry.diffusion_radius),
            },
            self.innovations_file,
        )
        self.innovations_file.write("\n")

    def log_culture(
        self,
        tick: int,
        agent_id: str,
        meme_id: str,
        action: str,
        source_id: str,
        trust_change: float,
        reputation: float,
        norm_violations: int,
    ):
        """Log cultural transmission events."""
        log_entry = CultureLog(
            tick=tick,
            agent_id=agent_id,
            meme_id=meme_id,
            action=action,
            source_id=source_id,
            trust_change=trust_change,
            reputation=reputation,
            norm_violations=norm_violations,
        )

        # Write to JSONL
        json.dump(
            {
                "tick": int(log_entry.tick),
                "agent_id": str(log_entry.agent_id),
                "meme_id": str(log_entry.meme_id),
                "action": str(log_entry.action),
                "source_id": str(log_entry.source_id),
                "trust_change": float(log_entry.trust_change),
                "reputation": float(log_entry.reputation),
                "norm_violations": int(log_entry.norm_violations),
            },
            self.culture_file,
        )
        self.culture_file.write("\n")

    def log_agent_birth(self, agent_id: str, birth_tick: int):
        """Log agent birth."""
        self.agent_lifespans[agent_id] = {
            "agent_id": agent_id,
            "birth_tick": birth_tick,
            "death_tick": None,
            "survival_time": None,
            "mean_conf": 0.0,
            "conf_brier": 0.0,
            "ws_reads_total": 0,
            "inventions_made": 0,
            "inventions_adopted": 0,
        }

    def log_agent_death(
        self, agent_id: str, death_tick: int, agent_stats: dict[str, Any]
    ):
        """Log agent death and update lifespan data."""
        if agent_id in self.agent_lifespans:
            lifespan_data = self.agent_lifespans[agent_id]
            lifespan_data["death_tick"] = death_tick
            lifespan_data["survival_time"] = death_tick - lifespan_data["birth_tick"]

            # Update with agent stats
            lifespan_data["mean_conf"] = agent_stats.get("mean_conf", 0.0)
            lifespan_data["ws_reads_total"] = agent_stats.get("ws_reads_total", 0)
            lifespan_data["inventions_made"] = agent_stats.get("inventions_made", 0)
            lifespan_data["inventions_adopted"] = agent_stats.get(
                "inventions_adopted", 0
            )

    def log_time_series(
        self,
        tick: int,
        num_agents: int,
        avg_energy: float,
        avg_consciousness: float,
        avg_innovation: float,
        avg_reputation: float,
        events_this_tick: int,
    ):
        """Log time series data."""
        self.time_series_data.append(
            {
                "tick": tick,
                "num_agents": num_agents,
                "avg_energy": avg_energy,
                "avg_consciousness": avg_consciousness,
                "avg_innovation": avg_innovation,
                "avg_reputation": avg_reputation,
                "events_this_tick": events_this_tick,
            }
        )

        # Write to CSV
        self.time_series_writer.writerow(
            {
                "tick": tick,
                "num_agents": num_agents,
                "avg_energy": avg_energy,
                "avg_consciousness": avg_consciousness,
                "avg_innovation": avg_innovation,
                "avg_reputation": avg_reputation,
                "events_this_tick": events_this_tick,
            }
        )

    def log_predictability(self, model_name: str, accuracy: float, logloss: float):
        """Log predictability analysis results."""
        self.predictability_writer.writerow(
            {"model": model_name, "accuracy": accuracy, "logloss": logloss}
        )

    def finalize(self):
        """Finalize logging and write remaining data."""
        # Write lifespan data
        for lifespan_data in self.agent_lifespans.values():
            self.lifespans_writer.writerow(lifespan_data)

        # Close all files
        self.decisions_file.close()
        self.integration_file.close()
        self.innovations_file.close()
        self.culture_file.close()
        self.lifespans_file.close()
        self.time_series_file.close()
        self.predictability_file.close()

    def get_output_dir(self) -> str:
        """Get the output directory path."""
        return self.output_dir
