"""
Core spacetime abstractions for retrocausality simulations.

Provides event structures, retrocausal links, and spacetime graph
representations that work with existing simulation modules.
"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Event:
    """Represents a spacetime event with state and metadata."""

    id: str
    t: int  # temporal coordinate
    state: dict[str, Any]  # must pass through to existing sims unchanged
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Generate consistency hash for the event."""
        self._hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute consistency hash for self-consistency checking."""
        content = {"id": self.id, "t": self.t, "state": self.state, "meta": self.meta}
        return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()[
            :16
        ]

    def apply_update(self, delta_state: dict[str, Any]) -> "Event":
        """Return new event with updated state and new hash."""
        new_state = {**self.state, **delta_state}
        return Event(id=self.id, t=self.t, state=new_state, meta=self.meta)


@dataclass
class RetroLink:
    """Represents a retrocausal connection from future to past."""

    t_future: int
    t_past: int
    payload: dict[str, Any]  # what "comes back" in time
    model: str  # "novikov" | "deutsch"
    bandwidth_bits: int = 0  # information capacity constraint

    def __post_init__(self):
        """Validate retrocausal link."""
        if self.t_future <= self.t_past:
            raise ValueError("RetroLink must go from future to past")
        if self.model not in ["novikov", "deutsch"]:
            raise ValueError("Model must be 'novikov' or 'deutsch'")


@dataclass
class Worldline:
    """Represents the timeline of a single agent."""

    agent_id: str
    events: list[Event] = field(default_factory=list)

    def add_event(self, event: Event) -> None:
        """Add event to worldline, maintaining temporal order."""
        self.events.append(event)
        self.events.sort(key=lambda e: e.t)

    def get_event_at(self, t: int) -> Event | None:
        """Get event at specific time, or None if not found."""
        for event in self.events:
            if event.t == t:
                return event
        return None


class SpacetimeGraph:
    """
    Spacetime graph supporting both forward causality and retrocausal links.

    By default, this is a DAG (forward-time only). RetroLinks are special
    edges that point from future events to past events (t_j > t_i).
    """

    def __init__(self):
        self.forward_edges: dict[str, set[str]] = {}  # event_id -> {future_event_ids}
        self.retro_links: list[RetroLink] = []
        self.events: dict[str, Event] = {}
        self.worldlines: dict[str, Worldline] = {}

    def add_event(self, event: Event) -> None:
        """Add event to spacetime graph."""
        self.events[event.id] = event

        # Add to appropriate worldline
        agent_id = event.state.get("agent_id", "default")
        if agent_id not in self.worldlines:
            self.worldlines[agent_id] = Worldline(agent_id)
        self.worldlines[agent_id].add_event(event)

    def add_retro_link(self, retro_link: RetroLink) -> None:
        """Add retrocausal link to the graph."""
        self.retro_links.append(retro_link)

    def get_retro_influences(self, t: int) -> list[RetroLink]:
        """Get all retrocausal influences affecting time t."""
        return [link for link in self.retro_links if link.t_past == t]

    def check_causality_violations(self) -> list[str]:
        """Check for causality violations in the graph."""
        violations = []

        for link in self.retro_links:
            future_event = self.events.get(f"t_{link.t_future}")
            past_event = self.events.get(f"t_{link.t_past}")

            if future_event and past_event:
                # Check if retro influence creates logical contradiction
                if future_event._hash == past_event._hash:
                    violations.append(f"Circular causality at t={link.t_past}")

        return violations

    def get_light_cone(self, event_id: str) -> set[str]:
        """Get all events in the future light cone of an event."""
        visited = set()
        queue = [event_id]

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            # Add forward neighbors
            for neighbor in self.forward_edges.get(current, []):
                if neighbor not in visited:
                    queue.append(neighbor)

        return visited - {event_id}  # Exclude the source event
