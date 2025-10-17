"""Global workspace implementation for consciousness simulation."""

import time
from dataclasses import dataclass
from typing import Any


@dataclass
class WorkspaceConflict:
    """Represents a conflict in the global workspace."""

    conflict_type: str
    agent_id: str
    timestamp: float
    resolution_time_ms: float


class Workspace:
    """Global workspace for information integration and broadcasting."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._data: dict[str, Any] = {}
        self.reads = 0
        self.writes = 0
        self.conflicts: list[WorkspaceConflict] = []
        self.broadcasts = 0
        self._write_locks: dict[str, str] = {}  # key -> agent_id

    def read(self, key: str | None = None) -> Any:
        """Read from workspace, incrementing read counter."""
        if not self.enabled:
            return None

        self.reads += 1
        if key is None:
            return self._data.copy()
        return self._data.get(key)

    def write(self, key: str, value: Any, agent_id: str) -> bool:
        """Write to workspace, handling conflicts and incrementing write counter."""
        if not self.enabled:
            return True

        start_time = time.time()
        self.writes += 1

        # Check for conflicts
        if key in self._write_locks and self._write_locks[key] != agent_id:
            # Conflict detected
            resolution_time = (time.time() - start_time) * 1000  # Convert to ms
            conflict = WorkspaceConflict(
                conflict_type="write_conflict",
                agent_id=agent_id,
                timestamp=time.time(),
                resolution_time_ms=resolution_time,
            )
            self.conflicts.append(conflict)

            # For now, allow overwrite (last writer wins)
            # In a more sophisticated system, this could implement conflict resolution

        self._write_locks[key] = agent_id
        self._data[key] = value
        return True

    def broadcast(self, message: dict[str, Any], agent_id: str) -> None:
        """Broadcast a message to all agents via workspace."""
        if not self.enabled:
            return

        self.broadcasts += 1
        # Store broadcast in workspace with timestamp
        broadcast_key = f"broadcast_{agent_id}_{time.time()}"
        self._data[broadcast_key] = {
            "message": message,
            "sender": agent_id,
            "timestamp": time.time(),
        }

    def get_conflicts(self) -> list[WorkspaceConflict]:
        """Get list of conflicts that occurred."""
        return self.conflicts.copy()

    def get_stats(self) -> dict[str, int]:
        """Get workspace statistics."""
        return {
            "reads": self.reads,
            "writes": self.writes,
            "broadcasts": self.broadcasts,
            "conflicts": len(self.conflicts),
            "data_size": len(self._data),
        }

    def clear(self) -> None:
        """Clear workspace data and reset counters."""
        self._data.clear()
        self._write_locks.clear()
        self.conflicts.clear()
        self.reads = 0
        self.writes = 0
        self.broadcasts = 0
