import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
# sentinelml/agents/state/checkpoint.py
"""
State checkpointing and recovery for agents.
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from sentinelml.core.base import BaseSentinelComponent


@dataclass
class Checkpoint:
    """Agent state checkpoint."""

    step_number: int
    state: Dict[str, Any]
    trajectory: List[Dict[str, Any]]
    timestamp: float
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_number": self.step_number,
            "state": self.state,
            "trajectory": self.trajectory,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Checkpoint":
        return cls(
            step_number=data["step_number"],
            state=data["state"],
            trajectory=data["trajectory"],
            timestamp=data["timestamp"],
            metadata=data["metadata"],
        )


class CheckpointManager(BaseSentinelComponent):
    """
    Manage agent state checkpoints for recovery.

    Saves and restores agent state at key points,
    enabling recovery from failures.

    Parameters
    ----------
    checkpoint_dir : str
        Directory to save checkpoints.
    checkpoint_interval : int, default=10
        Save checkpoint every N steps.
    max_checkpoints : int, default=5
        Maximum checkpoints to retain.

    Examples
    --------
    >>> manager = CheckpointManager(checkpoint_dir='./checkpoints')
    >>> for i, step in enumerate(trajectory):
    ...     if manager.should_checkpoint(i):
    ...         manager.save_checkpoint(i, agent_state, trajectory_so_far)
    ...     # ... execute step ...
    >>> # On failure, restore:
    >>> checkpoint = manager.load_latest_checkpoint()
    """

    def __init__(
        self,
        checkpoint_dir: str = "./checkpoints",
        name: str = "CheckpointManager",
        checkpoint_interval: int = 10,
        max_checkpoints: int = 5,
        verbose: bool = False,
    ):
        super().__init__(name=name, verbose=verbose)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_interval = checkpoint_interval
        self.max_checkpoints = max_checkpoints

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints: List[Checkpoint] = []

    def fit(self, X=None, y=None):
        """No fitting required."""
        self.is_fitted_ = True
        return self

    def should_checkpoint(self, step_number: int) -> bool:
        """Determine if checkpoint should be saved at this step."""
        return step_number % self.checkpoint_interval == 0 and step_number > 0

    def save_checkpoint(
        self,
        step_number: int,
        state: Dict[str, Any],
        trajectory: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Save a checkpoint.

        Returns
        -------
        path : str
            Path to saved checkpoint.
        """
        checkpoint = Checkpoint(
            step_number=step_number,
            state=state.copy(),
            trajectory=trajectory.copy(),
            timestamp=time.time(),
            metadata=metadata or {},
        )

        # Add to memory
        self.checkpoints.append(checkpoint)

        # Trim old checkpoints
        if len(self.checkpoints) > self.max_checkpoints:
            self.checkpoints = self.checkpoints[-self.max_checkpoints :]

        # Save to disk
        filename = f"checkpoint_{step_number}_{int(checkpoint.timestamp)}.json"
        filepath = self.checkpoint_dir / filename

        with open(filepath, "w") as f:
            json.dump(checkpoint.to_dict(), f, indent=2, default=str)

        if self.verbose:
            print(f"Checkpoint saved: {filepath}")

        # Clean old files
        self._cleanup_old_files()

        return str(filepath)

    def load_checkpoint(self, step_number: Optional[int] = None) -> Optional[Checkpoint]:
        """
        Load a checkpoint.

        Parameters
        ----------
        step_number : int, optional
            Specific step to load. If None, loads latest.

        Returns
        -------
        Checkpoint or None
        """
        if step_number is not None:
            # Find specific checkpoint
            for cp in reversed(self.checkpoints):
                if cp.step_number == step_number:
                    return cp

            # Try to load from disk
            return self._load_from_disk(step_number)

        # Return latest from memory
        if self.checkpoints:
            return self.checkpoints[-1]

        # Try to load latest from disk
        return self._load_latest_from_disk()

    def _load_from_disk(self, step_number: int) -> Optional[Checkpoint]:
        """Load specific checkpoint from disk."""
        pattern = f"checkpoint_{step_number}_*.json"
        matches = list(self.checkpoint_dir.glob(pattern))

        if not matches:
            return None

        # Load most recent
        latest = max(matches, key=lambda p: p.stat().st_mtime)

        with open(latest, "r") as f:
            data = json.load(f)

        return Checkpoint.from_dict(data)

    def _load_latest_from_disk(self) -> Optional[Checkpoint]:
        """Load latest checkpoint from disk."""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.json"))

        if not checkpoints:
            return None

        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)

        with open(latest, "r") as f:
            data = json.load(f)

        return Checkpoint.from_dict(data)

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints."""
        checkpoints = []

        for cp in self.checkpoints:
            checkpoints.append(
                {"step_number": cp.step_number, "timestamp": cp.timestamp, "in_memory": True}
            )

        # Also list disk checkpoints
        for filepath in self.checkpoint_dir.glob("checkpoint_*.json"):
            # Parse filename
            parts = filepath.stem.split("_")
            if len(parts) >= 2:
                try:
                    step = int(parts[1])
                    ts = int(parts[2]) if len(parts) > 2 else 0

                    # Check if already in memory list
                    if not any(c["step_number"] == step for c in checkpoints):
                        checkpoints.append(
                            {
                                "step_number": step,
                                "timestamp": ts,
                                "in_memory": False,
                                "path": str(filepath),
                            }
                        )
                except ValueError:
                    continue

        return sorted(checkpoints, key=lambda x: x["step_number"])

    def _cleanup_old_files(self):
        """Remove old checkpoint files."""
        all_files = sorted(
            self.checkpoint_dir.glob("checkpoint_*.json"), key=lambda p: p.stat().st_mtime
        )

        # Keep only max_checkpoints most recent
        if len(all_files) > self.max_checkpoints:
            for old_file in all_files[: -self.max_checkpoints]:
                old_file.unlink()
                if self.verbose:
                    print(f"Removed old checkpoint: {old_file}")

    def clear_all(self):
        """Remove all checkpoints."""
        self.checkpoints = []

        for filepath in self.checkpoint_dir.glob("checkpoint_*.json"):
            filepath.unlink()

        if self.verbose:
            print("All checkpoints cleared")
