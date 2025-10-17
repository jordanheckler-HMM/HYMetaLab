"""sim package initialization"""

__all__ = ["agent", "io_utils", "validate_run"]
"""Consciousness simulation package."""

from .agents import Agent
from .config import Config
from .culture import CultureSystem
from .experiments import create_zip_bundle, run_sim
from .innovations import InnovationSystem
from .logging_io import Logger
from .workspace import Workspace
from .world import World

__version__ = "1.0.0"
__author__ = "Consciousness Research Team"

__all__ = [
    "Config",
    "World",
    "Agent",
    "Workspace",
    "InnovationSystem",
    "CultureSystem",
    "Logger",
    "run_sim",
    "create_zip_bundle",
]
