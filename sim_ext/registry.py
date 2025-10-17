"""
Registry for all simulation modules.
"""

import importlib
from collections.abc import Callable

# Module registry
MODULES: dict[str, Callable] = {}


def register_module(name: str, func: Callable) -> None:
    """Register a simulation module."""
    MODULES[name] = func


def get_module(name: str) -> Callable:
    """Get a registered module."""
    if name not in MODULES:
        raise KeyError(f"Module '{name}' not found in registry")
    return MODULES[name]


def list_modules() -> list:
    """List all registered modules."""
    return list(MODULES.keys())


# Auto-register modules on import
def _auto_register():
    """Auto-register all available modules."""
    module_names = [
        "energy_thermo",
        "agent_health",
        "disease_epidemic",
        "info_layer",
        "ethics_norms",
        "multiscale",
        "phenomenology",
        "self_modeling",
        "uq_sensitivity",
        "bayes_infer",
    ]

    for name in module_names:
        try:
            module = importlib.import_module(f".{name}", package="sim_ext")
            if hasattr(module, "run"):
                register_module(name, module.run)
        except ImportError:
            pass  # Module not available


# Initialize registry
_auto_register()
