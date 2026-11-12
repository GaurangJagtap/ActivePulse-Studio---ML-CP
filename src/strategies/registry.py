"""Registry for all Active Learning query strategies."""

from typing import Dict, Type
from src.strategies.base import BaseStrategy
from src.strategies.random import RandomSamplingStrategy
from src.strategies.uncertainty import LeastConfidenceStrategy, MarginStrategy, EntropyStrategy
from src.strategies.bald import BALDStrategy
from src.strategies.coreset import CoreSetStrategy
from src.strategies.badge import BADGEStrategy
from src.strategies.hybrid import HybridStrategy
from src.strategies.cold_start import ColdStartClusteringStrategy

_STRATEGY_REGISTRY: Dict[str, Type[BaseStrategy]] = {
    "random": RandomSamplingStrategy,
    "least_confidence": LeastConfidenceStrategy,
    "margin": MarginStrategy,
    "entropy": EntropyStrategy,
    "bald": BALDStrategy,
    "coreset": CoreSetStrategy,
    "badge": BADGEStrategy,
    "hybrid": HybridStrategy,
    "cold_start_kmeans": ColdStartClusteringStrategy,
}


def register_strategy(name: str, strategy_cls: Type[BaseStrategy]) -> None:
    """Register a new query strategy class."""
    _STRATEGY_REGISTRY[name.lower()] = strategy_cls


def get_strategy(name: str, seed: int = 42, **kwargs) -> BaseStrategy:
    """Instantiate a query strategy by name."""
    name_lower = name.lower()
    if name_lower not in _STRATEGY_REGISTRY:
        raise ValueError(
            f"Strategy '{name}' not recognized. Registered strategies: {list(_STRATEGY_REGISTRY.keys())}"
        )
    return _STRATEGY_REGISTRY[name_lower](seed=seed, **kwargs)


def list_registered_strategies() -> list:
    return list(_STRATEGY_REGISTRY.keys())
