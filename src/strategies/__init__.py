"""Strategies module exports."""

from src.strategies.base import BaseStrategy
from src.strategies.random import RandomSamplingStrategy
from src.strategies.uncertainty import LeastConfidenceStrategy, MarginStrategy, EntropyStrategy
from src.strategies.bald import BALDStrategy
from src.strategies.coreset import CoreSetStrategy
from src.strategies.badge import BADGEStrategy
from src.strategies.hybrid import HybridStrategy
from src.strategies.cold_start import ColdStartClusteringStrategy
from src.strategies.registry import register_strategy, get_strategy, list_registered_strategies

__all__ = [
    "BaseStrategy",
    "RandomSamplingStrategy",
    "LeastConfidenceStrategy",
    "MarginStrategy",
    "EntropyStrategy",
    "BALDStrategy",
    "CoreSetStrategy",
    "BADGEStrategy",
    "HybridStrategy",
    "ColdStartClusteringStrategy",
    "register_strategy",
    "get_strategy",
    "list_registered_strategies",
]
