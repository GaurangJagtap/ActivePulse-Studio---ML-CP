"""Catalog endpoints for PyTorch models and Active Learning query strategies."""

from typing import List
from fastapi import APIRouter

from backend.api.schemas.common import ModelInfo, StrategyInfo
from backend.api.services.catalog_service import get_models_catalog, get_strategies_catalog

router = APIRouter(tags=["Catalog"])


@router.get("/models", response_model=List[ModelInfo], summary="List available PyTorch architectures")
def list_models() -> List[ModelInfo]:
    return get_models_catalog()


@router.get("/strategies", response_model=List[StrategyInfo], summary="List registered Active Learning strategies")
def list_strategies() -> List[StrategyInfo]:
    return get_strategies_catalog()

